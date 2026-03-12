"""
Description: 
Author: Jianping Zhou
Email: jianpingzhou0927@gmail.com
Date: 2023-08-23 17:56:01
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pk
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import argparse
import wandb
import time
import yaml

from utils import *
from models.model import MagiNet
from load_data import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path", type=str, default="configs/beijingPM2.5.yaml", help="config filepath"
)
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--learnable", type=int, default=1, help="1: learnable; 0: fixed")
parser.add_argument("--miss_ratio", type=float, default=None, help="override missing ratio")
args = parser.parse_args()


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)


class MMAFramework:
    """Manages the integrated MagiNet model with 4 independent branches (AMSTenc + MASTdec)."""

    VARIANT_LAYOUT = (
        ("MMA-1", "l1", "l1"),
        ("MMA-2", "l1", "l2"),
        ("MMA-3", "l2", "l1"),
        ("MMA-4", "l2", "l2"),
    )

    def __init__(
        self,
        device,
        base_model_kwargs,
        lr,
        delta=0.5,
        l1_lambda=1e-4,
        l2_lambda=1e-4,
        variant_lambdas=None,
        patience_limit=20,
        include_validation_in_training=False,
        selection_split="valid",
        log_strategy_steps=True,
    ):
        self.device = device
        self.delta = float(delta)
        self.patience_limit = int(patience_limit)
        self.include_validation_in_training = bool(include_validation_in_training)
        self.selection_split = str(selection_split)
        self.log_strategy_steps = log_strategy_steps
        self.eps = 1e-8
        self.variants = []
        self.accumulated_scores = torch.zeros(len(self.VARIANT_LAYOUT), dtype=torch.float32)
        self.current_weights = torch.ones(len(self.VARIANT_LAYOUT), dtype=torch.float32) / len(self.VARIANT_LAYOUT)
        self.best_weights = self.current_weights.clone()
        self.best_state = None
        self.best_ensemble_score = float("inf")
        self.best_epoch = -1
        self.completed_epochs = 0
        self.patience = 0
        self.history = {
            "train_loss": [],
            "val_rmse": [],
            "weights": [],
        }

        base_kwargs = dict(base_model_kwargs)
        base_kwargs.setdefault("log_strategy_steps", log_strategy_steps)

        # ONE unified model
        self.model = MagiNet(**base_kwargs).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for idx, (name, recon_mode, reg_mode) in enumerate(self.VARIANT_LAYOUT):
            if variant_lambdas is not None:
                reg_lambda = float(variant_lambdas[idx])
            else:
                reg_lambda = float(l1_lambda) if reg_mode == "l1" else float(l2_lambda)
            self.variants.append(
                {
                    "name": name,
                    "recon_mode": recon_mode,
                    "reg_mode": reg_mode,
                    "reg_lambda": reg_lambda,
                    "last_train_loss": float("inf"),
                }
            )

        if self.log_strategy_steps:
            self._print_framework_overview(l1_lambda, l2_lambda)

    def _print_framework_overview(self, l1_lambda, l2_lambda):
        print("[MMAFramework] Step 4: Adaptive MMA Ensemble Controller")
        print(
            "  - Model architecture: 4 independent branches, each with its own Encoder and Decoder."
        )
        for idx, (name, recon_mode, reg_mode) in enumerate(self.VARIANT_LAYOUT):
            lambda_val = self.variants[idx]["reg_lambda"]
            print(
                f"    • {name}: {recon_mode.upper()} reconstruction + {reg_mode.upper()} regularization (λ = {lambda_val})"
            )
        print(
            "  - Selection RMSE S^t(n) is tracked per branch, accumulated into A^t(n) = Σ_h S^t(h),"
        )
        print(
            "    using the {} split; adaptive weights ε^t(n) = softmax(-δ · A^t(n)) with δ = {} are refreshed every epoch.".format(
                self.selection_split,
                self.delta,
            )
        )
        print(
            "  - Ensemble predictions are late-fused as x̂ = Σ_t ε^t · x̂^t."
        )

    def _compute_reconstruction_loss(self, preds, target, mask, mode):
        mask = mask.to(preds.dtype)
        denom = torch.clamp(mask.sum(), min=1.0)
        diff = preds - target
        if mode == "l1":
            loss = torch.sum(torch.abs(diff) * mask) / denom
        else:
            mse = torch.sum((diff * mask) ** 2) / denom
            loss = torch.sqrt(mse + self.eps)
        return loss

    def _compute_regularization(self, model_part, mode):
        reg_value = torch.tensor(0.0, device=self.device)
        for param in model_part.parameters():
            if mode == "l1":
                reg_value = reg_value + torch.sum(torch.abs(param))
            else:
                reg_value = reg_value + 0.5 * torch.sum(param ** 2)
        return reg_value

    def _snapshot_state(self, model):
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def restore_best_states(self):
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            self.model.to(self.device)
        self.current_weights = self.best_weights.clone()

    def _train_one_epoch(self, train_loader):
        self.model.train()
        for variant in self.variants:
            variant["epoch_losses"] = []

        for _, (x, m, y) in enumerate(train_loader):
            x = x.to(self.device)
            m = m.to(self.device)
            target = y[:, :, :1, :].to(self.device)
            observed_mask = m[:, :, :1, :].to(self.device)

            self.optimizer.zero_grad()
            preds_list = self.model(x, m)
            
            total_loss = 0.0
            
            for idx, variant in enumerate(self.variants):
                recon_loss = self._compute_reconstruction_loss(
                    preds_list[idx], target, observed_mask, variant["recon_mode"]
                )
                reg_term = self._compute_regularization(
                    self.model.MASTdec_group[idx], variant["reg_mode"]
                )
                enc_reg = self._compute_regularization(
                    self.model.AMSTenc_group[idx], "l2"
                )
                loss = recon_loss + variant["reg_lambda"] * reg_term + 1e-4 * enc_reg
                variant["epoch_losses"].append(loss.item())
                total_loss = total_loss + loss

            total_loss.backward()
            self.optimizer.step()

        mean_losses = []
        for variant in self.variants:
            if variant["epoch_losses"]:
                variant["last_train_loss"] = float(np.mean(variant["epoch_losses"]))
            else:
                variant["last_train_loss"] = float("nan")
            mean_losses.append(variant["last_train_loss"])
        return mean_losses

    def _train_one_epoch_with_validation(self, train_loader, valid_loader):
        self.model.train()
        for variant in self.variants:
            variant["epoch_losses"] = []

        combined_batches = list(train_loader) + list(valid_loader)

        for _, (x, m, y) in enumerate(combined_batches):
            x = x.to(self.device)
            m = m.to(self.device)
            target = y[:, :, :1, :].to(self.device)
            observed_mask = m[:, :, :1, :].to(self.device)

            self.optimizer.zero_grad()
            preds_list = self.model(x, m)

            total_loss = 0.0

            for idx, variant in enumerate(self.variants):
                recon_loss = self._compute_reconstruction_loss(
                    preds_list[idx], target, observed_mask, variant["recon_mode"]
                )
                reg_term = self._compute_regularization(
                    self.model.MASTdec_group[idx], variant["reg_mode"]
                )
                enc_reg = self._compute_regularization(
                    self.model.AMSTenc_group[idx], "l2"
                )
                loss = recon_loss + variant["reg_lambda"] * reg_term + 1e-4 * enc_reg
                variant["epoch_losses"].append(loss.item())
                total_loss = total_loss + loss

            total_loss.backward()
            self.optimizer.step()

        mean_losses = []
        for variant in self.variants:
            if variant["epoch_losses"]:
                variant["last_train_loss"] = float(np.mean(variant["epoch_losses"]))
            else:
                variant["last_train_loss"] = float("nan")
            mean_losses.append(variant["last_train_loss"])
        return mean_losses

    def _evaluate_validation(self, valid_loader):
        sq_errors = [0.0 for _ in self.variants]
        abs_errors = [0.0 for _ in self.variants]
        counts = [0.0 for _ in self.variants]

        self.model.eval()

        with torch.no_grad():
            for _, (x, m, y) in enumerate(valid_loader):
                x = x.to(self.device)
                m = m.to(self.device)
                target = y[:, :, :1, :].to(self.device)
                observed_mask = m[:, :, :1, :].to(self.device)

                preds_list = self.model(x, m)
                for idx, variant in enumerate(self.variants):
                    preds = preds_list[idx]
                    diff = (preds - target) * observed_mask
                    sq_errors[idx] += torch.sum(diff ** 2).item()
                    abs_errors[idx] += torch.sum(torch.abs(diff)).item()
                    counts[idx] += torch.sum(observed_mask).item()

        rmse_values = []
        mae_values = []
        for idx in range(len(self.variants)):
            denom = max(counts[idx], self.eps)
            rmse = float(np.sqrt(sq_errors[idx] / denom))
            mae = float(abs_errors[idx] / denom)
            rmse_values.append(rmse)
            mae_values.append(mae)

        val_tensor = torch.tensor(rmse_values, dtype=torch.float32)
        return val_tensor, mae_values

    def _update_weights(self, val_scores):
        self.accumulated_scores += val_scores
        weights = torch.softmax(-self.delta * self.accumulated_scores, dim=0)
        self.current_weights = weights
        ensemble_score = torch.sum(weights * val_scores).item()

        improved = False
        if ensemble_score + 1e-8 < self.best_ensemble_score:
            self.best_ensemble_score = ensemble_score
            self.best_weights = weights.clone()
            self.best_state = self._snapshot_state(self.model)
            self.best_epoch = self.completed_epochs
            self.patience = 0
            improved = True
        else:
            self.patience += 1

        self.history["val_rmse"].append(val_scores.tolist())
        self.history["weights"].append(weights.tolist())
        return ensemble_score, improved

    def train(self, train_loader, valid_loader, epochs):
        for epoch in range(epochs):
            start_time = time.time()
            if self.include_validation_in_training:
                train_losses = self._train_one_epoch_with_validation(train_loader, valid_loader)
            else:
                train_losses = self._train_one_epoch(train_loader)
            val_scores, val_mae = self._evaluate_validation(valid_loader)
            self.completed_epochs = epoch + 1
            ensemble_score, improved = self._update_weights(val_scores)
            self.history["train_loss"].append(train_losses)
            epoch_time = time.time() - start_time

            self._log_epoch(epoch, train_losses, val_scores, val_mae, ensemble_score, epoch_time, improved)

            if self.patience > self.patience_limit:
                print("[MMAFramework] Early Stop triggered due to no ensemble improvement.")
                break

        if self.best_state is None:
            self.best_state = self._snapshot_state(self.model)
            self.best_weights = self.current_weights.clone()

    def _log_epoch(
        self,
        epoch_idx,
        train_losses,
        val_scores,
        val_mae,
        ensemble_score,
        epoch_time,
        improved,
    ):
        acc_np = self.accumulated_scores.detach().cpu().numpy()
        weights_np = self.current_weights.detach().cpu().numpy()
        val_np = val_scores.detach().cpu().numpy()
        print(f"[MMAFramework] Epoch {epoch_idx + 1} - Selection Split: {self.selection_split}")
        for idx, variant in enumerate(self.variants):
            print(
                "  {} | train loss = {:.6f} | S^t(n) = {:.6f} | A^t(n) = {:.6f} | ε^t(n) = {:.6f} | Selection MAE = {:.6f}".format(
                    variant["name"],
                    train_losses[idx],
                    val_np[idx],
                    acc_np[idx],
                    weights_np[idx],
                    val_mae[idx],
                )
            )
        print(
            "  Ensemble RMSE = {:.6f} | Patience {}/{} | Duration {:.2f}s".format(
                ensemble_score, self.patience, self.patience_limit, epoch_time
            )
        )
        if improved:
            print(
                "    ✓ Improved ensemble performance. Snapshotting states at epoch {}.".format(
                    epoch_idx + 1
                )
            )

    def save_best_models(self, save_root, dataset, miss_mechanism, miss_ratio, seed):
        self.restore_best_states()
        target_dir = os.path.join(save_root, f"{dataset}")
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        path = os.path.join(
            target_dir,
            f"unified_best_model_{miss_mechanism}_ms{miss_ratio}_seed{seed}.pth",
        )
        torch.save(self.best_state, path)

        weight_path = os.path.join(
            target_dir,
            f"mma_weights_{miss_mechanism}_ms{miss_ratio}_seed{seed}.pt",
        )
        torch.save(self.best_weights, weight_path)
        print("[MMAFramework] Saved best checkpoints:")
        print(f"  - {path}")
        print(f"  - Adaptive weight trace: {weight_path}")
        return [path], weight_path

    def predict_ensemble(
        self,
        test_loader,
        mean,
        std,
        dataset,
        miss_ratio,
        miss_mechanism,
        seed,
        result_root,
        collect_outputs=False,
    ):
        self.restore_best_states()
        weights = self.best_weights.detach().cpu().numpy()

        test_maes, test_rmses, test_mapes = [], [], []
        variant_metrics = {
            variant["name"]: {"mae": [], "rmse": [], "mape": []}
            for variant in self.variants
        }
        miss_data, predict_results, groundtruths = [], [], []

        if collect_outputs:
            missing_data_arrays = {variant["name"]: [] for variant in self.variants}
            observed_data_arrays = {variant["name"]: [] for variant in self.variants}
            full_data_arrays = {variant["name"]: [] for variant in self.variants}

        self.model.eval()

        with torch.no_grad():
            for _, (x, m, y) in enumerate(test_loader):
                x = x.to(self.device)
                m = m.to(self.device)

                preds_list = self.model(x, m)
                preds_per_variant = [p.detach().cpu().numpy() for p in preds_list]

                stacked_preds = np.stack(preds_per_variant, axis=0)
                weight_view = weights.reshape(-1, 1, 1, 1, 1)
                ensemble_pred = np.sum(stacked_preds * weight_view, axis=0)

                x_input = x[:, :, :1, :].detach().cpu().numpy()
                y_target = y[:, :, :1, :].detach().cpu().numpy()
                mask = m[:, :, :1, :].detach().cpu().numpy()
                if collect_outputs:
                    missing_bool = (1.0 - mask).astype(bool)
                    observed_bool = mask.astype(bool)
                else:
                    missing_bool = None
                    observed_bool = None

                unnorm_x = unnormalization(x_input, mean, std)
                unnorm_y = unnormalization(y_target, mean, std)
                unnorm_ensemble = unnormalization(ensemble_pred, mean, std)

                mae, rmse, mape = observed_eval_np(unnorm_ensemble, unnorm_y, mask)
                test_maes.append(mae)
                test_rmses.append(rmse)
                test_mapes.append(mape)

                for idx, variant in enumerate(self.variants):
                    unnorm_variant = unnormalization(stacked_preds[idx], mean, std)
                    v_mae, v_rmse, v_mape = observed_eval_np(unnorm_variant, unnorm_y, mask)
                    variant_metrics[variant["name"]]["mae"].append(v_mae)
                    variant_metrics[variant["name"]]["rmse"].append(v_rmse)
                    variant_metrics[variant["name"]]["mape"].append(v_mape)
                    if collect_outputs:
                        if missing_bool is not None and np.any(missing_bool):
                            miss_vals = unnorm_variant[missing_bool].reshape(-1)
                            missing_data_arrays[variant["name"]].append(miss_vals)
                        if observed_bool is not None and np.any(observed_bool):
                            observed_vals = unnorm_variant[observed_bool].reshape(-1)
                            observed_data_arrays[variant["name"]].append(observed_vals)
                        flat_vals = unnorm_variant.reshape(-1)
                        full_data_arrays[variant["name"]].append(flat_vals)

                predict_data = unnorm_ensemble * (1 - mask) + unnorm_x * mask
                unnorm_x = np.where(mask == 0, np.nan, unnorm_x)

                miss_data.append(unnorm_x)
                predict_results.append(predict_data)
                groundtruths.append(unnorm_y)

        test_mae = float(np.mean(test_maes))
        test_rmse = float(np.mean(test_rmses))
        test_mape = float(np.mean(test_mapes))
        print(
            "[MMAFramework] Test ensemble metrics — MAE {:.6f}, RMSE {:.6f}, MAPE {:.6f}".format(
                test_mae, test_rmse, test_mape * 100
            )
        )

        aggregated_variant_metrics = {}
        for name, metric_dict in variant_metrics.items():
            aggregated_variant_metrics[name] = {
                "mae": float(np.mean(metric_dict["mae"])) if metric_dict["mae"] else float("nan"),
                "rmse": float(np.mean(metric_dict["rmse"])) if metric_dict["rmse"] else float("nan"),
                "mape": float(np.mean(metric_dict["mape"])) if metric_dict["mape"] else float("nan"),
            }
            print(
                "    {} individual metrics — MAE {:.6f}, RMSE {:.6f}, MAPE {:.6f}".format(
                    name,
                    aggregated_variant_metrics[name]["mae"],
                    aggregated_variant_metrics[name]["rmse"],
                    aggregated_variant_metrics[name]["mape"] * 100,
                )
            )

        print(
            "    Final adaptive weights ε^t(n): {}".format(
                ", ".join(
                    "{} = {:.6f}".format(self.variants[idx]["name"], weights[idx])
                    for idx in range(len(weights))
                )
            )
        )

        result = {
            "missed_data": np.concatenate(miss_data, axis=0),
            "predict_results": np.concatenate(predict_results, axis=0),
            "groundtruths": np.concatenate(groundtruths, axis=0),
            "mae": test_mae,
            "rmse": test_rmse,
            "mape": test_mape,
            "variant_metrics": aggregated_variant_metrics,
        }

        if collect_outputs:
            result["missing_stats"] = {k: np.concatenate(v) if v else np.array([]) for k, v in missing_data_arrays.items()}
            result["observed_stats"] = {k: np.concatenate(v) if v else np.array([]) for k, v in observed_data_arrays.items()}
            result["full_stats"] = {k: np.concatenate(v) if v else np.array([]) for k, v in full_data_arrays.items()}

        return result

def predict(
    framework,
    result_path,
    test_loader,
    mean,
    std,
    dataset,
    miss_ratio,
    miss_mechanism,
    seed,
    collect_outputs=False,
):
    result = framework.predict_ensemble(
        test_loader,
        mean,
        std,
        dataset,
        miss_ratio,
        miss_mechanism,
        seed,
        result_path,
        collect_outputs=collect_outputs,
    )
    os.makedirs(os.path.join(result_path, dataset), exist_ok=True)
    res_file = os.path.join(result_path, dataset, f"result_{miss_mechanism}_ms{miss_ratio}_seed{seed}.pkl")
    with open(res_file, "wb") as f:
        pk.dump(result, f)
    print(f"[MMAFramework] Test results saved to {res_file}")
    return result

def main(args):
    if args.learnable == 1:
        learnable = True
    else:
        learnable = False
    config_filename = args.config_path
    with open(config_filename) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)

    dataset = config["data"]["dataset"]
    miss_mechanism = config["data"]["miss_mechanism"]
    # miss_pattern = config["data"]["miss_pattern"]
    miss_ratio = float(config["data"]["miss_ratio"])
    if args.miss_ratio is not None:
        miss_ratio = args.miss_ratio
    batch_size = int(config["data"]["batch_size"])
    val_batch_size = int(config["data"]["val_batch_size"])
    test_batch_size = int(config["data"]["test_batch_size"])
    selection_split = str(config["train"].get("selection_split", "valid")).lower()
    seqlen = int(config["model"]["seqlen"])
    num_nodes = int(config["model"]["num_nodes"])
    st_block = int(config["model"]["st_block"])
    in_channel = int(config["model"]["in_channel"])
    hidden_size = int(config["model"]["hidden_size"])
    K = int(config["model"]["K"])
    d_model = int(config["model"]["d_model"])
    n_heads = int(config["model"]["n_heads"])
    epochs = int(config["train"]["epochs"])
    lr = float(config["train"]["lr"])
    save_path = config["train"]["save_model_path"]
    result_path = config["train"]["result_path"]
    seed = args.seed

    device = torch.device(
        "cuda:{}".format(int(config["train"]["cuda"]))
        if torch.cuda.is_available()
        else "cpu"
    )

    seed_torch(seed)
    torch.set_num_threads(10)

    train_loader, valid_loader, test_loader, mean, std, A = generate_miss_loader(
        dataset,
        miss_mechanism,
        # miss_pattern,
        miss_ratio,
        seqlen,
        batch_size,
        val_batch_size,
        test_batch_size,
        selection_split=selection_split,
    )
    num_nodes = A.shape[0]

    # wandb.init(
    #     project="MagiNet",
    #     name="{}_lr{}_hiddensize{}_batchsize{}_seed{}".format(
    #         dataset, lr, hidden_size, batch_size, seed
    #     ),
    # )

    adj_mx = weight_matrix(A)

    log_strategy_steps = bool(config["train"].get("log_strategy_steps", True))

    base_model_kwargs = dict(
        device=device,
        num_nodes=num_nodes,
        seqlen=seqlen,
        in_channels=in_channel,
        hidden_dim=hidden_size,
        st_block=st_block,
        K=K,
        d_model=d_model,
        n_heads=n_heads,
        adj_mx=adj_mx,
        learnable=learnable,
        log_strategy_steps=log_strategy_steps,
    )

    mma_cfg = config["train"].get("mma", {})
    mma_delta = float(mma_cfg.get("delta", 0.5))
    default_lambda = float(mma_cfg.get("lambda", 1e-4))
    mma_l1_lambda = float(mma_cfg.get("lambda_l1", default_lambda))
    mma_l2_lambda = float(mma_cfg.get("lambda_l2", default_lambda))
    variant_lambdas = mma_cfg.get("variant_lambdas")
    if variant_lambdas is not None:
        if len(variant_lambdas) != 4:
            raise ValueError("train.mma.variant_lambdas must contain exactly 4 values.")
    patience_limit = int(config["train"].get("patience", 20))
    include_validation_in_training = bool(
        config["train"].get("include_validation_in_training", False)
    )

    framework = MMAFramework(
        device=device,
        base_model_kwargs=base_model_kwargs,
        lr=lr,
        delta=mma_delta,
        l1_lambda=mma_l1_lambda,
        l2_lambda=mma_l2_lambda,
        variant_lambdas=variant_lambdas,
        patience_limit=patience_limit,
        include_validation_in_training=include_validation_in_training,
        selection_split=selection_split,
        log_strategy_steps=log_strategy_steps,
    )

    framework.train(train_loader, valid_loader, epochs)

    if framework.best_epoch != -1:
        print(
            "[MMAFramework] Best ensemble epoch: {} (ensemble RMSE {:.6f})".format(
                framework.best_epoch, framework.best_ensemble_score
            )
        )

    framework.save_best_models(
        save_path,
        dataset,
        miss_mechanism,
        miss_ratio,
        seed,
    )

    # test
    predict(
        framework,
        result_path,
        test_loader,
        mean,
        std,
        dataset,
        miss_ratio,
        miss_mechanism,
        seed,
    )


if __name__ == "__main__":
    start_time = time.time()
    main(args)
    print("Spend Time: {}".format(time.time() - start_time))