import sys

with open('main.py', 'r', encoding='utf-8') as f:
    content = f.read()

start_idx = content.find('class MMAFramework:')
end_idx = content.find('def main(', start_idx)

if start_idx == -1 or end_idx == -1:
    print("Could not find boundaries")
    sys.exit(1)

new_mma_framework = """class MMAFramework:
    \"\"\"Manages the integrated MagiNet model with shared AMSTenc and 4 MASTdec branches.\"\"\"

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
        patience_limit=20,
        log_strategy_steps=True,
    ):
        self.device = device
        self.delta = float(delta)
        self.patience_limit = int(patience_limit)
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
            "  - Model architecture unified: Shared Adaptive Mask Spatio-Temporal Encoder feeding 4 decoders."
        )
        for name, recon_mode, reg_mode in self.VARIANT_LAYOUT:
            lambda_val = l1_lambda if reg_mode == "l1" else l2_lambda
            print(
                f"    • {name}: {recon_mode.upper()} reconstruction + {reg_mode.upper()} regularization (λ = {lambda_val})"
            )
        print(
            "  - Validation RMSE S^t(n) is tracked per branch, accumulated into A^t(n) = Σ_h S^t(h),"
        )
        print(
            "    and adaptive weights ε^t(n) = softmax(-δ · A^t(n)) with δ = {} are refreshed every epoch.".format(
                self.delta
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
                loss = recon_loss + variant["reg_lambda"] * reg_term
                variant["epoch_losses"].append(loss.item())
                total_loss = total_loss + loss

            enc_reg = self._compute_regularization(self.model.AMSTenc, "l2")
            total_loss = total_loss + 1e-4 * enc_reg
            
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
        print(f"[MMAFramework] Epoch {epoch_idx + 1}")
        for idx, variant in enumerate(self.variants):
            print(
                "  {} | train loss = {:.6f} | S^t(n) = {:.6f} | A^t(n) = {:.6f} | ε^t(n) = {:.6f} | Val MAE = {:.6f}".format(
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
            def _init_stats():
                return {
                    "sum": 0.0,
                    "sq_sum": 0.0,
                    "count": 0,
                    "min": float("inf"),
                    "max": float("-inf"),
                }

            missing_stats = {variant["name"]: _init_stats() for variant in self.variants}
            full_stats = {variant["name"]: _init_stats() for variant in self.variants}

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
                else:
                    missing_bool = None

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
                        flat_vals = unnorm_variant.reshape(-1)
                        stats = full_stats[variant["name"]]
                        stats["sum"] += float(np.sum(flat_vals))
                        stats["sq_sum"] += float(np.sum(flat_vals ** 2))
                        stats["count"] += flat_vals.size
                        stats["min"] = min(stats["min"], float(np.min(flat_vals)))
                        stats["max"] = max(stats["max"], float(np.max(flat_vals)))
                        if missing_bool is not None and np.any(missing_bool):
                            miss_vals = unnorm_variant[missing_bool].reshape(-1)
                            miss_stats = missing_stats[variant["name"]]
                            miss_stats["sum"] += float(np.sum(miss_vals))
                            miss_stats["sq_sum"] += float(np.sum(miss_vals ** 2))
                            miss_stats["count"] += miss_vals.size
                            miss_stats["min"] = min(
                                miss_stats["min"],
                                float(np.min(miss_vals)) if miss_vals.size else miss_stats["min"],
                            )
                            miss_stats["max"] = max(
                                miss_stats["max"],
                                float(np.max(miss_vals)) if miss_vals.size else miss_stats["max"],
                            )

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
            result["missing_stats"] = missing_stats
            result["full_stats"] = full_stats

        return result

"""

new_content = content[:start_idx] + new_mma_framework + content[end_idx:]

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Patching complete!")
