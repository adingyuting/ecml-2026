from __future__ import annotations
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import csv
import json
import random
import time
import numpy as np
import torch

from .data import SignalScaler, load_dataset, write_json, digest_file, publish_file, write_npz
from .masks import make_split, contaminate
from .model import Coordinator, ModelConfig, UDCGL, head_objectives
from . import __version__


@dataclass
class RunConfig:
    data_root: str
    dataset: str = "D1"
    pattern: str = "random"
    seed: int = 0
    split_seed: int | None = 0
    test_fraction: float = 0.2
    val_fraction: float = 0.1
    dataset_layout: str | None = None
    block_length: int | None = None
    zero_is_missing: bool = True
    scaler: str = "standard"
    epochs: int = 1000
    patience: int = 25
    batch_size: int = 128
    window: int = 12
    stride: int | None = None
    width: int = 64
    depth: int = 3
    heads: int = 1
    cheb_order: int = 3
    context_span: int = 8
    kernels: tuple[int, ...] = (3, 5, 7)
    dropout: float = 0.0
    learning_rate: float = 0.005
    regularization: float = 1e-4
    delta: float = 0.03
    variant: str = "full"
    single_head: int = 0
    outlier_rate: float = 0.0
    outlier_mode: str = "inject_missing"
    outlier_scope: str = "global"
    max_steps: int | None = None
    max_nodes: int | None = None
    max_train_batches: int | None = None
    threads: int = 2
    device: str = "cpu"
    backbone: str = "paper"
    optimizer: str | None = None
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    sgd_momentum: float = 0.9
    grad_clip: float | None = 5.0

    def __post_init__(self):
        self.kernels = tuple(self.kernels)

    @classmethod
    def from_manifest(cls, settings):
        if settings.get("backbone") != "paper":
            raise ValueError("This run was not trained with paper UDCGL")
        return cls(**{item.name: settings[item.name] for item in fields(cls) if item.name in settings})

    @property
    def effective_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        return "adamw" if self.dataset in {"D3", "D4"} else "adam"

    def validate(self):
        if self.backbone != "paper":
            raise ValueError("Only the paper UDCGL architecture is supported")
        for name in ["epochs", "batch_size", "window", "width", "depth", "threads", "heads", "cheb_order"]:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ["max_steps", "max_nodes", "max_train_batches", "stride"]:
            if getattr(self, name) is not None and getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0 or self.regularization < 0 or self.patience < 0:
            raise ValueError("Invalid learning rate, regularization or patience")
        if self.context_span < 0 or self.seed < 0 or (self.split_seed is not None and self.split_seed < 0):
            raise ValueError("Context span and random seed must be nonnegative")
        if self.effective_optimizer not in {"adam", "adamw", "sgd"} or self.weight_decay < 0:
            raise ValueError("Invalid optimizer or weight decay")
        if not (0 <= self.adam_beta1 < 1 and 0 <= self.adam_beta2 < 1 and 0 <= self.sgd_momentum < 1):
            raise ValueError("Optimizer beta/momentum must be in [0,1)")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError("Gradient clipping must be positive or None")
        if not 0 < self.test_fraction < 1 or not 0 < self.val_fraction < 1:
            raise ValueError("Test and validation fractions must lie in (0,1)")
        if self.test_fraction + self.val_fraction >= 1:
            raise ValueError("Test and validation fractions must leave fitting entries")
        if self.pattern not in {"random", "block", "sensor"}:
            raise ValueError("Unknown missingness pattern")
        if self.variant not in {"full", "zero_fill", "single_head", "static_average"}:
            raise ValueError("Unknown variant")
        if self.outlier_mode not in {"replace_observed", "inject_missing"}:
            raise ValueError("Unknown outlier mode")
        if self.outlier_scope not in {"global", "sensor"}:
            raise ValueError("Outlier scope must be global or sensor")


def window_starts(steps, window, stride=None):
    width = min(window, steps)
    stride = stride or width
    if not 1 <= stride <= width:
        raise ValueError("Stride must lie in 1..window to cover every time step")
    starts = list(range(0, steps - width + 1, stride))
    if starts[-1] != steps - width:
        starts.append(steps - width)
    return starts, width


def batch_windows(values, mask, starts, width, device):
    x = np.stack([values[:, s:s + width].T for s in starts])
    m = np.stack([mask[:, s:s + width].T for s in starts])
    times = np.stack([np.arange(s, s + width) for s in starts])
    return (torch.as_tensor(x, dtype=torch.float32, device=device),
            torch.as_tensor(m, dtype=torch.bool, device=device),
            torch.as_tensor(times, dtype=torch.long, device=device))


@torch.no_grad()
def predict_heads(model, input_values, input_mask, window=64, batch_size=4, device="cpu", stride=None):
    model.eval()
    nodes, steps = input_values.shape
    starts, width = window_starts(steps, window, stride)
    sums = np.zeros((len(model.decoders), nodes, steps), dtype=np.float64)
    counts = np.zeros(steps, dtype=np.int64)
    for offset in range(0, len(starts), batch_size):
        selected = starts[offset:offset + batch_size]
        x, m, times = batch_windows(input_values, input_mask, selected, width, device)
        pred = model(x, m, times).cpu().numpy()
        if not np.isfinite(pred).all():
            raise FloatingPointError("Nonfinite prediction")
        for j, start in enumerate(selected):
            sums[:, :, start:start + width] += pred[:, j].transpose(0, 2, 1)
            counts[start:start + width] += 1
    if not counts.all():
        raise ValueError("Prediction windows did not cover every time step")
    return (sums / counts[None, None, :]).astype(np.float32)


def metrics(predictions, reference, mask):
    if not mask.any():
        raise ValueError("No evaluation targets")
    residual = np.asarray(predictions, dtype=np.float64)[mask] - np.asarray(reference, dtype=np.float64)[mask]
    if not np.isfinite(residual).all():
        raise FloatingPointError("Nonfinite evaluation residual")
    return {"mae": float(np.abs(residual).mean()), "rmse": float(np.sqrt(np.square(residual).mean())),
            "count": int(mask.sum())}


def set_seed(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=True)


def load_run_dataset(config):
    layout = None
    if config.dataset_layout:
        from .data import load_layout
        layout = load_layout(config.dataset_layout)
    return load_dataset(config.data_root, config.dataset, config.zero_is_missing,
                        config.max_steps, config.max_nodes, layout)


def make_run_split(config, data):
    split = make_split(data.observed, config.test_fraction, config.val_fraction, config.pattern,
                       config.seed if config.split_seed is None else config.split_seed,
                       config.block_length)
    split.metadata["model_seed"] = config.seed
    return split


def train_run(config: RunConfig, output: str | Path, resume=False, progress=True):
    config.validate()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Run directory is not empty: {output}. Choose another directory or --resume.")
    if resume and not (output / "last.pt").exists():
        raise FileNotFoundError("--resume requires last.pt")
    output.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed, config.threads)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this Python environment")
    data = load_run_dataset(config)
    split = make_run_split(config, data)
    scaler = SignalScaler.fit(data.signal, split.fit, config.scaler)
    fit_values, input_mask, corruption_mask = contaminate(data.signal, data.observed, split,
                                                          config.outlier_rate, config.outlier_mode, config.seed + 101,
                                                          config.outlier_scope)
    input_values = np.zeros_like(data.signal, dtype=np.float32)
    input_values[input_mask] = scaler.transform(fit_values[input_mask]).astype(np.float32)
    reference_normalized = scaler.transform(data.signal)
    model_config = ModelConfig(nodes=data.signal.shape[0], steps=data.signal.shape[1], width=config.width,
                               depth=config.depth, heads=config.heads,
                               cheb_order=config.cheb_order, context_span=config.context_span,
                               kernels=tuple(config.kernels), dropout=config.dropout,
                               variant=config.variant, single_head=config.single_head)
    model = UDCGL(model_config, data.graph).to(config.device)
    optimizer_options = dict(lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer_name = config.effective_optimizer
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), momentum=config.sgd_momentum, **optimizer_options)
    else:
        cls = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
        optimizer = cls(model.parameters(), betas=(config.adam_beta1, config.adam_beta2), **optimizer_options)
    coordinator = Coordinator(len(model.decoders), config.delta, config.variant == "static_average")
    is_smoke = bool(config.max_steps or config.max_nodes or config.max_train_batches)
    manifest = {"implementation_version": __version__,
                "run_config": asdict(config), "model_config": model_config.to_dict(), "data": data.info,
                "split": split.metadata, "scaler": scaler.to_dict(), "is_smoke_test": is_smoke,
                "interpretation": "Implementation validation only; not publication results" if is_smoke else "New experiment, not reproduction of an existing paper table",
                "coordination": {"iteration_unit": "epoch", "loss": "MSE in fitting-scaled units; not automatically bounded in [0,1]",
                                 "timing": "weights from preceding validation epochs; then observe current losses"},
                "selection": "same validation subset for coordination and checkpoint selection; test scored after selection",
                "torch_version": str(torch.__version__), "parameters": sum(p.numel() for p in model.parameters()),
                "backbone": config.backbone, "effective_optimizer": optimizer_name,
                "outlier_count": int(corruption_mask.sum()), "input_count": int(input_mask.sum())}
    manifest["independent_test"] = split.metadata["independent_test"]
    best_val, best_epoch, stale, first_epoch = float("inf"), 0, 0, 1
    history = []
    if resume:
        old_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if old_manifest.get("implementation_version") != __version__:
            raise ValueError("Checkpoint implementation version differs; use a new run directory")
        old_cfg = asdict(RunConfig.from_manifest(old_manifest["run_config"]))
        new_cfg = asdict(config)
        for name in ["epochs", "patience"]:
            old_cfg.pop(name, None)
            new_cfg.pop(name, None)
        if json.dumps(old_cfg, sort_keys=True) != json.dumps(new_cfg, sort_keys=True):
            raise ValueError("Resume configuration differs from the original run")
        if old_manifest["data"]["data_sha256"] != data.info["data_sha256"]:
            raise ValueError("Source data changed since checkpoint")
        if old_manifest["data"]["graph_sha256"] != data.info["graph_sha256"]:
            raise ValueError("Source graph changed since checkpoint")
        with np.load(output / "inputs.npz", allow_pickle=False) as previous:
            for name, mask in [("fit_mask", split.fit), ("validation_mask", split.validation), ("test_mask", split.test)]:
                if not np.array_equal(previous[name], mask):
                    raise ValueError("Resume split masks differ from the original run; start a new run")
        saved = _load_checkpoint(output / "last.pt", config.device)
        if config.epochs < saved["epoch"]:
            raise ValueError("Cannot resume with fewer total epochs than already completed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        coordinator.cumulative = saved["cumulative"].cpu().double()
        first_epoch, best_epoch = saved["epoch"] + 1, saved["best_epoch"]
        best_val, stale = saved["best_val"], saved["stale"]
        torch.set_rng_state(saved["rng_state"].cpu())
        if config.device.startswith("cuda") and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
        history = saved.get("history")
        if history is None:
            history = json.loads((output / "history.json").read_text(encoding="utf-8"))
        history = [row for row in history if row["epoch"] <= saved["epoch"]]
        if [row["epoch"] for row in history] != list(range(1, saved["epoch"] + 1)):
            raise ValueError("Checkpoint/history epochs are inconsistent")
    write_json(output / "manifest.json", manifest)
    write_json(output / "history.json", history)
    write_npz(output / "inputs.npz", input_values=input_values, input_mask=input_mask,
                        original_known=data.observed, fit_mask=split.fit, validation_mask=split.validation,
                        test_mask=split.test, corruption_mask=corruption_mask, adjacency=data.graph)
    write_npz(output / "split_masks.npz", fit_mask=split.fit, validation_mask=split.validation,
                        test_mask=split.test)
    np.savetxt(output / "adjacency.csv", data.graph, delimiter=",", fmt="%.9g")
    starts, width = window_starts(data.signal.shape[1], config.window, config.stride)
    clock_start = time.perf_counter()
    already_stopped = bool(config.patience and stale >= config.patience)
    epochs_to_run = () if already_stopped else range(first_epoch, config.epochs + 1)
    for epoch in epochs_to_run:
        model.train()
        weights_before = coordinator.weights().numpy().copy()
        order = np.random.default_rng(config.seed + 1009 * epoch).permutation(starts)
        train_losses, used_batches = [], 0
        for offset in range(0, len(order), config.batch_size):
            if config.max_train_batches is not None and used_batches >= config.max_train_batches:
                break
            x, mask, times = batch_windows(input_values, input_mask, order[offset:offset + config.batch_size], width, config.device)
            if not bool(mask.any()):
                continue
            optimizer.zero_grad(set_to_none=True)
            predictions = model(x, mask, times)
            losses = head_objectives(model, predictions, x, mask, config.regularization)
            loss = losses.mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training objective")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip if config.grad_clip is not None else float("inf"), error_if_nonfinite=True)
            optimizer.step()
            train_losses.append(losses.detach().cpu().numpy())
            used_batches += 1
        if not used_batches:
            raise ValueError("No training batches contain fitting observations")
        head_predictions = predict_heads(model, input_values, input_mask, config.window, config.batch_size,
                                          config.device, config.stride)
        validation_losses = np.array([np.square(p[split.validation] - reference_normalized[split.validation]).mean()
                                      for p in head_predictions], dtype=np.float64)
        aggregate = np.einsum("j,jnt->nt", weights_before, head_predictions)
        val_mse = float(np.square(aggregate[split.validation] - reference_normalized[split.validation]).mean())
        if not np.isfinite(val_mse):
            raise FloatingPointError("Nonfinite validation objective")
        improved = val_mse < best_val
        if improved:
            best_val, best_epoch, stale = val_mse, epoch, 0
        else:
            stale += 1
        coordinator.observe(validation_losses)
        row = {"epoch": epoch, "train_objectives": np.mean(train_losses, axis=0).tolist(),
               "head_validation_mse": validation_losses.tolist(), "ensemble_validation_mse": val_mse,
               "weights_used": weights_before.tolist(), "weights_next": coordinator.weights().tolist(),
               "cumulative_losses": coordinator.cumulative.tolist(), "best_epoch": best_epoch,
               "batches": used_batches, "last_gradient_norm": float(grad_norm),
               "elapsed_seconds": time.perf_counter() - clock_start}
        history.append(row)
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "model_config": model_config.to_dict(), "adjacency": torch.from_numpy(data.graph),
                   "weights": torch.from_numpy(weights_before), "cumulative": coordinator.cumulative,
                   "epoch": epoch, "best_val": best_val, "best_epoch": best_epoch, "stale": stale,
                   "rng_state": torch.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if config.device.startswith("cuda") else None,
                   "history": history}
        if improved:
            with publish_file(output / "best.pt") as stream:
                torch.save(payload, stream)
        with publish_file(output / "last.pt") as stream:
            torch.save(payload, stream)
        write_json(output / "history.json", history)
        if progress:
            print(f"{config.dataset}/{config.pattern} epoch={epoch} val_mse={val_mse:.6g} "
                  f"weights={np.round(weights_before, 4).tolist()} best={best_epoch}", flush=True)
        if config.patience and stale >= config.patience:
            break
    if not (output / "best.pt").exists():
        raise RuntimeError("No valid checkpoint was produced")
    best = _load_checkpoint(output / "best.pt", config.device)
    model.load_state_dict(best["model"])
    scaled_heads = predict_heads(model, input_values, input_mask, config.window, config.batch_size,
                                 config.device, config.stride)
    head_raw = scaler.inverse(scaled_heads)
    final_weights = best["weights"].cpu().numpy()
    prediction = np.einsum("j,jnt->nt", final_weights, head_raw)
    completed = prediction.copy()
    completed[input_mask] = fit_values[input_mask]
    write_npz(output / "predictions.npz", prediction=prediction.astype(np.float32),
                        completed_input=completed.astype(np.float32), heads=head_raw.astype(np.float32),
                        weights=final_weights)
    result = {"dataset": config.dataset, "pattern": config.pattern, "seed": config.seed,
              "independent_test": split.metadata["independent_test"],
              "evaluation_role": split.metadata["evaluation_role"],
              "backbone": config.backbone,
              "variant": config.variant, "is_smoke_test": is_smoke, "best_epoch": int(best["epoch"]),
              "weights": final_weights.tolist(), "test": metrics(prediction, data.signal, split.test),
              "validation": metrics(prediction, data.signal, split.validation),
              "head_test": [metrics(p, data.signal, split.test) for p in head_raw],
              "realized_fractions": split.metadata["realized_fractions"],
              "seconds_this_invocation": time.perf_counter() - clock_start}
    write_json(output / "metrics.json", result)
    with (output / "history.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "validation_mse", "best_epoch", "weights_used", "weights_next", "seconds"])
        for row in history:
            writer.writerow([row["epoch"], row["ensemble_validation_mse"], row["best_epoch"],
                             json.dumps(row["weights_used"]), json.dumps(row["weights_next"]), row["elapsed_seconds"]])
    return result


def predict_saved(run_dir, output_file, device="cpu"):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest["run_config"]
    torch.set_num_threads(cfg["threads"])
    saved = _load_checkpoint(run_dir / "best.pt", device)
    model = UDCGL(ModelConfig.from_checkpoint(saved["model_config"]), saved["adjacency"].cpu().numpy()).to(device)
    model.load_state_dict(saved["model"])
    with np.load(run_dir / "inputs.npz", allow_pickle=False) as arrays:
        predictions = predict_heads(model, arrays["input_values"], arrays["input_mask"], cfg["window"],
                                     cfg["batch_size"], device, cfg["stride"])
    scaler = SignalScaler(**manifest["scaler"])
    raw = scaler.inverse(predictions)
    ensemble = np.einsum("j,jnt->nt", saved["weights"].cpu().numpy(), raw)
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, prediction=ensemble.astype(np.float32), heads=raw.astype(np.float32))
    return ensemble
