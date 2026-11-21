from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

from .data import DATASET_IDS, write_json
from .train import RunConfig, train_run, predict_saved

PROJECT = Path(__file__).resolve().parents[1]


def default_root():
    if os.environ.get("UDCGL_DATA_ROOT"):
        return os.environ["UDCGL_DATA_ROOT"]
    local = PROJECT / "configs/local.json"
    if local.exists():
        return json.loads(local.read_text(encoding="utf-8"))["data_root"]
    return None


def local_settings():
    path = PROJECT / "configs/local.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def add_training_args(parser):
    parser.add_argument("--config", type=Path, help="JSON configuration; command-line values override it")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--dataset", choices=list(DATASET_IDS))
    parser.add_argument("--dataset-layout", default=None,
                        help="JSON file mapping each dataset identifier to its directory")
    parser.add_argument("--pattern", choices=["random", "block", "sensor"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--scaler", choices=["standard", "minmax", "none"])
    parser.add_argument("--variant", dest="variant", choices=["full", "zero_fill", "single_head", "static_average"])
    parser.add_argument("--outlier-mode", dest="outlier_mode", choices=["replace_observed", "inject_missing"])
    parser.add_argument("--outlier-scope", dest="outlier_scope", choices=["global", "sensor"])
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--device", help="cpu, cuda or cuda:0")
    for name in ["test_fraction", "val_fraction", "learning_rate", "regularization", "delta", "dropout", "outlier_rate",
                 "weight_decay", "adam_beta1", "adam_beta2", "sgd_momentum", "grad_clip"]:
        help_text = {"test_fraction": "Fraction of known entries held out for testing",
                     "val_fraction": "Fraction of known entries reserved for validation"}.get(name)
        parser.add_argument("--" + name.replace("_", "-"), type=float, help=help_text)
    for name in ["epochs", "patience", "batch_size", "window", "stride", "width", "depth", "heads",
                 "cheb_order", "context_span", "single_head", "block_length", "threads",
                 "max_steps", "max_nodes", "max_train_batches", "split_seed"]:
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    parser.add_argument("--kernels", type=int, nargs="+")
    parser.add_argument("--zero-is-valid", dest="zero_is_missing", action="store_false", default=None,
                        help="Use only nonfinite entries as native missingness; zero is a valid measurement")
    parser.add_argument("--resume", action="store_true", help="Resume last.pt in this exact run directory")
    parser.add_argument("--output", type=Path)


def training_config(args):
    base = local_settings()
    if args.command == "train" and args.resume and args.output:
        manifest = args.output / "manifest.json"
        if manifest.is_file():
            saved = json.loads(manifest.read_text(encoding="utf-8"))["run_config"]
            base.update({key: value for key, value in saved.items()})
    if args.config:
        base.update(json.loads(args.config.read_text(encoding="utf-8")))
    valid = {item.name for item in RunConfig.__dataclass_fields__.values()}
    unknown = set(base) - valid
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    for name, value in vars(args).items():
        if name in valid and value is not None:
            base[name] = value
    if not base.get("data_root"):
        base["data_root"] = default_root()
    if not base["data_root"]:
        raise ValueError("Supply --data-root or set UDCGL_DATA_ROOT")
    if "kernels" in base:
        base["kernels"] = tuple(base["kernels"])
    if base.get("dataset_layout") and not Path(base["dataset_layout"]).is_absolute():
        base["dataset_layout"] = str((PROJECT / base["dataset_layout"]).resolve())
    return RunConfig(**base)


def main(argv=None):
    parser = argparse.ArgumentParser(description="UDCGL implementation / sensor data recovery experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Inspect the prepared datasets without modifying them")
    inspect.add_argument("--data-root", default=None)
    inspect.add_argument("--dataset-layout", default=None)
    inspect.add_argument("--output", type=Path, default=Path("runs/data_inventory.json"))
    inspect.add_argument("--zero-is-valid", action="store_true")
    inspect.add_argument("--datasets", nargs="+", choices=list(DATASET_IDS), default=list(DATASET_IDS))
    train = commands.add_parser("train", help="Train one dataset/pattern/seed and score the selected checkpoint")
    add_training_args(train)
    pred = commands.add_parser("predict", help="Reload the frozen checkpoint and predict from stored fitting inputs")
    pred.add_argument("--run", type=Path, required=True)
    pred.add_argument("--output", type=Path, required=True)
    pred.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            from .data import load_layout
            from .train import load_run_dataset
            settings = {key: value for key, value in local_settings().items()
                        if key in set(RunConfig.__dataclass_fields__)}
            for key in ["data_root", "dataset_layout"]:
                if getattr(args, key) is not None:
                    settings[key] = getattr(args, key)
            settings.setdefault("data_root", default_root())
            if args.zero_is_valid:
                settings["zero_is_missing"] = False
            layout = load_layout(settings.get("dataset_layout")) if settings.get("dataset_layout") else None
            inventory = [load_run_dataset(RunConfig(**{**settings, "dataset": name})).info
                         for name in args.datasets]
            write_json(args.output, inventory)
            for item in inventory:
                print(f"{item['dataset']}: {item['original_shape']}, known={item['known_count']}, "
                      f"native_missing={item['native_missing_count']}, range={item['known_range']}")
                for note in item["notes"]:
                    print("  " + note)
        elif args.command == "train":
            cfg = training_config(args)
            output = args.output or Path("runs") / f"{cfg.dataset}_{cfg.pattern}_seed{cfg.seed}"
            result = train_run(cfg, output, args.resume)
            print(json.dumps({"output": str(output.resolve()), **result}, ensure_ascii=False, indent=2))
        elif args.command == "predict":
            prediction = predict_saved(args.run, args.output, args.device)
            print(f"Saved {prediction.shape} prediction to {args.output}")
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, FloatingPointError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
