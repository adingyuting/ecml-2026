from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
import numpy as np
from scipy.io import loadmat

DATASET_IDS = ("D1", "D2", "D3", "D4")

VALUE_NAMES = ("values.csv", "values.mat")
LINK_NAMES = ("adjacency.csv", "adjacency.mat")
VALUE_VARIABLE = "values"
LINK_VARIABLE = "adjacency"


def digest_file(target: Path) -> str:
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def publish_file(target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(target: Path, payload) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    with publish_file(target) as stream:
        stream.write(body)


def write_npz(target, **arrays):
    with publish_file(target) as stream:
        np.savez_compressed(stream, **arrays)


def load_matrix(target: Path, variable: str | None = None) -> np.ndarray:
    suffix = target.suffix.lower()
    if suffix == ".mat":
        contents = loadmat(target)
        if variable is not None:
            if variable not in contents:
                raise ValueError(f"{target} does not contain the MATLAB variable {variable}")
            payload = contents[variable]
        else:
            options = [item for name, item in contents.items()
                       if not name.startswith("__") and isinstance(item, np.ndarray)
                       and item.ndim == 2 and item.dtype.kind in "fiu"]
            if len(options) != 1:
                raise ValueError(f"Specify an unambiguous MATLAB variable for {target}")
            payload = options[0]
    elif suffix == ".npy":
        payload = np.load(target, allow_pickle=False)
    else:
        text = target.read_text(encoding="utf-8-sig").splitlines()
        if not text:
            raise ValueError(f"Empty matrix file: {target}")
        head = text[0]
        if "::" in head:
            payload = np.asarray([[float(cell) for cell in row.split("::")]
                                  for row in text if row.strip()])
        else:
            payload = np.loadtxt(target, delimiter="," if "," in head else None, encoding="utf-8-sig")
    payload = np.asarray(payload, dtype=np.float64)
    if payload.ndim != 2:
        raise ValueError(f"Expected a sensor x time matrix, got {payload.shape}: {target}")
    return payload


def normalize_graph(target: Path, sensor_count: int) -> np.ndarray:
    graph = load_matrix(target)
    if graph.shape != (sensor_count, sensor_count) or not np.isfinite(graph).all():
        raise ValueError(f"Invalid adjacency {graph.shape}, expected {(sensor_count, sensor_count)}")
    offdiagonal = graph.copy()
    np.fill_diagonal(offdiagonal, 0)
    if np.any(offdiagonal < -1e-8):
        if np.any(offdiagonal > 1e-8) or not np.allclose(graph.sum(1), 0, atol=1e-5):
            raise ValueError("Graph has mixed-sign off-diagonal entries; supply adjacency explicitly")
        graph = -offdiagonal
    graph = np.maximum(graph, graph.T)
    np.fill_diagonal(graph, 0)
    if (graph < 0).any():
        raise ValueError("Adjacency must be nonnegative")
    return graph.astype(np.float32)


def load_layout(layout_path):
    if not layout_path:
        return {}
    target = Path(layout_path).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    if not target.is_file():
        raise FileNotFoundError(f"Dataset layout file not found: {target}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Dataset layout must be a JSON object keyed by dataset identifier")
    return payload


def resolve_dataset_dir(root: Path, dataset_id: str, layout: dict) -> Path:
    override = layout.get(dataset_id)
    if override is None:
        folder = root / dataset_id
    elif isinstance(override, dict):
        folder = Path(override["folder"]).expanduser()
    else:
        folder = Path(str(override)).expanduser()
    if not folder.is_absolute():
        folder = root / folder
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Dataset folder {folder} is missing. Every dataset uses the same layout: one directory per "
            f"dataset identifier holding {VALUE_NAMES[0]} and {LINK_NAMES[0]} "
            f"(or their .mat counterparts with the variables {VALUE_VARIABLE} and {LINK_VARIABLE}).")
    return folder


def pick_file(folder: Path, names, variable: str | None = None) -> Path:
    for name in names:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"None of {list(names)} was found in {folder}. Every dataset uses the same layout: one directory "
        f"per dataset identifier holding {VALUE_NAMES[0]} and {LINK_NAMES[0]} "
        f"(or their .mat counterparts with the variables {VALUE_VARIABLE} and {LINK_VARIABLE}).")


@dataclass
class DatasetBundle:
    signal: np.ndarray
    observed: np.ndarray
    graph: np.ndarray
    info: dict


def load_dataset(root, dataset_id, zero_is_missing=True, max_steps: int | None = None,
                 max_nodes: int | None = None, layout=None) -> DatasetBundle:
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {base}")
    folder = resolve_dataset_dir(base, dataset_id, layout or {})
    value_file = pick_file(folder, VALUE_NAMES)
    link_file = pick_file(folder, LINK_NAMES)
    signal = load_matrix(value_file, VALUE_VARIABLE if value_file.suffix.lower() == ".mat" else None)
    if signal.shape[0] < 2:
        raise ValueError(f"Expected at least two sensors in {value_file}, got {signal.shape}")
    graph = normalize_graph(link_file, signal.shape[0])
    observed = np.isfinite(signal)
    if zero_is_missing:
        observed &= signal != 0
    if not observed.any():
        raise ValueError(f"No measured entries in {value_file}")
    measured = signal[observed]
    info = {"dataset": dataset_id, "data_file": str(value_file), "data_sha256": digest_file(value_file),
            "graph_file": str(link_file), "graph_sha256": digest_file(link_file),
            "original_shape": list(signal.shape), "known_count": int(observed.sum()),
            "native_missing_count": int((~observed).sum()),
            "known_range": [float(measured.min()), float(measured.max())],
            "zero_is_missing": bool(zero_is_missing),
            "notes": ["Entries equal to zero are treated as missing unless --zero-is-valid is passed."]}
    sensors = min(max_nodes or signal.shape[0], signal.shape[0])
    steps = min(max_steps or signal.shape[1], signal.shape[1])
    signal, observed = signal[:sensors, :steps].copy(), observed[:sensors, :steps].copy()
    graph = graph[:sensors, :sensors].copy()
    info.update({"used_shape": [sensors, steps], "truncated_for_smoke": [sensors, steps] != info["original_shape"]})
    return DatasetBundle(signal, observed, graph, info)


@dataclass
class SignalScaler:
    center: float
    spread: float
    mode: str

    @classmethod
    def fit(cls, values, fitting_mask, mode="standard"):
        selected = np.asarray(values)[np.asarray(fitting_mask, dtype=bool)]
        if len(selected) == 0 or not np.isfinite(selected).all():
            raise ValueError("Scaler requires finite fitting observations")
        if mode == "standard":
            center, spread = float(selected.mean()), float(selected.std())
        elif mode == "minmax":
            center, spread = float(selected.min()), float(np.ptp(selected))
        elif mode == "none":
            center, spread = 0.0, 1.0
        else:
            raise ValueError(mode)
        return cls(center, spread if spread > 1e-8 else 1.0, mode)

    def transform(self, values):
        return (np.asarray(values, dtype=np.float64) - self.center) / self.spread

    def inverse(self, values):
        return np.asarray(values) * self.spread + self.center

    def to_dict(self):
        return {"center": self.center, "spread": self.spread, "mode": self.mode}
