from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass
class MaskSplit:
    fit: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    metadata: dict

    def validate(self, known):
        masks = [self.fit, self.validation, self.test]
        if any(m.shape != known.shape or m.dtype != bool for m in masks):
            raise ValueError("Masks must be boolean and match the sensor matrix")
        if any(not m.any() for m in masks):
            raise ValueError("Fitting, validation and test sets must all be nonempty")
        if np.any((self.fit.astype(int) + self.validation + self.test) != known.astype(int)):
            raise ValueError("Split masks must be disjoint and exactly partition known entries")


def _holdout(known, test_fraction, pattern, rng, block_length=None):
    known = np.asarray(known, dtype=bool)
    if known.ndim != 2 or not 0 < test_fraction < 1 or not known.any():
        raise ValueError("Require a nonempty sensors x time mask and a fraction in (0,1)")
    n, t = known.shape
    count = int(known.sum())
    target = max(1, int(round(test_fraction * count)))
    test = np.zeros_like(known)
    details = {}
    if pattern == "random":
        selected = rng.choice(np.flatnonzero(known), size=target, replace=False)
        test.flat[selected] = True
    elif pattern == "block":
        length = block_length if block_length is not None else max(2, int(round(t * 0.1)))
        if not 1 <= length <= t:
            raise ValueError("Block length must be between 1 and the number of time steps")
        candidates = [(i, start) for i in range(n) for start in range(0, t, length)]
        rng.shuffle(candidates)
        blocks = []
        remaining = target
        for i, start in candidates:
            available = np.flatnonzero(known[i, start:start + length]) + start
            if not len(available):
                continue
            chosen = available[:remaining]
            test[i, chosen] = True
            blocks.append([int(i), int(start), int(chosen[-1]) + 1])
            remaining -= len(chosen)
            if remaining == 0:
                break
        if remaining:
            raise ValueError("Not enough known entries for block holdout")
        details = {"block_length": int(length), "blocks": blocks}
    elif pattern == "sensor":
        counts = known.sum(1)
        order = rng.permutation(np.flatnonzero(counts > 0))
        if len(order) < 2:
            raise ValueError("Whole-sensor holdout needs at least two measured sensors")
        cumulative = np.cumsum(counts[order])
        k = int(np.argmin(np.abs(cumulative[:-1] - target))) + 1
        selected_nodes = order[:k]
        test[selected_nodes] = known[selected_nodes]
        details = {"held_out_sensor_indices": [int(i) for i in selected_nodes],
                   "ratio_is_quantized_by_whole_sensors": True}
    else:
        raise ValueError(f"Unknown missingness pattern: {pattern}")
    return test, details


def make_split(known, test_fraction=0.2, val_fraction=0.1, pattern="random",
               seed=0, block_length=None) -> MaskSplit:
    known = np.asarray(known, dtype=bool)
    if not (0 < test_fraction < 1 and 0 < val_fraction < 1):
        raise ValueError("Require test and validation fractions in (0,1)")
    if test_fraction + val_fraction >= 1:
        raise ValueError("Test and validation fractions must leave fitting entries")
    rng = np.random.default_rng(seed)
    test, details = _holdout(known, test_fraction, pattern, rng, block_length)
    count = int(known.sum())
    candidates = np.flatnonzero(known & ~test)
    n_val = max(1, int(round(val_fraction * count)))
    if n_val >= len(candidates):
        raise ValueError("Split leaves insufficient fitting data")
    validation = np.zeros_like(known)
    validation.flat[rng.choice(candidates, size=n_val, replace=False)] = True
    fit = known & ~validation & ~test
    meta = {"pattern": pattern, "seed": int(seed),
            "independent_test": True,
            "evaluation_role": "independent held-out test",
            "requested_test_fraction": test_fraction, "requested_val_fraction": val_fraction,
            "mask_sha256": {name: hashlib.sha256(mask.tobytes()).hexdigest()
                            for name, mask in [("fit", fit), ("validation", validation), ("test", test)]},
            "counts": {"fit": int(fit.sum()), "validation": int(validation.sum()), "test": int(test.sum())},
            "realized_fractions": {"fit": float(fit.sum() / count),
                                   "validation": float(validation.sum() / count),
                                   "test": float(test.sum() / count)}, **details}
    split = MaskSplit(fit, validation, test, meta)
    split.validate(known)
    return split


def contaminate(values, known, split, rate=0.0, mode="replace_observed", seed=0, extrema_scope="global"):
    if not 0 <= rate <= 1:
        raise ValueError("Outlier rate must lie in [0,1]")
    if mode not in {"replace_observed", "inject_missing"}:
        raise ValueError(f"Unknown outlier mode: {mode}")
    if extrema_scope not in {"global", "sensor"}:
        raise ValueError("Extrema scope must be global or sensor")
    rng = np.random.default_rng(seed)
    fitting_values = np.array(values, copy=True)
    fitting_mask = split.fit.copy()
    corrupt = np.zeros_like(known)
    amount = int(round(rate * split.fit.sum()))
    if amount:
        if mode == "replace_observed":
            eligible = np.flatnonzero(split.fit)
        elif mode == "inject_missing":
            eligible = np.flatnonzero(~known)
        else:
            raise ValueError(mode)
        if amount > len(eligible):
            raise ValueError(f"Requested {amount} outliers but only {len(eligible)} eligible entries for {mode}; "
                             "do not silently substitute a different contamination protocol")
        chosen = rng.choice(eligible, amount, replace=False)
        observed = values[split.fit]
        low, high = np.full(amount, observed.min()), np.full(amount, observed.max())
        if extrema_scope == "sensor":
            nodes = chosen // values.shape[1]
            for node in np.unique(nodes):
                local = values[node, split.fit[node]]
                if len(local):
                    low[nodes == node], high[nodes == node] = local.min(), local.max()
        fitting_values.flat[chosen] = np.where(rng.integers(0, 2, amount), high, low)
        fitting_mask.flat[chosen] = True
        corrupt.flat[chosen] = True
    if np.any(fitting_mask & (split.validation | split.test)):
        raise ValueError("Contamination overlaps a held-out reference")
    return fitting_values, fitting_mask, corrupt
