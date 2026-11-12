"""Paired bootstrap confidence intervals for metric differences between runs (ADR-013).

Usage:
    uv run python -m ctr.compare --split test --out reports/compare/dcnv2_vs_lr.json \
        --run lr_s0=reports/lr.json --run dcnv2_s0=reports/dcnv2/dcnv2_s0.json \
        --pair dcnv2_s0:lr_s0

Every resample draws one Poisson(1) weight per row of the split (the Poisson
bootstrap, a large-n stand-in for multinomial resampling) and applies the same
weights to every run, so a difference is measured on identical rows and the shared
row-level noise cancels. LogLoss, AUC and PCOC are recomputed per resample; ECE is
not, because its equal-count bins would move with the weights. AUC is computed from
an ordering sorted once per run, so a resample costs O(n) instead of a sort.

Downsampled runs (`negative_keep_rate < 1`) are prior-corrected first (ADR-005), so a
LogLoss difference compares calibrated predictions.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl

from ctr.baseline import TEST_BLOCKS, VAL_BLOCKS
from ctr.calibration import prior_correct
from ctr.data.dataset import split_files
from ctr.metrics import EPS

METRICS = ("logloss", "auc", "pcoc")
SPLITS = {"val": VAL_BLOCKS, "test": TEST_BLOCKS}


def _loss_terms(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return -np.where(y == 1, np.log(p), np.log1p(-p))


def weighted_logloss(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    return float(np.dot(w, _loss_terms(np.asarray(y), p)) / w.sum())


class _SortedScores:
    """One run's predictions sorted once, with tie groups, for O(n) weighted AUC."""

    def __init__(self, y: np.ndarray, p: np.ndarray):
        self.order = np.argsort(p, kind="stable")
        p_sorted = p[self.order]
        self.y_sorted = y[self.order].astype(np.float64)
        new_group = np.empty(len(p), dtype=bool)
        new_group[0] = True
        np.not_equal(p_sorted[1:], p_sorted[:-1], out=new_group[1:])
        self.starts = np.flatnonzero(new_group)

    def auc(self, w: np.ndarray) -> float:
        ws = w[self.order]
        pos = ws * self.y_sorted
        pos_g = np.add.reduceat(pos, self.starts)
        neg_g = np.add.reduceat(ws - pos, self.starts)
        below = np.cumsum(neg_g) - neg_g
        return float(np.dot(pos_g, below + 0.5 * neg_g) / (pos_g.sum() * neg_g.sum()))


def weighted_auc(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    """AUC where row i counts w[i] times; ties between a positive and a negative count 1/2."""
    return _SortedScores(np.asarray(y), np.asarray(p, dtype=np.float64)).auc(np.asarray(w))


def percentile_ci(samples: np.ndarray, level: float = 0.95) -> list[float]:
    a = (1 - level) / 2 * 100
    lo, hi = np.percentile(samples, [a, 100 - a])
    return [float(lo), float(hi)]


class PairedBootstrap:
    def __init__(self, y: np.ndarray, preds: Mapping[str, np.ndarray]):
        self.y = np.asarray(y).astype(np.int8, copy=False)
        n = len(self.y)
        for name, p in preds.items():
            if len(p) != n:
                raise ValueError(f"{name}: {len(p)} predictions for {n} labels")
        self.preds = {k: np.asarray(v, dtype=np.float64) for k, v in preds.items()}
        self._loss = {k: _loss_terms(self.y, p) for k, p in self.preds.items()}
        self._sorted = {k: _SortedScores(self.y, p) for k, p in self.preds.items()}

    def _metrics(self, name: str, w: np.ndarray) -> dict[str, float]:
        w_sum = w.sum()
        return {
            "logloss": float(np.dot(w, self._loss[name]) / w_sum),
            "auc": self._sorted[name].auc(w),
            "pcoc": float(np.dot(w, self.preds[name]) / np.dot(w, self.y)),
        }

    def point(self) -> dict[str, dict[str, float]]:
        ones = np.ones(len(self.y))
        return {name: self._metrics(name, ones) for name in self.preds}

    def sample(self, n_boot: int, seed: int = 0) -> dict[str, dict[str, np.ndarray]]:
        rng = np.random.default_rng(seed)
        out = {name: {m: np.empty(n_boot) for m in METRICS} for name in self.preds}
        for b in range(n_boot):
            w = rng.poisson(1.0, len(self.y)).astype(np.float64)
            for name in self.preds:
                for m, v in self._metrics(name, w).items():
                    out[name][m][b] = v
        return out

    @staticmethod
    def summarise(point: dict, samples: dict, a: str, b: str) -> dict:
        """Difference a − b per metric, with its paired 95% percentile interval."""
        return {
            m: {
                "a": point[a][m],
                "b": point[b][m],
                "diff": point[a][m] - point[b][m],
                "ci95": percentile_ci(samples[a][m] - samples[b][m]),
            }
            for m in METRICS
        }

    def compare(self, a: str, b: str, n_boot: int = 1000, seed: int = 0) -> dict:
        return self.summarise(self.point(), self.sample(n_boot, seed), a, b)


def load_labels(encoded: Path | str, split: str) -> np.ndarray:
    """Labels of `split` in the row order ctr.evaluate.predict writes predictions in."""
    files = split_files(encoded, SPLITS[split])
    return pl.read_parquet(files, columns=["label"])["label"].to_numpy()


def load_run(report_path: Path | str, split: str) -> np.ndarray:
    """A run's saved predictions on `split`, prior-corrected if it was downsampled."""
    report = json.loads(Path(report_path).read_text())
    p = np.load(Path(report["config"]["artifacts"]) / f"{split}_pred.npy").astype(np.float64)
    w = report.get("negative_keep_rate", 1.0)
    return p if w == 1 else prior_correct(p, w)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--encoded", type=Path, default=Path("data/encoded"))
    parser.add_argument("--split", choices=sorted(SPLITS), default="test")
    parser.add_argument("--run", action="append", required=True, help="NAME=REPORT_JSON")
    parser.add_argument("--pair", action="append", required=True, help="A:B (reports A − B)")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    runs = dict(spec.split("=", 1) for spec in args.run)
    pairs = [tuple(spec.split(":", 1)) for spec in args.pair]
    y = load_labels(args.encoded, args.split)
    boot = PairedBootstrap(y, {name: load_run(path, args.split) for name, path in runs.items()})
    point = boot.point()
    samples = boot.sample(args.n_boot, args.seed)

    result = {
        "split": args.split,
        "n": len(y),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "runs": runs,
        "point": point,
        "pairs": {f"{a}-{b}": boot.summarise(point, samples, a, b) for a, b in pairs},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for key, diff in result["pairs"].items():
        for m, d in diff.items():
            lo, hi = d["ci95"]
            print(f"{key:28} {m:8} {d['diff']:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]")


if __name__ == "__main__":
    main()
