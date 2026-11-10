"""Constant-CTR baseline: predict the pooled CTR of the training blocks for every row.

Only label counts are needed, so everything comes from the block manifest. The
prediction uses training blocks only. Its LogLoss on a split is the denominator of
that split's NE, so this baseline has NE = 1 by definition and AUC = 0.5.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

TRAIN_BLOCKS = (1, 2, 3, 4, 5)
VAL_BLOCKS = (6,)
TEST_BLOCKS = (7,)


def _check_counts(positives: int, n: int) -> None:
    if n <= 0 or not 0 <= positives <= n:
        raise ValueError(f"invalid counts: positives={positives}, n={n}")


def constant_logloss(positives: int, n: int, p: float) -> float:
    """Mean binary cross-entropy of predicting `p` for all `n` rows."""
    _check_counts(positives, n)
    if not 0 < p < 1:
        raise ValueError(f"p must be in (0, 1), got {p}")
    return -(positives * math.log(p) + (n - positives) * math.log1p(-p)) / n


def _entropy(positives: int, n: int) -> float:
    """LogLoss of the split's own CTR: the best any constant could do on that split."""
    _check_counts(positives, n)
    if positives in (0, n):
        return 0.0
    return constant_logloss(positives, n, positives / n)


def _totals(manifest: dict, blocks: tuple[int, ...]) -> tuple[int, int]:
    infos = [manifest["blocks"][str(k)] for k in blocks]
    return sum(i["rows"] for i in infos), sum(i["positives"] for i in infos)


def constant_baseline(manifest: dict) -> dict:
    """Score the training-block CTR on the train, validation and test splits."""
    train_rows, train_positives = _totals(manifest, TRAIN_BLOCKS)
    prediction = train_positives / train_rows

    splits = {}
    for name, blocks in (("train", TRAIN_BLOCKS), ("val", VAL_BLOCKS), ("test", TEST_BLOCKS)):
        rows, positives = _totals(manifest, blocks)
        ctr = positives / rows
        splits[name] = {
            "blocks": list(blocks),
            "rows": rows,
            "positives": positives,
            "ctr": ctr,
            "logloss": constant_logloss(positives, rows, prediction),
            "entropy": _entropy(positives, rows),
            "pcoc": prediction / ctr,
        }
    return {"train_blocks": list(TRAIN_BLOCKS), "prediction": prediction, "splits": splits}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--blocks", type=Path, default=Path("data/blocks"))
    parser.add_argument("--out", type=Path, default=Path("reports/baseline_constant.json"))
    args = parser.parse_args()

    manifest = json.loads((args.blocks / "manifest.json").read_text())
    result = constant_baseline(manifest)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"prediction={result['prediction']:.6f}")
    for name, s in result["splits"].items():
        print(
            f"{name:5} rows={s['rows']:>10,} ctr={s['ctr']:.6f} logloss={s['logloss']:.6f} "
            f"entropy={s['entropy']:.6f} pcoc={s['pcoc']:.4f}"
        )


if __name__ == "__main__":
    main()
