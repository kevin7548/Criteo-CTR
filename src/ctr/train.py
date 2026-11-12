"""Train a CTR model on blocks 1–5, choose the epoch on block 6, report blocks 6 and 7.

Usage:
    uv run python -m ctr.train --config configs/lr.toml

Training rows are shuffled within the training blocks (ADR-010); validation and
test are always predicted in original row order and on their full original
distribution. The epoch with the lowest validation LogLoss is kept, and training
stops after `patience` epochs without improvement. Hyperparameters are chosen on
validation only; the test block is scored once, with the kept epoch.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from ctr.baseline import TEST_BLOCKS, TRAIN_BLOCKS, VAL_BLOCKS
from ctr.data.dataset import iter_batches, split_files
from ctr.evaluate import evaluate_split, predict, resolve_device, train_ctr
from ctr.models import build_model


@dataclass
class TrainConfig:
    model: str = "lr"
    model_args: dict = field(default_factory=dict)
    seed: int = 0
    device: str = "auto"
    # data
    encoded: Path = Path("data/encoded")
    max_train_shards: int | None = None
    max_val_shards: int | None = None
    evaluate_test: bool = True
    # train
    batch_size: int = 4096
    lr: float = 1e-3
    epochs: int = 3
    patience: int = 1
    shards_per_chunk: int = 5
    eval_batch_size: int = 1 << 16
    # output
    artifacts: Path = Path("artifacts/lr")
    report: Path = Path("reports/lr.json")

    def __post_init__(self) -> None:
        for name in ("encoded", "artifacts", "report"):
            setattr(self, name, Path(getattr(self, name)))

    def to_dict(self) -> dict:
        return {
            f.name: str(v) if isinstance(v := getattr(self, f.name), Path) else v
            for f in fields(self)
        }


_SECTIONS = {
    "data": {"encoded", "max_train_shards", "max_val_shards", "evaluate_test"},
    "train": {"batch_size", "lr", "epochs", "patience", "shards_per_chunk", "eval_batch_size"},
    "output": {"artifacts", "report"},
}
_TOP_LEVEL = {"seed", "device"}


def load_config(path: Path | str) -> TrainConfig:
    """Read a TOML config. `[model]` takes `name` plus keyword arguments for the model."""
    raw = tomllib.loads(Path(path).read_text())
    values: dict = {}
    unknown: list[str] = []
    for key, value in raw.items():
        if key == "model":
            args = dict(value)
            values["model"] = args.pop("name", TrainConfig.model)
            values["model_args"] = args
        elif key in _SECTIONS:
            for k, v in value.items():
                (values.__setitem__(k, v) if k in _SECTIONS[key] else unknown.append(f"{key}.{k}"))
        elif key in _TOP_LEVEL:
            values[key] = value
        else:
            unknown.append(key)
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {', '.join(unknown)}")
    return TrainConfig(**values)


def _train_epoch(model, optimizer, files, cfg: TrainConfig, epoch: int, device: str) -> float:
    model.train()
    total, rows = 0.0, 0
    batches = iter_batches(
        files,
        cfg.batch_size,
        shuffle=True,
        seed=cfg.seed * 1_000_003 + epoch,
        shards_per_chunk=cfg.shards_per_chunk,
    )
    for ids, labels in batches:
        x = torch.from_numpy(ids).to(device).long()
        y = torch.from_numpy(labels).to(device).float()
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
        rows += len(labels)
    return total / rows


def _relative(files: list[Path]) -> list[str]:
    return [f"{f.parent.name}/{f.name}" for f in files]


def run(cfg: TrainConfig) -> dict:
    started = time.perf_counter()
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)

    manifest = json.loads((cfg.encoded / "manifest.json").read_text())
    base_rate = train_ctr(manifest, TRAIN_BLOCKS)
    train_files = split_files(cfg.encoded, TRAIN_BLOCKS, cfg.max_train_shards)
    val_files = split_files(cfg.encoded, VAL_BLOCKS, cfg.max_val_shards)

    model = build_model(cfg.model, manifest["field_sizes"], base_rate=base_rate, **cfg.model_args)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history, best_state, best_loss, best_epoch = [], None, float("inf"), 0
    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = _train_epoch(model, optimizer, train_files, cfg, epoch, device)
        y_val, p_val = predict(model, val_files, cfg.eval_batch_size, device)
        val = evaluate_split(y_val, p_val, base_rate)
        val.pop("calibration_bins")
        history.append(
            {
                "epoch": epoch,
                "train_logloss": train_loss,
                "val": val,
                "seconds": time.perf_counter() - epoch_start,
            }
        )
        print(
            f"epoch {epoch}: train_logloss={train_loss:.5f} val_logloss={val['logloss']:.5f} "
            f"ne={val['ne']:.4f} auc={val['auc']:.4f} ece={val['ece']:.5f} "
            f"pcoc={val['pcoc']:.4f} ({history[-1]['seconds']:.0f}s)",
            flush=True,
        )
        if val["logloss"] < best_loss:
            best_loss, best_epoch = val["logloss"], epoch
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        elif epoch - best_epoch >= cfg.patience:
            break

    model.load_state_dict(best_state)
    cfg.artifacts.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, cfg.artifacts / "model.pt")

    report = {
        "config": cfg.to_dict(),
        "device": device,
        "base_rate": base_rate,
        "n_params": sum(p.numel() for p in model.parameters()),
        "train_files": _relative(train_files),
        "train_rows": sum(pq.ParquetFile(f).metadata.num_rows for f in train_files),
        "best_epoch": best_epoch,
        "history": history,
    }
    splits = [("val", val_files)]
    if cfg.evaluate_test:
        splits.append(("test", split_files(cfg.encoded, TEST_BLOCKS)))
    for name, files in splits:
        y, p = predict(model, files, cfg.eval_batch_size, device)
        np.save(cfg.artifacts / f"{name}_pred.npy", p.astype(np.float32))
        report[name] = evaluate_split(y, p, base_rate)
    report["seconds"] = time.perf_counter() - started

    cfg.report.parent.mkdir(parents=True, exist_ok=True)
    cfg.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    report = run(load_config(args.config))
    for name in ("val", "test"):
        if name in report:
            s = report[name]
            print(
                f"{name:4} logloss={s['logloss']:.5f} ne={s['ne']:.4f} auc={s['auc']:.4f} "
                f"ece={s['ece']:.5f} pcoc={s['pcoc']:.4f}"
            )


if __name__ == "__main__":
    main()
