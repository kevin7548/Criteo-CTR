"""Chronological Parquet blocks -> integer-id blocks, with a vocab from training blocks only.

The vocab (ADR-002, ADR-009) is counted over blocks 1–5 and then applied to all
seven blocks shard by shard, so validation and test values never reach it. The
output mirrors the block layout of ctr.data.split, with every field as an Int32 id:

    out_dir/
      manifest.json          # source block counts + vocab settings + field sizes
      vocab/vocab.parquet, vocab/meta.json
      block_1/part-00000.parquet, ...
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import polars as pl

from ctr.baseline import TRAIN_BLOCKS
from ctr.data.split import block_files, scan_block
from ctr.data.vocab import FIELDS, Vocab, build_vocab, encode

DEFAULT_MIN_COUNT = 10


def encode_blocks(
    blocks_dir: Path | str,
    out_dir: Path | str,
    min_count: int = DEFAULT_MIN_COUNT,
    train_blocks: tuple[int, ...] = TRAIN_BLOCKS,
) -> dict:
    """Build the vocab from `train_blocks` and encode every block listed in the manifest.

    Writes into `<out_dir>.tmp` and renames on success.
    """
    blocks_dir, out_dir = Path(blocks_dir), Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists; remove it to re-encode")
    tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    source = json.loads((blocks_dir / "manifest.json").read_text())
    try:
        train = pl.concat([scan_block(blocks_dir, k) for k in train_blocks])
        vocab = build_vocab(train, min_count=min_count)
        vocab.save(tmp_dir / "vocab")

        for k in range(1, source["n_blocks"] + 1):
            block_out = tmp_dir / f"block_{k}"
            block_out.mkdir(parents=True)
            for shard in block_files(blocks_dir, k):
                encode(pl.scan_parquet(shard), vocab).sink_parquet(
                    block_out / shard.name, compression="zstd"
                )
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    manifest = {
        **source,
        "vocab": {"min_count": min_count, "train_blocks": list(train_blocks)},
        "fields": FIELDS,
        "field_sizes": vocab.sizes,
    }
    (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    tmp_dir.rename(out_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--blocks", type=Path, default=Path("data/blocks"))
    parser.add_argument("--out", type=Path, default=Path("data/encoded"))
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT)
    args = parser.parse_args()

    manifest = encode_blocks(args.blocks, args.out, min_count=args.min_count)
    vocab = Vocab.load(args.out / "vocab")
    print(f"total ids={sum(manifest['field_sizes']):,} (min_count={args.min_count})")
    for f, size in zip(FIELDS, manifest["field_sizes"], strict=True):
        n = sum(manifest["blocks"][str(k)]["rows"] for k in TRAIN_BLOCKS)
        print(
            f"{f:>3}: ids={size:>9,} missing={vocab.missing[f] / n:6.2%} "
            f"oov={vocab.oov[f] / n:6.2%}"
        )


if __name__ == "__main__":
    main()
