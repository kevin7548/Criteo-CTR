# CriteoCTR-Lab

> Read this in: English | [한국어](README.ko.md)

A CTR prediction study on the Criteo click logs that measures, and fixes, **problems that AUC does not show but LogLoss and calibration do**.

## Core claim

If you train on fewer non-click samples (negatives), AUC stays almost the same, but the mean predicted CTR inflates sharply. The problem is invisible to AUC and shows up only in LogLoss and ECE. When the sampling rate is known, a single line, `logit(q) = logit(p) + log w`, corrects it.

The result is summarized in three numbers.

| | Measurement | Expectation |
|---|---|---|
| 1 | AUC: full-data model vs. downsampled model | Nearly the same |
| 2 | Mean predicted CTR vs. actual CTR | Strongly inflated in the downsampled model |
| 3 | ECE and LogLoss after correction | Back to the full-data level (minus the effect of the discarded data) |

## Data

- Criteo Display Advertising Challenge (Kaggle) `train.txt`: about 45.8M rows over 7 days; 1 label, 13 numeric and 26 categorical features
- `test.txt` has no labels and is not used.
- Split: 7 blocks that keep the original row order. Blocks 1–5 train, 6 validation, 7 test.
- Vocabularies and normalization statistics are computed on the training blocks only.

## Experiments

| Stage | Content | Status |
|---|---|---|
| 1 | TSV → Parquet, chronological split, constant-CTR baseline | Done |
| 2 | LogLoss, AUC, ECE, NE, and PCOC implemented without ML libraries | Done |
| 3 | Logistic regression (LR) | Done |
| 4 | Negative downsampling (`w` ∈ {1, 0.5, 0.25, 0.1}) and prior correction | Done |
| 5 | DCNv2 vs. LR (3 seeds, paired bootstrap confidence intervals) | Done |
| 6 | (Optional) LogLoss and ECE by feature-frequency bucket (head/mid/tail/OOV) | Not started |

## Results

Full results and interpretation for stages 1–5 are in **[reports/RESULTS.md](reports/RESULTS.md)** (Korean). Summary on the test block (block 7):

| Model | LogLoss | NE | AUC | ECE | PCOC |
|---|---|---|---|---|---|
| Constant CTR | 0.57278 | 1 | 0.5 | – | 0.983 |
| LR (mean of 3 seeds) | 0.46171 | 0.8061 | 0.7909 | 0.0028 | 1.001 |
| DCNv2 (mean of 3 seeds) | 0.44825 | 0.7826 | 0.8064 | 0.0079 | 1.019 |
| LR, 10% of negatives (w = 0.1), before correction | 0.90582 | 1.5814 | 0.7886 | 0.4020 | 2.548 |
| LR, w = 0.1, after correction | 0.46418 | 0.8104 | 0.7886 | 0.0088 | 0.971 |

- Keeping 10% of negatives leaves AUC almost unchanged (0.7909 → 0.7886), but the mean predicted CTR inflates 2.55× and NE goes above 1 before correction. The one-line correction `logit(q) = logit(p) + log w` brings ECE from 0.402 to 0.0088 and LogLoss from 0.906 to 0.464.
- Keeping 50% of negatives cuts training rows by 37% with no LogLoss difference distinguishable from zero (paired bootstrap, 1,000 resamples).
- DCNv2 beats LR on both LogLoss and AUC, but its ECE is almost 3× worse. Better ranking does not bring better calibration with it.

## Metrics

- **LogLoss / NE**: probability quality. NE divides LogLoss by the LogLoss of a constant prediction at the training-block CTR.
- **AUC**: ranking quality. It is invariant to monotonic transforms, so it cannot detect calibration problems.
- **ECE**: calibration error over 20 equal-mass bins. Identical predictions are kept in the same bin (ADR-008).
- **PCOC**: mean predicted CTR / actual CTR.

## Limitations

- The Criteo data is already a sampled distribution, and the original sampling rate is not public. Calibration results here are relative to that sampled distribution; the project does not claim to recover real production CTR.
- The blocks are chronological but are not exact day boundaries.
- The data has no user, session, or auction information, so this project covers only the pointwise scoring model inside a ranking system, not the full system.

## Out of scope

DeepFM, ONNX inference benchmarks, latency and memory comparisons, LightGBM, FTRL, ranking metrics (NDCG etc.), and claims about real A/B tests.

## Design decisions

[docs/ADR.md](docs/ADR.md) (Korean)

## Use of AI tools

The code and documents were written with Claude Code under the TDD and pre-commit rules in `AGENTS.md`. I chose the research question, the evaluation protocol (chronological split, metrics, comparison rules), and the experiments, and reviewed the results.

## Development

```bash
uv sync
uv run pytest -q
# train.txt → data/blocks/block_{1..7} (ADR-007)
uv run python -m ctr.data.split --src kaggle-display-advertising-challenge-dataset/train.txt --out data/blocks
# constant-CTR baseline → reports/baseline_constant.json
uv run python -m ctr.baseline
# training-block vocabulary + id encoding for all blocks → data/encoded (ADR-009)
uv run python -m ctr.data.encode --blocks data/blocks --out data/encoded --min-count 10
# LR: smoke run on the first shards, then the full run → reports/lr.json, artifacts/lr/ (ADR-010)
uv run python -m ctr.train --config configs/lr_smoke.toml
uv run python -m ctr.train --config configs/lr.toml
# negative downsampling (ADR-011) → reports/downsampling/
for w in 050 025 010; do uv run python -m ctr.train --config configs/lr_w$w.toml; done
# DCNv2 vs. LR, seeds 0–2 (ADR-012) → reports/model_comparison/
for c in dcnv2_s0 dcnv2_s1 dcnv2_s2 lr_s1 lr_s2; do uv run python -m ctr.train --config configs/$c.toml; done
# paired bootstrap confidence intervals (ADR-013); full commands in reports/RESULTS.md
uv run python -m ctr.compare --help
uv run ruff check .
```
