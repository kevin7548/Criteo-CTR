# Codex Execution Harness

This document defines how Codex should execute the CriteoCTR-Lab project without losing the reasoning behind each experiment.

## Repository Contract

Every meaningful experiment should leave behind:

- the command used to run it;
- the config file or CLI arguments;
- the data split identifier;
- the git commit hash when available;
- MLflow run ID;
- metrics artifact;
- short decision note explaining what was learned.

## Planned Directory Layout

```text
.
├── README.md
├── docs/
│   ├── harness.md
│   ├── model_card.md
│   └── portfolio_summary.md
├── configs/
│   ├── data.yaml
│   ├── baseline_ftrl.yaml
│   ├── deepfm.yaml
│   └── dcnv2.yaml
├── src/
│   └── criteo_ctr_lab/
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       └── benchmarking/
├── tests/
├── scripts/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── artifacts/
└── mlruns/
```

Large data files, generated artifacts, and local MLflow runs should not be committed.

## Make Targets

The project should eventually expose these stable commands:

```bash
make prepare-data
make test
make train-baseline
make train-deepfm
make train-dcnv2
make evaluate
make calibrate
make benchmark
make report
```

## First Execution Sequence

1. Check available disk space and confirm whether the Criteo train file is fully local or still an iCloud placeholder.
2. Initialize git and create `.gitignore`.
3. Create the Python package skeleton.
4. Add config files for data and the first baseline.
5. Implement TSV schema validation.
6. Convert a small sample to Parquet.
7. Add leakage and split tests.
8. Build the Constant CTR baseline before any neural model.

## Experiment Tracking Choice

Use MLflow for this project.

Reasoning:

- it is common in production-oriented MLOps stacks;
- it works well locally without requiring a hosted account;
- it naturally supports experiment runs, artifacts, and model registry concepts;
- it gives interview material around reproducibility and model lifecycle, not only pretty experiment dashboards.

Weights & Biases is still excellent for research-heavy collaboration and rich visual experiment comparison. For this portfolio, MLflow is the better signal because the project is meant to show production-minded ML engineering judgment.

## Git Workflow

Recommended branch flow:

```text
main
└── feat/data-harness
└── feat/baselines
└── feat/deep-models
└── feat/calibration-diagnostics
└── feat/inference-benchmark
```

Each merged unit should correspond to a visible deliverable, not just code movement.

## Codex Working Rule

Before changing code, Codex should state:

- what part of the plan is being implemented;
- which files will be touched;
- what will be verified after the change.

After changing code, Codex should report:

- what changed;
- what was verified;
- what remains risky or unfinished.
