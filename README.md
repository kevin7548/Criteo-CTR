# CriteoCTR-Lab

Large-scale sparse click log experiments for pointwise ad click scoring.

## Project Goal

This project studies whether feature-interaction models such as DeepFM and DCNv2 justify their extra inference cost over linear CTR baselines on sparse advertising logs.

The target role framing is Recommendation, Search, and Ads ML Engineer intern/new-grad positions. The project is not presented as a full ranking system because the Criteo dataset does not include query, session, request, user, auction, or candidate-list identifiers. Instead, it focuses on a pointwise scoring layer commonly used inside ranking and advertising systems.

## Core Question

How much do non-linear feature-interaction models improve click prediction quality over linear baselines, and when does that improvement justify additional latency and memory cost?

## Data

- Dataset: Criteo Display Advertising Challenge
- Format: label, 13 numerical features, 26 categorical features
- Split policy: preserve original order and create chronological blocks
- Train: blocks 1-5
- Validation: block 6A
- Calibration fitting: block 6B
- Final test: block 7

The blocks are chronological blocks, not calendar days. The project will only claim temporal distance and drift patterns, not exact retraining intervals.

## Scope

Included:

- Out-of-core preprocessing with sharded Parquet
- Train-only preprocessing statistics
- Leakage tests
- Constant CTR and linear baselines
- DeepFM and DCNv2 experiments
- Calibration on the sampled data distribution
- Controlled prior-shift sensitivity experiment
- Segment-wise robustness diagnostics
- Model quality, latency, memory, and size trade-off analysis
- Reproducible command harness

Excluded from the core project:

- NDCG, Recall@K, and GAUC
- CVR, ROAS, auction outcome, and true production CTR recovery
- Real A/B test claims
- Spark, Airflow, Kafka, Kubernetes, and TorchRec

Spark may be revisited only as a separate extension if the project moves to a larger Criteo 1TB subset where single-node processing is empirically insufficient.

## Stack

- Data: Polars, PyArrow, sharded Parquet
- Training: PyTorch, IterableDataset
- Experiments: MLflow
- Evaluation: scikit-learn, custom calibration and segment reports
- Inference benchmark: ONNX Runtime
- Quality gates: pytest
- Reproducibility: Makefile and CLI scripts

## Models

1. Constant CTR baseline
2. Hashed Logistic Regression with FTRL-Proximal
3. DeepFM
4. DCNv2-small
5. DCNv2-large

LightGBM is used only on a chronological sample as a diagnostic baseline, not as a full-data production candidate.

DeepFM and DCNv2-small should be compared under matched embedding and dense parameter budgets. DCNv2-large is treated as a scaling experiment.

## Metrics

Prediction quality:

- LogLoss
- Normalized Cross-Entropy
- ROC-AUC
- PR-AUC
- Brier score
- ECE
- PCOC

Robustness:

- Chronological block performance
- Head, mid, tail categorical frequency segments
- Seen and OOV categorical segments
- Missing-value segments
- Numerical availability segments

Systems:

- Model size
- Peak RSS
- p50, p95, p99 model-only latency
- QPS under fixed benchmark conditions

## Calibration Policy

Calibration is measured only on the sampled Criteo distribution.

Because the original positive and negative sampling rates are unknown, this project will not claim to recover true production CTR. Instead, it will:

- fit Platt scaling or isotonic regression on block 6B;
- evaluate sampled-distribution calibration on block 7;
- run a controlled known-prior-shift experiment where the shift is artificially created and therefore measurable;
- report hypothetical deployment-prior sensitivity only as prediction and threshold sensitivity, not as true ECE or true CTR recovery.

## Five-Week Plan

| Week | Deliverable |
| --- | --- |
| 1 | Hydrate data, convert TSV to sharded Parquet, create chronological blocks, compute train-only stats, add leakage tests |
| 2 | Build Constant CTR, FTRL/LR, LightGBM sample baseline, and evaluation harness |
| 3 | Run preprocessing, hashing, vocabulary, DeepFM, and DCNv2 experiments |
| 4 | Complete calibration, controlled prior-shift, temporal drift, and segment-wise diagnostics |
| 5 | Profile inference, export ONNX, benchmark latency/memory, write final README, model card, and portfolio summary |

Optional weeks 6-7:

- offline replay / shadow comparison document
- A/B test design document
- technical blog post
- auction sensitivity analysis
- larger-scale Spark extension on Criteo 1TB subset

## Parallel Preparation Track

This project does not replace coding-test preparation.

During the five-week core period:

- project work: about 4 hours per day
- algorithms and CS: about 2 hours per day
- one weekly mock test
- weekly review of failed problems and complexity mistakes

## Definition of Done

- Full training pipeline runs out-of-core
- Split is reproducible and order-preserving
- Preprocessing uses train-derived statistics only
- Leakage tests pass
- Baselines and deep models are compared under clear budgets
- Calibration claims are limited to valid distributions
- Segment reports reveal weak regions behind global metrics
- ONNX benchmark uses fixed measurement conditions
- Final report includes failures, limitations, and decision reasons
- Resume and portfolio summaries can be adapted for Ads, Search, and Recommendation roles
