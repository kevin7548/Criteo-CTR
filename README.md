# CriteoCTR-Lab

Large-scale sparse click log experiments for pointwise ad click scoring.

## Project Goal

This project studies whether feature-interaction models such as DeepFM and DCNv2 justify their extra inference cost over linear CTR baselines on sparse advertising logs.

The project is not presented as a full ranking system because the Criteo dataset does not include query, session, request, user, auction, or candidate-list identifiers. Instead, it focuses on a pointwise scoring layer commonly used inside ranking and advertising systems.

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
| 5 | Profile inference, export ONNX, benchmark latency/memory, write final README and model card |

Optional weeks 6-7:

- offline replay / shadow comparison document
- A/B test design document
- technical blog post
- auction sensitivity analysis
- larger-scale Spark extension on Criteo 1TB subset

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

## 한국어

대규모 sparse click log를 사용해 광고 클릭 여부를 점별(pointwise)로 예측하는 실험 프로젝트입니다.

## 프로젝트 목표

이 프로젝트는 DeepFM, DCNv2 같은 feature-interaction 모델이 sparse 광고 로그에서 선형 CTR baseline보다 더 나은 예측 품질을 제공하는지, 그리고 그 개선이 추가 inference 비용을 정당화할 만큼 충분한지 검증합니다.

Criteo 데이터셋에는 query, session, request, user, auction, candidate-list 식별자가 포함되어 있지 않으므로 이 프로젝트는 전체 ranking system으로 제시하지 않습니다. 대신 ranking 및 advertising system 내부에서 흔히 사용되는 pointwise scoring layer에 초점을 둡니다.

## 핵심 질문

비선형 feature-interaction 모델은 선형 baseline 대비 클릭 예측 품질을 얼마나 개선하며, 어떤 조건에서 추가 latency와 memory 비용을 감수할 가치가 있는가?

## 데이터

- Dataset: Criteo Display Advertising Challenge
- Format: label, 13개 numerical feature, 26개 categorical feature
- Split policy: 원본 순서를 유지한 chronological block 분할
- Train: blocks 1-5
- Validation: block 6A
- Calibration fitting: block 6B
- Final test: block 7

각 block은 calendar day가 아니라 시간 순서를 보존한 chronological block입니다. 따라서 이 프로젝트는 정확한 재학습 주기가 아니라 temporal distance와 drift pattern만 다룹니다.

## 범위

포함:

- Sharded Parquet 기반 out-of-core preprocessing
- Train-only preprocessing statistics
- Leakage tests
- Constant CTR 및 linear baselines
- DeepFM 및 DCNv2 experiments
- Sampled data distribution 기준 calibration
- Controlled prior-shift sensitivity experiment
- Segment-wise robustness diagnostics
- Model quality, latency, memory, size trade-off analysis
- Reproducible command harness

핵심 범위에서 제외:

- NDCG, Recall@K, GAUC
- CVR, ROAS, auction outcome, true production CTR recovery
- 실제 A/B test claim
- Spark, Airflow, Kafka, Kubernetes, TorchRec

Spark는 단일 노드 처리가 실험적으로 충분하지 않은 Criteo 1TB subset으로 확장할 때 별도 extension으로 검토할 수 있습니다.

## 기술 스택

- Data: Polars, PyArrow, sharded Parquet
- Training: PyTorch, IterableDataset
- Experiments: MLflow
- Evaluation: scikit-learn, custom calibration and segment reports
- Inference benchmark: ONNX Runtime
- Quality gates: pytest
- Reproducibility: Makefile and CLI scripts

## 모델

1. Constant CTR baseline
2. Hashed Logistic Regression with FTRL-Proximal
3. DeepFM
4. DCNv2-small
5. DCNv2-large

LightGBM은 full-data production candidate가 아니라 chronological sample에서 diagnostic baseline으로만 사용합니다.

DeepFM과 DCNv2-small은 matched embedding 및 dense parameter budget 아래에서 비교합니다. DCNv2-large는 scaling experiment로 취급합니다.

## 지표

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

## Calibration 정책

Calibration은 sampled Criteo distribution에서만 측정합니다.

원본 positive/negative sampling rate를 알 수 없기 때문에 이 프로젝트는 true production CTR을 복원한다고 주장하지 않습니다. 대신 다음을 수행합니다.

- block 6B에서 Platt scaling 또는 isotonic regression fitting
- block 7에서 sampled-distribution calibration 평가
- 인위적으로 생성한 prior shift에 대한 controlled experiment 수행
- hypothetical deployment-prior sensitivity는 true ECE나 true CTR recovery가 아니라 prediction 및 threshold sensitivity로만 보고

## 5주 계획

| Week | Deliverable |
| --- | --- |
| 1 | Hydrate data, convert TSV to sharded Parquet, create chronological blocks, compute train-only stats, add leakage tests |
| 2 | Build Constant CTR, FTRL/LR, LightGBM sample baseline, and evaluation harness |
| 3 | Run preprocessing, hashing, vocabulary, DeepFM, and DCNv2 experiments |
| 4 | Complete calibration, controlled prior-shift, temporal drift, and segment-wise diagnostics |
| 5 | Profile inference, export ONNX, benchmark latency/memory, write final README and model card |

선택 확장:

- offline replay / shadow comparison document
- A/B test design document
- technical blog post
- auction sensitivity analysis
- larger-scale Spark extension on Criteo 1TB subset

## 완료 기준

- 전체 training pipeline이 out-of-core 방식으로 실행됨
- Split이 재현 가능하며 원본 순서를 보존함
- Preprocessing이 train-derived statistics만 사용함
- Leakage tests 통과
- Baseline과 deep model이 명확한 budget 조건에서 비교됨
- Calibration claim이 유효한 distribution 범위로 제한됨
- Segment report가 global metric 뒤에 가려진 취약 구간을 드러냄
- ONNX benchmark가 고정된 측정 조건을 사용함
- Final report가 실패 사례, 한계, 의사결정 이유를 포함함
