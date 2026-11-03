# CriteoCTR-Lab

Criteo 클릭 로그로 **AUC로는 안 보이지만 LogLoss와 calibration에서 드러나는 문제**를 측정하고 고치는 CTR 예측 실험입니다.

## 핵심 주장

클릭이 아닌 샘플(negative)을 줄여서 학습하면 AUC는 거의 그대로지만, 평균 예측 CTR이 크게 부풀어 오릅니다. 이 문제는 AUC에는 드러나지 않고 LogLoss와 ECE에서만 보입니다. 샘플링 비율을 알면 `logit(q) = logit(p) + log w` 한 줄로 보정할 수 있습니다.

결과는 세 숫자로 정리합니다.

| | 측정 | 기대 |
|---|---|---|
| 1 | AUC: 전체 데이터 모델 vs downsampled 모델 | 거의 같음 |
| 2 | 평균 예측 CTR vs 실제 CTR | downsampled 모델에서 크게 부풀어 오름 |
| 3 | 보정 후 ECE · LogLoss | 전체 데이터 모델 수준으로 회복 (버린 데이터만큼의 차이는 남음) |

## 데이터

- Criteo Display Advertising Challenge (Kaggle) `train.txt`: 약 4,580만 행, 7일치, 라벨 1개 + 수치 피처 13개 + 범주형 피처 26개
- `test.txt`는 라벨이 없어 쓰지 않습니다.
- 분할: 원본 순서를 유지한 7개 블록. 학습 1–5, 검증 6, 테스트 7
- vocab과 정규화 통계는 학습 블록에서만 계산합니다.

## 실험

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | TSV → Parquet 변환, 시간순 분할, 상수 CTR 기준선 | 예정 |
| 2 | LogLoss, AUC, ECE, NE, PCOC 직접 구현 | 예정 |
| 3 | LR 학습 | 예정 |
| 4 | Negative downsampling(`w` ∈ {1, 0.5, 0.25, 0.1}) 후 보정 | 예정 |
| 5 | DCNv2와 LR 비교 (seed 3개, bootstrap 신뢰구간) | 예정 |
| 6 | (선택) 피처 빈도 구간(head/mid/tail/OOV)별 LogLoss · ECE | 예정 |

## 지표

- **LogLoss / NE**: 확률 품질. NE는 상수 CTR 기준선의 LogLoss로 나눈 값입니다.
- **AUC**: 순위 품질. 단조 변환에 불변이라 calibration 문제를 잡지 못합니다.
- **ECE**: 샘플 수가 같은 20개 구간 기준 calibration 오차입니다.
- **PCOC**: 평균 예측 CTR / 실제 CTR.

## 한계

- Criteo 데이터 자체가 이미 샘플링된 분포이고 원래 샘플링 비율은 공개되지 않았습니다. 이 프로젝트의 calibration 결과는 그 샘플링된 분포 기준이며, 실제 서비스 CTR을 복원한다고 주장하지 않습니다.
- 블록은 시간순이지만 정확한 날짜 경계는 아닙니다.
- 데이터에 사용자·세션·경매 정보가 없어서, 이 프로젝트는 전체 랭킹 시스템이 아니라 그 안의 pointwise 점수 모델만 다룹니다.

## 범위 밖

DeepFM, ONNX 추론 벤치마크, latency·memory 비교, LightGBM, FTRL, 랭킹 지표(NDCG 등), 실제 A/B 테스트 주장.

## 설계 결정

[docs/ADR.md](docs/ADR.md)

## 개발

```bash
uv sync
uv run pytest -q
uv run ruff check .
```
