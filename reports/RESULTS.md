# CriteoCTR-Lab 결과 (1–5단계)

Criteo Kaggle `train.txt` 45,840,617행을 원본 순서대로 7개 블록으로 나눠 학습 1–5 / 검증 6 / 테스트 7로 쓴다(ADR-001). 모든 지표는 `src/ctr/metrics.py` 직접 구현이고, NE 분모는 학습 블록 CTR(0.25539)을 상수로 예측한 LogLoss다(ADR-008). ECE는 샘플 수가 같은 20개 구간이다(ADR-004). 표의 수치는 별도 표시가 없으면 **테스트 블록 7**(6,548,660행) 기준이다.

> calibration 결과는 모두 Criteo가 공개한, 이미 샘플링된 분포 기준이다. 실제 서비스 CTR을 복원한다는 뜻이 아니다.

## 한눈에

| | 결과 |
|---|---|
| 기준선 | 상수 CTR 예측 LogLoss 0.57278 (NE 1, AUC 0.5) |
| LR (3단계) | LogLoss 0.46184, NE 0.8063, AUC 0.7909, ECE 0.0041, PCOC 0.993 |
| downsampling (4단계, w = 0.1) | AUC 0.7909 → 0.7886, 평균 pCTR 2.55배, 보정 후 ECE 0.402 → 0.0088, LogLoss 0.906 → 0.464 |
| DCNv2 vs LR (5단계, seed 3개 평균) | LogLoss 0.44825 vs 0.46171 (Δ −0.0135), AUC 0.8064 vs 0.7909 (Δ +0.0155), 그러나 ECE 0.0079 vs 0.0028 |

---

## 1단계: 데이터 분할과 상수 CTR 기준선

`train.txt`를 한 번 스트리밍으로 읽어 블록별 Parquet shard(최대 100만 행, zstd)로 저장했다(ADR-007). 블록 경계는 `k·N // 7`이라 블록 크기 차이가 최대 1행이다.

| 블록 | 용도 | 행 수 | 양성 | CTR |
|---|---|---|---|---|
| 1 | 학습 | 6,548,659 | 1,627,580 | 0.24854 |
| 2 | 학습 | 6,548,660 | 1,680,792 | 0.25666 |
| 3 | 학습 | 6,548,659 | 1,670,406 | 0.25508 |
| 4 | 학습 | 6,548,660 | 1,659,866 | 0.25347 |
| 5 | 학습 | 6,548,659 | 1,723,519 | 0.26319 |
| 6 | 검증 | 6,548,660 | 1,682,652 | 0.25695 |
| 7 | 테스트 | 6,548,660 | 1,700,623 | 0.25969 |
| 1–5 | 학습 합계 | 32,743,297 | 8,362,163 | 0.25539 |

상수 기준선은 모든 행에 학습 블록 CTR 0.25539를 예측한다(`reports/baseline_constant.json`).

| split | LogLoss | 자기 CTR의 entropy | PCOC |
|---|---|---|---|
| 학습 1–5 | 0.56817 | 0.56817 | 1.000 |
| 검증 6 | 0.56984 | 0.56984 | 0.994 |
| 테스트 7 | 0.57278 | 0.57273 | 0.983 |

테스트 블록의 CTR이 학습보다 1.7% 높다. 그래서 학습 분포에 정확히 맞춘 모델도 테스트에서는 PCOC가 1보다 조금 낮게 나온다. 아래 표의 PCOC 0.97–0.99는 대부분 이 시간에 따른 CTR 변화 때문이다.

## 2단계: 지표 직접 구현

LogLoss, NE, AUC(동점은 평균 순위), ECE, PCOC를 `src/ctr/metrics.py`에 구현하고 테스트 87개로 확인했다. sklearn(`log_loss`, `roc_auc_score`, `calibration_curve`)은 테스트에서 정답 비교용으로만 쓴다(ADR-003).

- **NE 분모**는 학습 블록 CTR이다. 평가 split의 라벨로 분모를 만들지 않는다.
- **ECE 동점 처리**: 같은 예측값 묶음은 쪼개지 않고 가운데 행이 속한 구간에 넣는다. 위치대로 자르면 상수 예측의 ECE가 라벨 순서(=시간 순서)에 따라 0.4까지 흔들리는 것을 테스트로 확인했다(ADR-008).
- **AUC 단조 불변성**: 예측을 부풀려도 AUC는 그대로고 LogLoss와 ECE만 나빠지는 것을 테스트로 확인했다. 4단계의 실제 데이터 결과가 이 성질을 그대로 보여준다.

## 3단계: LR

39개 필드(수치 13개는 `floor(ln(x)²)` 구간화)를 모두 범주형 id로 바꾸고, 학습 블록에서 10번 이상 나온 값만 vocab에 남겼다(ADR-009, id 832,960개). LR은 1차원 embedding 합 + bias다(ADR-006). Adam, 배치 4096, shard 5개 단위 셔플(ADR-010). 학습률은 앞쪽 5개 shard에서 {3e-4, 1e-3, 3e-3, 1e-2} 중 1e-3을 골랐다(`reports/lr_smoke_sweep.log`). 검증 LogLoss가 가장 낮은 2번째 epoch을 남겼다.

| split | LogLoss | NE | AUC | ECE | PCOC |
|---|---|---|---|---|---|
| 상수 CTR (테스트) | 0.57278 | 1 | 0.5 | – | 0.983 |
| LR 검증 | 0.45745 | 0.8028 | 0.7938 | 0.00326 | 1.0006 |
| LR 테스트 | 0.46184 | 0.8063 | 0.7909 | 0.00409 | 0.9925 |

downsampling 없이 학습한 LR은 거의 calibrated 되어 있다. 4단계의 기준 모델이다.

## 4단계: Negative downsampling과 prior correction

학습 블록에서만 양성은 모두, 음성은 비율 `w`로 남겨 LR을 다시 학습했다(ADR-011). 남길 행은 shard마다 고정이라 모든 epoch에서 같은 데이터를 본다. 검증·테스트는 원본 분포 전체로 평가한다. 보정은 학습 없이 `logit(q) = logit(p) + log w` 한 줄이다(ADR-005). 학습률 등 다른 설정은 `w = 1`과 같다.

| w | 학습 행 | 샘플 CTR | | LogLoss | NE | AUC | ECE | PCOC | 평균 pCTR |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 32,743,297 | 0.2554 | – | 0.46184 | 0.8063 | 0.7909 | 0.00409 | 0.9925 | 0.2577 |
| 0.5 | 20,552,331 | 0.4069 | 보정 전 | 0.49723 | 0.8681 | 0.7909 | 0.10921 | 1.4206 | 0.3689 |
| | | | 보정 후 | 0.46184 | 0.8063 | 0.7909 | 0.00487 | 0.9854 | 0.2559 |
| 0.25 | 14,456,662 | 0.5784 | 보정 전 | 0.61449 | 1.0728 | 0.7904 | 0.23442 | 1.9027 | 0.4941 |
| | | | 보정 후 | 0.46247 | 0.8074 | 0.7904 | 0.00663 | 0.9791 | 0.2543 |
| 0.1 | 10,799,797 | 0.7743 | 보정 전 | 0.90582 | 1.5814 | 0.7886 | 0.40197 | 2.5479 | 0.6617 |
| | | | 보정 후 | 0.46418 | 0.8104 | 0.7886 | 0.00876 | 0.9706 | 0.2521 |

(실제 테스트 CTR 0.2597. 모든 `w`에서 검증 기준 최적 epoch은 2.)

README의 세 가지 주장에 대한 결과:

1. **AUC는 거의 그대로다.** w = 0.1에서도 0.7909 → 0.7886(−0.0023)이다. 보정은 단조 변환이라 보정 전후 AUC가 소수점 끝자리까지 같다.
2. **평균 예측 CTR은 크게 부풀어 오른다.** PCOC 1.42 / 1.90 / 2.55배로, 샘플 CTR ÷ 원래 CTR(1.59 / 2.26 / 3.03)보다 조금 작은 정도다. 보정 전 NE는 w = 0.25부터 1을 넘어, 상수 예측보다 나쁜 모델이 된다. 그 사이 AUC는 0.79 그대로다.
3. **보정 후 ECE·LogLoss는 w = 1 수준으로 돌아오고, 버린 데이터만큼의 차이만 남는다.** w = 0.1에서 ECE 0.402 → 0.0088, LogLoss 0.906 → 0.464다.

`w = 1` 모델과의 차이(보정 후, 테스트 블록, paired bootstrap 1000회 95% 구간, ADR-013, `reports/downsampling/compare_test.json`):

| w | ΔLogLoss | ΔAUC | ΔPCOC |
|---|---|---|---|
| 0.5 | −0.00000 [−0.00004, +0.00003] | +0.00007 [+0.00002, +0.00011] | −0.0071 [−0.0072, −0.0071] |
| 0.25 | +0.00062 [+0.00057, +0.00068] | −0.00045 [−0.00052, −0.00039] | −0.0134 [−0.0135, −0.0133] |
| 0.1 | +0.00234 [+0.00225, +0.00241] | −0.00225 [−0.00234, −0.00216] | −0.0219 [−0.0220, −0.0218] |

- w = 0.5는 음성의 절반(학습 행의 37%)을 버려도 LogLoss 차이가 0과 구분되지 않는다.
- w = 0.1은 학습 행을 1/3로 줄이고 LogLoss +0.0023(0.5%), AUC −0.0023만큼 손해를 본다. 구간이 0을 포함하지 않으므로 실제로 있는 차이지만 작다.
- 보정 후 PCOC는 w가 작을수록 1에서 조금씩 더 멀어진다(0.993 → 0.971). 테스트 CTR 상승(1단계)에 더해, prior correction이 "학습 분포에서 calibrated"라는 가정(ADR-005)에 기대기 때문이다. 이 차이를 더 줄이려면 검증 블록으로 맞추는 Platt scaling 같은 사후 보정이 필요하다. 이 프로젝트에서는 하지 않았다.

## 5단계: DCNv2 vs LR

DCNv2(parallel, embedding 16, full-rank cross 3층, MLP 400×3, 파라미터 15,069,169개, ADR-012)를 LR과 같은 데이터·학습 루프·epoch 선택 규칙으로 학습했다. 학습률은 앞쪽 5개 shard에서 {3e-4, 1e-3, 3e-3, 1e-2} 중 검증 LogLoss가 가장 낮은 3e-3을 골랐다(`reports/dcnv2_smoke_sweep.log`). 두 모델 모두 seed 0, 1, 2로 학습했다. LR의 seed는 셔플 순서만 바꾸고(weight는 0에서 시작), DCNv2의 seed는 초기화와 셔플 순서를 바꾼다.

**seed 3개 평균 ± 표준편차**

| 모델 | split | LogLoss | NE | AUC | ECE | PCOC |
|---|---|---|---|---|---|---|
| LR | 검증 | 0.45743 ± 0.00013 | 0.8027 | 0.79370 ± 0.00013 | 0.00356 ± 0.00028 | 1.010 |
| DCNv2 | 검증 | 0.44394 ± 0.00046 | 0.7791 | 0.80912 ± 0.00035 | 0.00850 ± 0.00195 | 1.026 |
| LR | 테스트 | 0.46171 ± 0.00018 | 0.8061 | 0.79091 ± 0.00014 | 0.00281 ± 0.00112 | 1.001 |
| DCNv2 | 테스트 | 0.44825 ± 0.00049 | 0.7826 | 0.80640 ± 0.00044 | 0.00787 ± 0.00193 | 1.019 |

seed별 테스트 값(`reports/lr.json`, `reports/model_comparison/*.json`):

| seed | LR LogLoss | DCNv2 LogLoss | LR AUC | DCNv2 AUC | LR ECE | DCNv2 ECE | 남긴 epoch (LR / DCNv2) |
|---|---|---|---|---|---|---|---|
| 0 | 0.46184 | 0.44880 | 0.79087 | 0.80593 | 0.00409 | 0.01009 | 2 / 1 |
| 1 | 0.46150 | 0.44785 | 0.79106 | 0.80681 | 0.00200 | 0.00664 | 1 / 1 |
| 2 | 0.46178 | 0.44809 | 0.79079 | 0.80647 | 0.00234 | 0.00686 | 1 / 1 |

**같은 seed끼리의 차이 DCNv2 − LR**(테스트 블록, paired bootstrap 1000회 95% 구간, ADR-013, `reports/model_comparison/compare_test.json`):

| seed | ΔLogLoss | ΔAUC | ΔPCOC |
|---|---|---|---|
| 0 | −0.01305 [−0.01319, −0.01290] | +0.01506 [+0.01488, +0.01523] | +0.0341 [+0.0338, +0.0343] |
| 1 | −0.01365 [−0.01379, −0.01349] | +0.01575 [+0.01558, +0.01591] | +0.0071 [+0.0069, +0.0073] |
| 2 | −0.01369 [−0.01384, −0.01354] | +0.01568 [+0.01551, +0.01584] | +0.0123 [+0.0120, +0.0125] |

해석:

- **DCNv2가 확률 품질과 순위 모두에서 낫다.** LogLoss는 2.9%(NE 0.806 → 0.783) 낮고 AUC는 0.0155 높다. seed 3개 모두 구간이 0에서 멀리 떨어져 있고, seed 간 표준편차(LogLoss 0.0005)도 차이(0.0135)의 1/25 수준이다. seed 간 표준편차는 bootstrap 95% 구간의 반폭(약 0.00015)의 3배라, 불확실성은 테스트 행보다 학습 seed에서 더 크게 온다.
- **그런데 calibration은 DCNv2가 더 나쁘다.** ECE가 LR의 약 2.8배(0.0079 vs 0.0028)이고, seed마다 0.0066–0.0101로 흔들린다. seed 1·2는 낮은 구간을 0.003–0.004 낮게, 가장 높은 구간들을 최대 0.018(seed 2)–0.026(seed 1) 높게 예측한다. 예측이 양 끝으로 지나치게 벌어진(과신) 모양이다. seed 0은 중간 구간을 최대 0.016 높게 예측하고, 평균도 3.4% 높다(PCOC 1.027). LR(seed 1)은 구간 오차가 대부분 0.003 안쪽이다(가장 높은 두 구간만 0.007–0.011).
- 즉 AUC와 LogLoss가 함께 좋아져도 calibration은 따로 확인해야 한다. 4단계의 downsampling처럼 극적이지는 않지만, "AUC로는 보이지 않는 확률 품질 문제"가 모델 구조를 바꿀 때도 생긴다. 검증 블록으로 맞추는 사후 보정(Platt/temperature scaling)이 DCNv2에 효과가 있을 것으로 보이지만, 이 프로젝트에서는 측정하지 않았다.
- **DCNv2는 첫 epoch에서 이미 최적이다.** 세 seed 모두 2번째 epoch에서 학습 LogLoss는 0.446 → 0.431로 내려가고 검증 LogLoss는 올라갔다(CTR 딥러닝 모델에서 흔히 보고되는 1-epoch 과적합). LR은 epoch 1과 2의 검증 LogLoss 차이가 0.0004 안쪽이다.
- **비용**: Apple M1 MPS에서 DCNv2 epoch당 약 12분(LR 약 67초), 파라미터 18배(1,507만 vs 83만).
- 참고: 공개 벤치마크(BARS 등)의 Criteo AUC 0.81대는 무작위 분할 기준이다. 이 프로젝트는 시간순 분할이라(ADR-001) 수치를 직접 비교할 수 없다.

## 재현

```bash
uv run python -m ctr.data.split --src kaggle-display-advertising-challenge-dataset/train.txt --out data/blocks
uv run python -m ctr.baseline
uv run python -m ctr.data.encode --blocks data/blocks --out data/encoded --min-count 10
uv run python -m ctr.train --config configs/lr.toml                     # 3단계
for w in 050 025 010; do uv run python -m ctr.train --config configs/lr_w$w.toml; done   # 4단계
for c in dcnv2_s0 dcnv2_s1 dcnv2_s2 lr_s1 lr_s2; do uv run python -m ctr.train --config configs/$c.toml; done  # 5단계
uv run python -m ctr.compare --split test --out reports/downsampling/compare_test.json \
  --run w1=reports/lr.json --run w050=reports/downsampling/lr_w050.json \
  --run w025=reports/downsampling/lr_w025.json --run w010=reports/downsampling/lr_w010.json \
  --pair w050:w1 --pair w025:w1 --pair w010:w1
uv run python -m ctr.compare --split test --out reports/model_comparison/compare_test.json \
  --run lr_s0=reports/lr.json --run lr_s1=reports/model_comparison/lr_s1.json \
  --run lr_s2=reports/model_comparison/lr_s2.json --run dcnv2_s0=reports/model_comparison/dcnv2_s0.json \
  --run dcnv2_s1=reports/model_comparison/dcnv2_s1.json --run dcnv2_s2=reports/model_comparison/dcnv2_s2.json \
  --pair dcnv2_s0:lr_s0 --pair dcnv2_s1:lr_s1 --pair dcnv2_s2:lr_s2
```

환경: Apple M1 (16GB), PyTorch MPS. MPS 연산은 실행마다 비트 단위로 같지 않다(ADR-010).
