# 프로젝트: CriteoCTR-Lab

Criteo Kaggle 클릭 로그로 "AUC로는 안 보이지만 LogLoss와 calibration에서 드러나는 문제"를 다루는 CTR 실험 프로젝트. 범위와 실험 계획은 `README.md`, 설계 결정은 `docs/ADR.md`를 따른다.

## 기술 스택
- Python 3.12 (uv로 관리)
- 데이터: Polars, PyArrow, sharded Parquet (out-of-core)
- 학습: PyTorch (Apple Silicon MPS)
- 테스트/린트: pytest, ruff

## 디렉토리 구조
```
src/ctr/
├── data/          # TSV→Parquet 변환, 시간순 블록 분할, vocab, Dataset
├── metrics.py     # LogLoss, AUC, ECE, NE, PCOC (직접 구현)
├── calibration.py # negative downsampling, prior correction
├── models/        # LR, DCNv2
├── train.py
└── evaluate.py
tests/             # src/ 구조를 따라가는 test_<module>.py
configs/           # 실험 설정
reports/           # 결과 표와 그림 (커밋 대상)
data/, artifacts/  # 생성물 (git 제외)
```

## 데이터 규칙
- CRITICAL: `train.txt`만 사용한다. `test.txt`는 라벨이 없다.
- CRITICAL: 분할은 원본 행 순서를 보존한 7개 블록으로 한다. 학습 1–5, 검증 6, 테스트 7. 행을 섞지 않는다.
- CRITICAL: vocab, 빈도, 수치 정규화 통계는 학습 블록(1–5)에서만 계산한다. 검증·테스트 데이터를 보는 통계는 누수다.
- CRITICAL: negative downsampling은 학습 데이터에만 적용한다. 검증·테스트는 원본 분포 그대로 평가한다.
- CRITICAL: calibration 결과는 "Criteo의 이미 샘플링된 분포" 기준이다. 실제 서비스 CTR을 복원한다고 쓰지 않는다.

## 평가 규칙
- 지표는 `src/ctr/metrics.py`의 직접 구현을 쓴다. sklearn은 테스트에서 정답 비교용으로만 쓴다.
- ECE는 샘플 수가 같은 20개 구간을 기본으로 한다. 결과를 본 뒤 구간 방식을 바꾸지 않는다.
- 모델 비교에는 AUC만 쓰지 않는다. LogLoss, NE, ECE, PCOC를 함께 보고한다.

## 개발 프로세스
- CRITICAL: `src/` 아래 모듈은 테스트를 먼저 작성한다 (TDD). `.claude/hooks/tdd-guard.sh`가 `tests/` 아래 `test_<module>.py` 없이 `src/*.py`를 수정하는 것을 막는다.
- 커밋 전 `ruff check`와 `pytest`가 통과해야 한다. `.claude/hooks/pre-commit-checks.sh`가 커밋 시 자동 실행한다.
- 전체 데이터 실험 전에 학습 블록 앞쪽 일부로 먼저 돌려본다.
- 커밋 메시지는 conventional commits 형식을 따른다 (feat:, fix:, docs:, refactor:, test:, chore:).

## 명령어
```bash
uv sync                 # 환경 설치
uv run pytest -q        # 테스트
uv run ruff check .     # 린트
```
