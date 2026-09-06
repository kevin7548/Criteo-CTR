---
name: harness
description: Harness 프레임워크의 phase와 step 실행 계획을 설계하고 phases 인덱스 및 step 파일을 생성·관리한다. 사용자가 구현 계획, 단계별 작업 분해, harness phase 생성 또는 실행 준비를 요청할 때 사용한다.
---

# Harness 워크플로우

이 프로젝트는 Harness 프레임워크를 사용한다. 다음 순서로 작업한다.

## 1. 탐색

`docs/` 아래의 PRD, ARCHITECTURE, ADR 등을 읽고 기획·아키텍처·설계 의도를 파악한다. 탐색을 병렬화할 실익이 있고 현재 환경이 허용하면 서브에이전트를 사용한다.

## 2. 논의

구현 전에 구체화하거나 기술적으로 결정해야 할 사항이 결과를 크게 바꾸면 사용자와 논의한다.

## 3. Step 설계

사용자가 구현 계획 작성을 지시하면 여러 step으로 나눈 초안을 작성해 피드백을 요청한다.

설계 원칙:

1. 하나의 step에서 하나의 레이어나 모듈만 다룬다. 여러 모듈을 동시에 수정해야 하면 step을 나눈다.
2. 각 step 파일은 독립된 Codex 세션에서 실행된다. 외부 대화를 참조하지 말고 필요한 정보를 파일 안에 적는다.
3. 관련 문서와 이전 step에서 생성하거나 수정한 파일 경로를 명시한다.
4. 함수와 클래스는 인터페이스 수준으로 지시하고 구현은 에이전트 재량에 맡긴다. 멱등성, 보안, 데이터 무결성 같은 핵심 규칙은 명시한다.
5. Acceptance Criteria에는 실제 실행 가능한 검증 명령을 넣는다.
6. 주의사항은 `X를 하지 마라. 이유: Y` 형식으로 구체화한다.
7. step 이름은 핵심 작업을 나타내는 kebab-case slug를 사용한다.

## 4. 파일 생성

사용자가 승인하면 다음 파일을 생성한다.

### `phases/index.json`

```json
{
  "phases": [
    { "dir": "0-mvp", "status": "pending" }
  ]
}
```

- `status`는 `pending`, `completed`, `error`, `blocked` 중 하나다.
- 완료, 실패, 차단 타임스탬프는 `execute.py`가 상태 변경 시 기록한다.

### `phases/{task-name}/index.json`

```json
{
  "project": "<프로젝트명>",
  "phase": "<task-name>",
  "steps": [
    { "step": 0, "name": "project-setup", "status": "pending" },
    { "step": 1, "name": "core-types", "status": "pending" },
    { "step": 2, "name": "api-layer", "status": "pending" }
  ]
}
```

- `project`는 `AGENTS.md`를 참고한다.
- `phase`는 task 디렉토리명과 일치시킨다.
- `steps[].step`은 0부터 시작한다.
- `steps[].name`은 kebab-case slug다.
- 초기 상태는 모두 `pending`이다.
- 완료 시 Codex가 `summary`, 실행기가 `completed_at`을 기록한다.
- 실패 시 Codex가 `error_message`, 실행기가 `failed_at`을 기록한다.
- 차단 시 Codex가 `blocked_reason`, 실행기가 `blocked_at`을 기록한다.
- `summary`에는 다음 step에 유용한 생성 파일과 핵심 결정을 한 줄로 남긴다.
- `created_at`과 step의 `started_at`은 실행기가 기록하므로 생성 시 넣지 않는다.

### `phases/{task-name}/step{N}.md`

```markdown
# Step {N}: {이름}

## 읽어야 할 파일

- `/docs/ARCHITECTURE.md`
- `/docs/ADR.md`
- {이전 step에서 생성하거나 수정한 파일 경로}

## 작업

{파일 경로, 인터페이스, 로직, 반드시 지켜야 할 규칙을 포함한 구현 지시}

## Acceptance Criteria

```bash
npm run build
npm test
```

## 검증 절차

1. Acceptance Criteria 명령을 실행한다.
2. ARCHITECTURE.md의 구조, ADR의 기술 선택, AGENTS.md의 CRITICAL 규칙 준수를 확인한다.
3. 결과에 따라 task index의 step을 업데이트한다.
   - 성공: `completed`와 한 줄 `summary`
   - 3회 수정 후 실패: `error`와 구체적인 `error_message`
   - 사용자 개입 필요: `blocked`와 구체적인 `blocked_reason`

## 금지사항

- {하지 말아야 할 것과 이유}
- 기존 테스트를 깨뜨리지 마라.
```

## 5. 실행

```bash
python3 scripts/execute.py {task-name}
python3 scripts/execute.py {task-name} --push
```

실행기는 다음을 자동 처리한다.

- `feat-{task-name}` 브랜치 생성 또는 checkout
- `AGENTS.md`와 `docs/*.md` 가드레일 주입
- 완료된 step의 summary를 다음 step에 전달
- 실패 시 최대 3회 재시도
- 코드 변경과 실행 메타데이터를 분리해 커밋
- 상태별 타임스탬프 기록

에러나 차단을 해결한 뒤 해당 step의 상태를 `pending`으로 되돌리고 오류 필드를 삭제한 다음 재실행한다.
