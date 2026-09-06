#!/usr/bin/env bash
# TDD Guard Hook — PreToolUse[Edit|Write|apply_patch]
# Blocks implementation edits when no corresponding test file exists.
#
# Adapted from:
# https://github.com/jha0313/codex-live-demo/blob/feat-0-mvp/.codex/hooks/tdd-guard.sh
#
# Codex apply_patch inputs carry changed paths in .tool_input.command. Direct
# Edit/Write-compatible inputs may instead provide .tool_input.file_path.

set -u

INPUT=$(cat)

if [ -z "$INPUT" ]; then
  exit 0
fi

PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)

deny() {
  local reason="$1"
  python3 - "$reason" <<'PY'
import json
import sys

print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": sys.argv[1],
    }
}, ensure_ascii=False))
PY
}

PATHS=$(
  python3 - "$INPUT" <<'PY'
import json
import re
import sys

try:
    payload = json.loads(sys.argv[1])
except Exception:
    sys.exit(0)

tool_input = payload.get("tool_input") or {}
items = []

for key in ("file_path", "path", "filename"):
    value = tool_input.get(key)
    if isinstance(value, str) and value:
        items.append(("update", value))

command = tool_input.get("command") or tool_input.get("cmd") or ""
if isinstance(command, str):
    for line in command.splitlines():
        match = re.match(r"^\*\*\* (Add|Update|Delete) File: (.+)$", line)
        if match:
            items.append((match.group(1).lower(), match.group(2).strip()))
            continue
        match = re.match(r"^\*\*\* Move to: (.+)$", line)
        if match:
            items.append(("update", match.group(1).strip()))

seen = set()
for action, path in items:
    key = (action, path)
    if key in seen:
        continue
    seen.add(key)
    print(f"{action}\t{path}")
PY
)

if [ -z "$PATHS" ]; then
  exit 0
fi

has_javascript_test_for() {
  local file_path="$1"
  local dir_name base_name parent_dir ext

  dir_name=$(dirname "$file_path")
  base_name=$(basename "$file_path" | sed -E 's/\.(ts|tsx|js|jsx)$//')
  parent_dir=$(dirname "$dir_name")

  for ext in ts tsx js jsx; do
    [ -f "${dir_name}/${base_name}.test.${ext}" ] && return 0
    [ -f "${dir_name}/${base_name}.spec.${ext}" ] && return 0
    [ -f "${dir_name}/__tests__/${base_name}.test.${ext}" ] && return 0
    [ -f "${dir_name}/__tests__/${base_name}.spec.${ext}" ] && return 0
    [ -f "${parent_dir}/__tests__/${base_name}.test.${ext}" ] && return 0
    [ -f "${parent_dir}/__tests__/${base_name}.spec.${ext}" ] && return 0
    [ -f "${PROJECT_ROOT}/src/__tests__/${base_name}.test.${ext}" ] && return 0
    [ -f "${PROJECT_ROOT}/src/__tests__/${base_name}.spec.${ext}" ] && return 0
  done

  return 1
}

has_python_test_for() {
  local file_path="$1"
  local dir_name base_name parent_dir

  dir_name=$(dirname "$file_path")
  base_name=$(basename "$file_path" .py)
  parent_dir=$(dirname "$dir_name")

  [ -f "${dir_name}/test_${base_name}.py" ] && return 0
  [ -f "${dir_name}/${base_name}_test.py" ] && return 0
  [ -f "${dir_name}/tests/test_${base_name}.py" ] && return 0
  [ -f "${parent_dir}/tests/test_${base_name}.py" ] && return 0
  [ -f "${PROJECT_ROOT}/tests/test_${base_name}.py" ] && return 0
  [ -f "${PROJECT_ROOT}/scripts/test_${base_name}.py" ] && return 0

  return 1
}

while IFS=$'\t' read -r action file_path; do
  [ -z "$file_path" ] && continue
  [ "$action" = "delete" ] && continue

  case "$file_path" in
    *test*|*spec*|*.test.*|*.spec.*|*__tests__*) continue ;;
  esac

  case "$file_path" in
    *.json|*.css|*.scss|*.md|*.yml|*.yaml|*.env*|*.config.*|*tailwind*|*postcss*|*next.config*|*tsconfig*) continue ;;
  esac

  case "$file_path" in
    */types/*|*/types.ts|*/types.d.ts|*/__init__.py) continue ;;
  esac

  case "$file_path" in
    */layout.tsx|*/layout.ts|*/page.tsx|*/page.ts|*/loading.tsx|*/error.tsx|*/not-found.tsx|*/globals.css) continue ;;
  esac

  case "$file_path" in
    *.ts|*.tsx|*.js|*.jsx)
      if ! has_javascript_test_for "$file_path"; then
        base_name=$(basename "$file_path" | sed -E 's/\.(ts|tsx|js|jsx)$//')
        deny "TDD GUARD: '${base_name}'에 대한 테스트 파일이 존재하지 않습니다. 구현 코드를 작성하기 전에 테스트를 먼저 작성하세요. (테스트 파일 예: ${base_name}.test.ts)"
        exit 0
      fi
      ;;
    *.py)
      if ! has_python_test_for "$file_path"; then
        base_name=$(basename "$file_path" .py)
        deny "TDD GUARD: '${base_name}'에 대한 테스트 파일이 존재하지 않습니다. 구현 코드를 작성하기 전에 테스트를 먼저 작성하세요. (테스트 파일 예: test_${base_name}.py)"
        exit 0
      fi
      ;;
  esac
done <<< "$PATHS"

exit 0
