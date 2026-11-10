import json
import os
import shlex
import subprocess
from pathlib import Path

HOOK = Path(__file__).parents[2] / ".claude" / "hooks" / "pre-commit-checks.sh"


def run_hook(tmp_path: Path, command: str, **checks: str):
    env = os.environ.copy()
    env.update(checks)
    return subprocess.run(
        ["bash", str(HOOK)],
        input=json.dumps({"tool_name": "Bash", "tool_input": {"command": command}}),
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=env,
        check=False,
    )


def append_command(log_file: Path, value: str) -> str:
    return f"printf '{value}\\n' >> {shlex.quote(str(log_file))}"


def test_ignores_non_commit_commands(tmp_path):
    result = run_hook(
        tmp_path,
        "git status",
        CHECK_LINT_CMD="exit 99",
        CHECK_TEST_CMD="exit 99",
    )

    assert result.returncode == 0
    assert result.stdout == ""


def test_runs_lint_then_test_before_commit(tmp_path):
    log_file = tmp_path / "checks.log"

    result = run_hook(
        tmp_path,
        "git add -A && git commit -m 'test'",
        CHECK_LINT_CMD=append_command(log_file, "lint"),
        CHECK_TEST_CMD=append_command(log_file, "test"),
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert log_file.read_text().splitlines() == ["lint", "test"]


def test_denies_commit_on_first_failed_check(tmp_path):
    log_file = tmp_path / "checks.log"

    result = run_hook(
        tmp_path,
        "git commit --amend",
        CHECK_LINT_CMD="printf 'lint failed' >&2; exit 7",
        CHECK_TEST_CMD=append_command(log_file, "test"),
    )

    assert result.returncode == 0
    decision = json.loads(result.stdout)["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"
    assert "lint" in decision["permissionDecisionReason"]
    assert "lint failed" in decision["permissionDecisionReason"]
    assert not log_file.exists()
