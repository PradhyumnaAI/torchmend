from pathlib import Path

from torchmend.cli.main import main as cli_main
from torchmend.core.pipeline import repair_code, repair_file


def test_clean_script_skips_llm_call():
    called = False

    def fake_llm(prompt: str) -> str:
        nonlocal called
        called = True
        return ""

    res = repair_code("x = 1 + 1\nprint(x)\n", llm_fn=fake_llm)
    assert res.status == "clean"
    assert res.attempts == 0
    assert called is False
    assert res.to_sft_example() is None


def test_runtime_crash_fixed_and_verified_on_first_attempt():
    buggy = "a = 10\nb = 0\nprint(a / b)\n"

    def fake_llm(prompt: str) -> str:
        assert "ZeroDivisionError" in prompt
        return (
            "<<<<<<< SEARCH\n"
            "b = 0\n"
            "=======\n"
            "b = 2\n"
            ">>>>>>> REPLACE\n"
        )

    res = repair_code(buggy, llm_fn=fake_llm, filename="calc.py")
    assert res.status == "fixed"
    assert res.attempts == 1
    assert res.patched_code == "a = 10\nb = 2\nprint(a / b)\n"
    assert "-b = 0" in res.diff
    assert "+b = 2" in res.diff

    sft = res.to_sft_example()
    assert sft is not None
    assert "ZeroDivisionError" in sft["prompt"]
    assert "<<<<<<< SEARCH" in sft["completion"]


def test_retry_loop_recovers_after_bad_first_patch():
    buggy = "a = 10\nb = 0\nprint(a / b)\n"
    prompts_seen: list[str] = []

    def flaky_llm(prompt: str) -> str:
        prompts_seen.append(prompt)
        if len(prompts_seen) == 1:
            return "<<<<<<< SEARCH\nnonexistent = 1\n=======\nb = 2\n>>>>>>> REPLACE"
        return "<<<<<<< SEARCH\nb = 0\n=======\nb = 5\n>>>>>>> REPLACE"

    res = repair_code(buggy, llm_fn=flaky_llm, max_attempts=2)
    assert res.status == "fixed"
    assert res.attempts == 2
    assert "Previous Attempt Feedback" in prompts_seen[1]


def test_verification_rejects_patch_that_still_crashes():
    buggy = "x = 1 / 0\n"

    def bad_llm(prompt: str) -> str:
        return "<<<<<<< SEARCH\nx = 1 / 0\n=======\nx = 2 / 0\n>>>>>>> REPLACE"

    res = repair_code(buggy, llm_fn=bad_llm, max_attempts=2)
    assert res.status == "failed"
    assert res.attempts == 2
    assert res.to_sft_example() is None


def test_repair_file_respects_apply_flag(tmp_path: Path):
    script = tmp_path / "demo.py"
    script.write_text("x = 1 / 0\n", encoding="utf-8")

    def fixer(_prompt: str) -> str:
        return "<<<<<<< SEARCH\nx = 1 / 0\n=======\nx = 1 / 1\n>>>>>>> REPLACE"

    dry = repair_file(script, llm_fn=fixer, apply=False)
    assert dry.status == "fixed"
    assert script.read_text(encoding="utf-8") == "x = 1 / 0\n"

    applied = repair_file(script, llm_fn=fixer, apply=True)
    assert applied.status == "fixed"
    assert script.read_text(encoding="utf-8") == "x = 1 / 1\n"


def test_cli_check_command_reports_clean_and_buggy(tmp_path: Path):
    ok_script = tmp_path / "ok.py"
    ok_script.write_text("x = 2 + 2\n", encoding="utf-8")
    assert cli_main(["check", str(ok_script), "--run"]) == 0

    bad_script = tmp_path / "bad.py"
    bad_script.write_text("x = 1 / 0\n", encoding="utf-8")
    assert cli_main(["check", str(bad_script), "--run"]) == 1