from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from torchmend.core.ast_slicer import slice_code
from torchmend.core.patcher import apply_patch
from torchmend.core.runtime_capture import RuntimeReport, capture_code
from torchmend.core.static_rules import StaticIssue, analyze_code

SYSTEM_PROMPT = (
    "You are TorchMend, a precision Python and PyTorch debugger. "
    "Fix the reported bug(s) using ONLY strict SEARCH/REPLACE blocks:\n"
    "<<<<<<< SEARCH\n"
    "<exact existing lines from source>\n"
    "=======\n"
    "<fixed lines>\n"
    ">>>>>>> REPLACE\n"
    "Make the smallest possible change. Do not invent line numbers."
)


@dataclass
class RepairResult:
    status: str  # "clean" | "fixed" | "failed"
    original_code: str
    patched_code: str | None = None
    diff: str = ""
    attempts: int = 0
    initial_static_issues: list[StaticIssue] = field(default_factory=list)
    initial_runtime: RuntimeReport | None = None
    final_static_issues: list[StaticIssue] = field(default_factory=list)
    final_runtime: RuntimeReport | None = None
    prompt_used: str = ""
    raw_llm_output: str = ""
    error: str | None = None

    def to_sft_example(self) -> dict[str, str] | None:
        """Exports a verified (prompt, completion) pair for Student LLM SFT."""
        if self.status != "fixed" or not self.prompt_used or not self.raw_llm_output:
            return None
        return {
            "system": SYSTEM_PROMPT,
            "prompt": self.prompt_used,
            "completion": self.raw_llm_output,
        }


def format_static_issues(issues: list[StaticIssue]) -> str:
    """Formats static analysis findings into a compact prompt block."""
    if not issues:
        return "Static Analysis: No issues detected."
    lines = ["Static Analysis Findings:"]
    for issue in issues:
        lines.append(
            f"  - [{issue.rule_id}] ({issue.severity}, line {issue.line}): "
            f"{issue.message} Suggestion: {issue.suggestion}"
        )
    return "\n".join(lines)


def build_repair_prompt(
    source: str,
    static_issues: list[StaticIssue],
    runtime_report: RuntimeReport | None,
    previous_feedback: str | None = None,
) -> str:
    """Builds the canonical grounded prompt for both Teacher and Student LLMs."""
    target_lines: list[int] = [
        issue.line for issue in static_issues if issue.line and issue.line > 0
    ]
    if runtime_report and runtime_report.crash_line:
        target_lines.append(runtime_report.crash_line)

    sliced = slice_code(source, target_lines=target_lines or None)

    sections = [
        format_static_issues(static_issues),
        runtime_report.to_prompt_block()
        if runtime_report is not None
        else "Runtime Capture: skipped.",
        sliced.to_prompt_block(),
    ]
    if previous_feedback:
        sections.append(
            "Previous Attempt Feedback (fix this in your next patch):\n"
            f"{previous_feedback}"
        )
    sections.append("Return ONLY the SEARCH/REPLACE patch block(s) needed to fix this.")
    return "\n\n".join(sections)


def _is_verified_clean(
    static_issues: list[StaticIssue],
    runtime_report: RuntimeReport | None,
    fix_warnings: bool = False,
) -> bool:
    blocking_static = [
        issue
        for issue in static_issues
        if fix_warnings or issue.severity == "error"
    ]
    if blocking_static:
        return False
    return runtime_report is None or runtime_report.success


def make_openai_compatible_backend(
    model: str,
    base_url: str = "http://localhost:11434/v1",
    api_key: str | None = None,
    temperature: float = 0.0,
    timeout_sec: float = 60.0,
) -> Callable[[str], str]:
    """Creates a zero-dependency LLM caller for Ollama, Groq, vLLM, or OpenAI."""
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    token = api_key or os.environ.get("TORCHMEND_API_KEY") or "local"

    def _call(prompt: str) -> str:
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request to {endpoint} failed: {exc}") from exc
        return str(body["choices"][0]["message"]["content"])

    return _call


def repair_code(
    code: str,
    llm_fn: Callable[[str], str],
    filename: str = "script.py",
    run_runtime: bool = True,
    fix_warnings: bool = False,
    max_attempts: int = 2,
    timeout_sec: float = 10.0,
) -> RepairResult:
    """Runs diagnose -> slice -> prompt -> patch -> re-verify closed loop."""
    initial_static = analyze_code(code)
    initial_runtime = (
        capture_code(code, timeout_sec=timeout_sec) if run_runtime else None
    )

    if _is_verified_clean(initial_static, initial_runtime, fix_warnings=fix_warnings):
        return RepairResult(
            status="clean",
            original_code=code,
            patched_code=code,
            attempts=0,
            initial_static_issues=initial_static,
            initial_runtime=initial_runtime,
            final_static_issues=initial_static,
            final_runtime=initial_runtime,
        )

    previous_feedback: str | None = None
    last_prompt = ""
    last_output = ""
    last_error: str | None = None
    final_static = initial_static
    final_runtime = initial_runtime

    for attempt in range(1, max_attempts + 1):
        prompt = build_repair_prompt(
            source=code,
            static_issues=initial_static,
            runtime_report=initial_runtime,
            previous_feedback=previous_feedback,
        )
        last_prompt = prompt

        try:
            llm_output = llm_fn(prompt)
        except Exception as exc: # noqa: BLE001
            return RepairResult(
                status="failed",
                original_code=code,
                attempts=attempt,
                initial_static_issues=initial_static,
                initial_runtime=initial_runtime,
                final_static_issues=final_static,
                final_runtime=final_runtime,
                prompt_used=last_prompt,
                error=f"LLM backend error: {exc}",
            )

        last_output = llm_output
        patch_res = apply_patch(code, llm_output, filename=filename)
        if not patch_res.success or patch_res.patched_code is None:
            last_error = patch_res.error
            previous_feedback = f"Patch failed to apply: {patch_res.error}"
            continue

        candidate = patch_res.patched_code
        final_static = analyze_code(candidate)
        final_runtime = (
            capture_code(candidate, timeout_sec=timeout_sec) if run_runtime else None
        )

        if _is_verified_clean(final_static, final_runtime, fix_warnings=fix_warnings):
            return RepairResult(
                status="fixed",
                original_code=code,
                patched_code=candidate,
                diff=patch_res.diff,
                attempts=attempt,
                initial_static_issues=initial_static,
                initial_runtime=initial_runtime,
                final_static_issues=final_static,
                final_runtime=final_runtime,
                prompt_used=last_prompt,
                raw_llm_output=last_output,
            )

        feedback_parts = []
        if final_static:
            feedback_parts.append(format_static_issues(final_static))
        if final_runtime and not final_runtime.success:
            feedback_parts.append(final_runtime.to_prompt_block())
        previous_feedback = "\n".join(feedback_parts)
        last_error = "Patched script still failed verification checks."

    return RepairResult(
        status="failed",
        original_code=code,
        attempts=max_attempts,
        initial_static_issues=initial_static,
        initial_runtime=initial_runtime,
        final_static_issues=final_static,
        final_runtime=final_runtime,
        prompt_used=last_prompt,
        raw_llm_output=last_output,
        error=last_error,
    )


def repair_file(
    script_path: str | Path,
    llm_fn: Callable[[str], str],
    apply: bool = False,
    run_runtime: bool = True,
    fix_warnings: bool = False,
    max_attempts: int = 2,
    timeout_sec: float = 10.0,
) -> RepairResult:
    """Repairs a script on disk; writes back only if verified and apply=True."""
    path = Path(script_path)
    original_code = path.read_text(encoding="utf-8")
    result = repair_code(
        code=original_code,
        llm_fn=llm_fn,
        filename=path.name,
        run_runtime=run_runtime,
        fix_warnings=fix_warnings,
        max_attempts=max_attempts,
        timeout_sec=timeout_sec,
    )
    if result.status == "fixed" and apply and result.patched_code is not None:
        path.write_text(result.patched_code, encoding="utf-8")
    return result