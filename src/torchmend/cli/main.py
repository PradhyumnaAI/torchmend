from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torchmend.core.pipeline import (
    format_static_issues,
    make_openai_compatible_backend,
    repair_file,
)
from torchmend.core.runtime_capture import capture_file
from torchmend.core.static_rules import analyze_code


def _cmd_check(args: argparse.Namespace) -> int:
    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 2

    code = path.read_text(encoding="utf-8")
    issues = analyze_code(code)
    print(format_static_issues(issues))

    has_failure = any(issue.severity == "error" for issue in issues)

    if args.run:
        print()
        report = capture_file(path, timeout_sec=args.timeout)
        print(report.to_prompt_block())
        if not report.success:
            has_failure = True

    return 1 if has_failure else 0


def _cmd_fix(args: argparse.Namespace) -> int:
    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 2

    llm_fn = make_openai_compatible_backend(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    result = repair_file(
        script_path=path,
        llm_fn=llm_fn,
        apply=args.apply,
        run_runtime=not args.no_run,
        fix_warnings=args.fix_warnings,
        max_attempts=args.max_attempts,
        timeout_sec=args.timeout,
    )

    if result.status == "clean":
        print(f"[CLEAN] No blocking issues found in {path.name}.")
        return 0

    if result.status == "fixed":
        print(f"[FIXED] Verified in {result.attempts} attempt(s):\n")
        print(result.diff)
        if args.apply:
            print(f"Applied verified patch to {path}.")
        else:
            print("Dry run only. Re-run with --apply to write changes to disk.")
        return 0

    print(f"[FAILED] Could not verify a fix after {result.attempts} attempt(s).")
    if result.error:
        print(f"Reason: {result.error}", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torchmend",
        description="Runtime-augmented debugger and verified patcher for PyTorch.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check_p = sub.add_parser("check", help="Run static analysis and runtime capture.")
    check_p.add_argument("file", help="Path to Python/PyTorch script.")
    check_p.add_argument(
        "--run",
        action="store_true",
        help="Also execute script in isolated harness to capture crashes.",
    )
    check_p.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Runtime capture timeout in seconds (default: 10).",
    )
    check_p.set_defaults(func=_cmd_check)

    fix_p = sub.add_parser("fix", help="Diagnose, patch, and verify a script via LLM.")
    fix_p.add_argument("file", help="Path to Python/PyTorch script.")
    fix_p.add_argument(
        "--model",
        default="qwen2.5-coder:1.5b",
        help="Model name (default: qwen2.5-coder:1.5b for local Ollama).",
    )
    fix_p.add_argument(
        "--base-url",
        default="http://localhost:11434/v1",
        help="OpenAI-compatible API base URL (Ollama, Groq, vLLM, etc.).",
    )
    fix_p.add_argument(
        "--api-key",
        default=None,
        help="API key (or set TORCHMEND_API_KEY env var).",
    )
    fix_p.add_argument(
        "--apply",
        action="store_true",
        help="Write the verified patch directly to disk (default: preview diff only).",
    )
    fix_p.add_argument(
        "--no-run",
        action="store_true",
        help="Skip runtime execution and check static rules only.",
    )
    fix_p.add_argument(
        "--fix-warnings",
        action="store_true",
        help="Also require static warnings (not just errors) to be resolved.",
    )
    fix_p.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Maximum patch-and-verify retries (default: 2).",
    )
    fix_p.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Runtime capture timeout in seconds (default: 10).",
    )
    fix_p.set_defaults(func=_cmd_fix)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())