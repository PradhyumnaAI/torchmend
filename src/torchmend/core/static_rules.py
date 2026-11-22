from __future__ import annotations

import ast
import json
import subprocess
from dataclasses import dataclass

INPLACE_OPS = {
    "relu_",
    "sigmoid_",
    "tanh_",
    "add_",
    "sub_",
    "mul_",
    "div_",
    "clamp_",
    "zero_",
}


@dataclass(frozen=True)
class StaticIssue:
    rule_id: str
    line: int
    severity: str  # "error" | "warning"
    message: str
    suggestion: str
    source: str  # "python_ast" | "pytorch_ast" | "ruff"


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _has_backward(node: ast.AST) -> bool:
    return any(
        isinstance(sub, ast.Call) and _call_name(sub) == "backward"
        for sub in ast.walk(node)
    )


def _has_no_grad_decorator(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in fn.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Attribute) and target.attr in {
            "no_grad",
            "inference_mode",
        }:
            return True
        if isinstance(target, ast.Name) and target.id in {
            "no_grad",
            "inference_mode",
        }:
            return True
    return False


def _has_no_grad_with(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.With):
            for item in sub.items:
                ctx = item.context_expr
                if isinstance(ctx, ast.Call) and _call_name(ctx) in {
                    "no_grad",
                    "inference_mode",
                }:
                    return True
    return False


def analyze_pytorch_ast(code: str) -> list[StaticIssue]:
    """Runs deterministic PyTorch silent-bug checks using Python's AST."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [
            StaticIssue(
                rule_id="PY000",
                line=exc.lineno or 1,
                severity="error",
                message=f"SyntaxError: {exc.msg}",
                suggestion="Fix the Python syntax error before running PyTorch checks.",
                source="python_ast",
            )
        ]

    issues: list[StaticIssue] = []

    uses_ce_loss = any(
        (
            isinstance(n, ast.Attribute)
            and n.attr in {"CrossEntropyLoss", "cross_entropy"}
        )
        or (
            isinstance(n, ast.Name)
            and n.id in {"CrossEntropyLoss", "cross_entropy"}
        )
        for n in ast.walk(tree)
    )

    # PT001: Double softmax with CrossEntropyLoss (scoped to __init__ and forward)
    if uses_ce_loss:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in {
                "forward",
                "__init__",
            }:
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call):
                        name = _call_name(sub)
                        if name in {"softmax", "log_softmax", "Softmax", "LogSoftmax"}:
                            issues.append(
                                StaticIssue(
                                    rule_id="PT001",
                                    line=sub.lineno,
                                    severity="error",
                                    message=f"`{name}` inside `{node.name}()` used alongside CrossEntropyLoss.",
                                    suggestion="Return raw logits from the model; CrossEntropyLoss applies LogSoftmax internally.",
                                    source="pytorch_ast",
                                )
                            )

    # PT002 & PT005: Innermost training loop checks
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and _has_backward(node):
            has_inner_training_loop = any(
                isinstance(sub, ast.For) and sub is not node and _has_backward(sub)
                for sub in ast.walk(node)
            )
            if has_inner_training_loop:
                continue

            calls = [sub for sub in ast.walk(node) if isinstance(sub, ast.Call)]
            call_names = {_call_name(c) for c in calls}

            if "step" in call_names and "zero_grad" not in call_names:
                issues.append(
                    StaticIssue(
                        rule_id="PT002",
                        line=node.lineno,
                        severity="error",
                        message="Training loop calls `.backward()` and `.step()` without `.zero_grad()`.",
                        suggestion="Call `optimizer.zero_grad()` before computing gradients each iteration.",
                        source="pytorch_ast",
                    )
                )

            loss_vars = {
                c.func.value.id
                for c in calls
                if isinstance(c.func, ast.Attribute)
                and c.func.attr == "backward"
                and isinstance(c.func.value, ast.Name)
            }

            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.AugAssign)
                    and isinstance(sub.op, ast.Add)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id in loss_vars
                ):
                    issues.append(
                        StaticIssue(
                            rule_id="PT005",
                            line=sub.lineno,
                            severity="warning",
                            message=f"Accumulating tensor `{sub.value.id}` directly inside loop retains the autograd graph.",
                            suggestion=f"Use `{sub.value.id}.item()` or `{sub.value.id}.detach()` to prevent VRAM leaks.",
                            source="pytorch_ast",
                        )
                    )

    # PT003 & PT004: Function-level checks
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            calls = [sub for sub in ast.walk(node) if isinstance(sub, ast.Call)]

            has_eval = any(
                isinstance(c.func, ast.Attribute) and c.func.attr == "eval"
                for c in calls
            )
            has_loop = any(isinstance(sub, ast.For) for sub in ast.walk(node))
            if (
                has_eval
                and has_loop
                and not _has_no_grad_decorator(node)
                and not _has_no_grad_with(node)
            ):
                issues.append(
                    StaticIssue(
                        rule_id="PT003",
                        line=node.lineno,
                        severity="warning",
                        message=f"Function `{node.name}` runs an `.eval()` loop without `torch.no_grad()` or `torch.inference_mode()`.",
                        suggestion="Wrap evaluation passes in `with torch.no_grad():` or decorate with `@torch.inference_mode()`.",
                        source="pytorch_ast",
                    )
                )

            if node.name == "forward":
                for c in calls:
                    if isinstance(c.func, ast.Attribute) and c.func.attr in INPLACE_OPS:
                        clean_op = c.func.attr.rstrip("_")
                        issues.append(
                            StaticIssue(
                                rule_id="PT004",
                                line=c.lineno,
                                severity="warning",
                                message=f"In-place operation `.{c.func.attr}()` inside `forward()` can break autograd.",
                                suggestion=f"Replace `.{c.func.attr}()` with out-of-place `{clean_op}`.",
                                source="pytorch_ast",
                            )
                        )

    issues.sort(key=lambda item: (item.line, item.rule_id))
    return issues


def run_ruff_check(code: str, filename: str = "snippet.py") -> list[StaticIssue]:
    """Runs Ruff over a raw Python string via stdin and parses JSON diagnostics."""
    try:
        proc = subprocess.run(
            [
                "ruff",
                "check",
                "--output-format=json",
                "--stdin-filename",
                filename,
                "-",
            ],
            input=code,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    if not proc.stdout.strip():
        return []

    try:
        raw_items = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []

    issues: list[StaticIssue] = []
    for item in raw_items:
        code_id = item.get("code") or "RUFF"
        line = item.get("location", {}).get("row", 1)
        msg = item.get("message", "Lint issue detected.")
        fix = item.get("fix")
        suggestion = (
            fix.get("message", "Auto-fixable via ruff.")
            if isinstance(fix, dict)
            else "Review the flagged line."
        )
        issues.append(
            StaticIssue(
                rule_id=code_id,
                line=line,
                severity="error" if code_id == "E999" else "warning",
                message=msg,
                suggestion=suggestion,
                source="ruff",
            )
        )
    return issues


def analyze_code(code: str, include_ruff: bool = True) -> list[StaticIssue]:
    """Entry point combining deterministic PyTorch AST checks and Ruff."""
    issues = analyze_pytorch_ast(code)
    if include_ruff and not any(i.rule_id == "PY000" for i in issues):
        issues.extend(run_ruff_check(code))
    issues.sort(key=lambda item: (item.line, item.rule_id))
    return issues