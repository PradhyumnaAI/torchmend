from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HARNESS_SCRIPT = r'''
import json
import math
import os
import runpy
import sys
import traceback


def _summarize_value(val, max_str_len=120):
    t = type(val)
    mod = getattr(t, "__module__", "")
    name = t.__name__

    # PyTorch Tensor / Parameter detection (works without forcing global torch import)
    if (
        mod.startswith("torch")
        and hasattr(val, "shape")
        and hasattr(val, "dtype")
        and hasattr(val, "device")
    ):
        info = {
            "kind": "tensor",
            "type": f"{mod}.{name}",
            "shape": list(val.shape),
            "dtype": str(val.dtype),
            "device": str(val.device),
            "requires_grad": bool(getattr(val, "requires_grad", False)),
        }
        try:
            numel_fn = getattr(val, "numel", None)
            is_float_fn = getattr(val, "is_floating_point", None)
            is_complex_fn = getattr(val, "is_complex", None)
            if callable(numel_fn) and numel_fn() > 0 and (
                (callable(is_float_fn) and is_float_fn())
                or (callable(is_complex_fn) and is_complex_fn())
            ):
                import torch

                with torch.no_grad():
                    info["has_nan"] = bool(torch.isnan(val).any().item())
                    info["has_inf"] = bool(torch.isinf(val).any().item())
        except Exception:
            pass
        return info

    # PyTorch nn.Module detection (captures layer shapes like Linear(128, 10))
    if hasattr(val, "_modules") and hasattr(val, "named_children"):
        try:
            children = {}
            for k, child in list(val.named_children())[:15]:
                s = repr(child).replace("\n", " ")
                children[k] = s[:max_str_len]
            return {
                "kind": "nn_module",
                "type": name,
                "children": children,
            }
        except Exception:
            pass

    # NumPy ndarray detection
    if mod.startswith("numpy") and name == "ndarray":
        return {
            "kind": "ndarray",
            "shape": list(val.shape),
            "dtype": str(val.dtype),
        }

    # Python primitives & container lengths
    if isinstance(val, bool) or val is None or isinstance(val, int):
        return {"kind": "primitive", "type": name, "value": val}
    if isinstance(val, float):
        safe_val = str(val) if (math.isnan(val) or math.isinf(val)) else val
        return {"kind": "primitive", "type": "float", "value": safe_val}
    if isinstance(val, str):
        preview = val if len(val) <= max_str_len else val[:max_str_len] + "..."
        return {"kind": "primitive", "type": "str", "value": preview}
    if isinstance(val, (list, tuple, set, dict)):
        return {"kind": "container", "type": name, "len": len(val)}

    return None


def _find_user_frame(tb, target_path):
    target_norm = os.path.normcase(os.path.abspath(target_path))
    last_user_tb = None
    cur = tb
    while cur is not None:
        frame_file = os.path.normcase(os.path.abspath(cur.tb_frame.f_code.co_filename))
        if frame_file == target_norm:
            last_user_tb = cur
        cur = cur.tb_next
    return last_user_tb


def main():
    target_script = sys.argv[1]
    report_path = sys.argv[2]
    sys.argv = [target_script] + sys.argv[3:]

    target_dir = os.path.dirname(os.path.abspath(target_script))
    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)

    report = {
        "crashed": False,
        "exception_type": None,
        "exception_message": None,
        "crash_line": None,
        "crash_function": None,
        "traceback": "",
        "locals_snapshot": {},
    }

    try:
        runpy.run_path(target_script, run_name="__main__")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, allow_nan=False)
        sys.exit(code)
    except BaseException as exc:
        tb = exc.__traceback__
        user_tb = _find_user_frame(tb, target_script)

        report["crashed"] = True
        report["exception_type"] = type(exc).__name__
        report["exception_message"] = str(exc)
        report["traceback"] = "".join(traceback.format_exception(type(exc), exc, tb))

        if isinstance(exc, SyntaxError):
            report["crash_line"] = exc.lineno
            report["crash_function"] = "<module>"
        elif user_tb is not None:
            report["crash_line"] = user_tb.tb_lineno
            report["crash_function"] = user_tb.tb_frame.f_code.co_name
            snapshot = {}
            for var_name, var_val in list(user_tb.tb_frame.f_locals.items())[:40]:
                if var_name.startswith("__"):
                    continue
                summary = _summarize_value(var_val)
                if summary is not None:
                    snapshot[var_name] = summary
            report["locals_snapshot"] = snapshot

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, allow_nan=False)
        sys.exit(1)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, allow_nan=False)


if __name__ == "__main__":
    main()
'''


@dataclass
class RuntimeReport:
    success: bool
    timed_out: bool
    returncode: int
    stdout: str
    stderr: str
    exception_type: str | None = None
    exception_message: str | None = None
    crash_line: int | None = None
    crash_function: str | None = None
    traceback: str = ""
    locals_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_prompt_block(self) -> str:
        """Formats crash details and live tensor metadata for SLM prompt injection."""
        if self.success:
            return "Execution succeeded with zero errors."

        lines = [
            f"Exception: {self.exception_type}: {self.exception_message}",
            f"Location: function `{self.crash_function}` at line {self.crash_line}",
        ]
        if self.locals_snapshot:
            lines.append("Live Runtime State at Crash Frame:")
            for name, info in self.locals_snapshot.items():
                kind = info.get("kind")
                if kind == "tensor":
                    extras = []
                    if info.get("has_nan"):
                        extras.append("HAS_NAN")
                    if info.get("has_inf"):
                        extras.append("HAS_INF")
                    flag_str = f" [{' '.join(extras)}]" if extras else ""
                    lines.append(
                        f"  - {name}: Tensor(shape={info['shape']}, dtype={info['dtype']}, "
                        f"device={info['device']}, requires_grad={info['requires_grad']}){flag_str}"
                    )
                elif kind == "nn_module":
                    lines.append(f"  - {name}: {info['type']} layers={info['children']}")
                elif kind == "ndarray":
                    lines.append(
                        f"  - {name}: ndarray(shape={info['shape']}, dtype={info['dtype']})"
                    )
                elif kind == "container":
                    lines.append(f"  - {name}: {info['type']}(len={info['len']})")
                else:
                    lines.append(f"  - {name}: {info['type']} = {info.get('value')!r}")
        return "\n".join(lines)


def _run_with_harness(
    target_path: Path,
    timeout_sec: float = 10.0,
    script_args: list[str] | None = None,
    cwd: Path | None = None,
) -> RuntimeReport:
    with tempfile.TemporaryDirectory(prefix="torchmend_run_") as tmpdir:
        tmp = Path(tmpdir)
        harness_path = tmp / "_harness.py"
        report_path = tmp / "report.json"
        harness_path.write_text(HARNESS_SCRIPT, encoding="utf-8")

        cmd = [sys.executable, str(harness_path), str(target_path), str(report_path)]
        if script_args:
            cmd.extend(script_args)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=str(cwd) if cwd else None,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return RuntimeReport(
                success=False,
                timed_out=True,
                returncode=-1,
                stdout=exc.stdout if isinstance(exc.stdout, str) else "",
                stderr=exc.stderr if isinstance(exc.stderr, str) else "",
                exception_type="TimeoutError",
                exception_message=f"Execution exceeded {timeout_sec}s timeout.",
            )

        data: dict[str, Any] = {}
        if report_path.exists():
            data = json.loads(report_path.read_text(encoding="utf-8"))

        crashed = bool(data.get("crashed", proc.returncode != 0))
        return RuntimeReport(
            success=(proc.returncode == 0 and not crashed),
            timed_out=False,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            exception_type=data.get("exception_type"),
            exception_message=data.get("exception_message"),
            crash_line=data.get("crash_line"),
            crash_function=data.get("crash_function"),
            traceback=data.get("traceback", ""),
            locals_snapshot=data.get("locals_snapshot", {}),
        )


def capture_code(code: str, timeout_sec: float = 10.0) -> RuntimeReport:
    """Executes raw Python source code in an isolated subprocess and captures crash state."""
    with tempfile.TemporaryDirectory(prefix="torchmend_src_") as tmpdir:
        target_path = Path(tmpdir) / "user_script.py"
        target_path.write_text(code, encoding="utf-8")
        return _run_with_harness(target_path, timeout_sec=timeout_sec)


def capture_file(
    script_path: str | Path,
    timeout_sec: float = 10.0,
    script_args: list[str] | None = None,
) -> RuntimeReport:
    """Executes an existing Python file on disk and captures crash state."""
    resolved = Path(script_path).resolve()
    return _run_with_harness(
        resolved,
        timeout_sec=timeout_sec,
        script_args=script_args,
        cwd=resolved.parent,
    )