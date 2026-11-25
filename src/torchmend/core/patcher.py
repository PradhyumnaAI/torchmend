from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass
from pathlib import Path

PATCH_PATTERN = re.compile(
    r"^[ \t]*<<<<<<< SEARCH[ \t]*\n(.*?)\n"
    r"[ \t]*=======[ \t]*\n(.*?)(?:\n)?[ \t]*>>>>>>> REPLACE[ \t]*$",
    re.DOTALL | re.MULTILINE,
)


@dataclass(frozen=True)
class PatchBlock:
    search: str
    replace: str


@dataclass
class PatchResult:
    success: bool
    patched_code: str | None = None
    diff: str = ""
    applied_blocks: int = 0
    error: str | None = None


def parse_patch_blocks(llm_output: str) -> list[PatchBlock]:
    """Extracts all strict SEARCH/REPLACE blocks from raw LLM text."""
    normalized = llm_output.replace("\r\n", "\n").replace("\r", "\n")
    blocks: list[PatchBlock] = []
    for match in PATCH_PATTERN.finditer(normalized):
        blocks.append(PatchBlock(search=match.group(1), replace=match.group(2)))
    return blocks


def make_unified_diff(
    original_code: str,
    patched_code: str,
    filename: str = "script.py",
) -> str:
    """Generates a standard unified diff between original and patched source."""
    orig_lines = original_code.splitlines(keepends=True)
    new_lines = patched_code.splitlines(keepends=True)
    diff_iter = difflib.unified_diff(
        orig_lines,
        new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
    )
    return "".join(diff_iter)


def apply_patch(
    original_code: str,
    llm_output: str,
    filename: str = "script.py",
    verify_syntax: bool = True,
) -> PatchResult:
    """Applies SEARCH/REPLACE blocks safely and validates resulting Python syntax."""
    blocks = parse_patch_blocks(llm_output)
    if not blocks:
        return PatchResult(
            success=False,
            error="No valid SEARCH/REPLACE patch blocks found in output.",
        )

    normalized_orig = original_code.replace("\r\n", "\n").replace("\r", "\n")
    current_code = normalized_orig

    for idx, block in enumerate(blocks, start=1):
        if not block.search.strip():
            return PatchResult(
                success=False,
                error=(
                    f"Patch block #{idx} has an empty or whitespace-only "
                    "SEARCH section."
                ),
            )

        count = current_code.count(block.search)
        if count == 0:
            return PatchResult(
                success=False,
                error=f"Patch block #{idx} SEARCH text was not found verbatim in source.",
            )
        if count > 1:
            return PatchResult(
                success=False,
                error=(
                    f"Patch block #{idx} SEARCH text matched {count} locations "
                    "(ambiguous); include more surrounding context lines."
                ),
            )

        if block.replace == "" and f"{block.search}\n" in current_code:
            current_code = current_code.replace(f"{block.search}\n", "", 1)
        else:
            current_code = current_code.replace(block.search, block.replace, 1)

    if verify_syntax:
        try:
            ast.parse(current_code)
        except SyntaxError as exc:
            return PatchResult(
                success=False,
                error=f"Patched code failed syntax check (line {exc.lineno}): {exc.msg}",
            )

    diff_text = make_unified_diff(normalized_orig, current_code, filename=filename)
    return PatchResult(
        success=True,
        patched_code=current_code,
        diff=diff_text,
        applied_blocks=len(blocks),
        error=None,
    )


def apply_patch_to_file(
    file_path: str | Path,
    llm_output: str,
    dry_run: bool = True,
    verify_syntax: bool = True,
) -> PatchResult:
    """Applies SEARCH/REPLACE edits to a file (preview-only when dry_run=True)."""
    path = Path(file_path)
    original_code = path.read_text(encoding="utf-8")
    result = apply_patch(
        original_code,
        llm_output,
        filename=path.name,
        verify_syntax=verify_syntax,
    )
    if result.success and not dry_run and result.patched_code is not None:
        path.write_text(result.patched_code, encoding="utf-8")
    return result