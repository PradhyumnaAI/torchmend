from __future__ import annotations

import ast
from dataclasses import dataclass, field


@dataclass
class CodeSlice:
    sliced_code: str
    included_ranges: list[tuple[int, int]] = field(default_factory=list)
    was_sliced: bool = False
    total_lines: int = 0
    syntax_fallback: bool = False

    def to_prompt_block(self) -> str:
        """Formats the sliced source cleanly for LLM prompt injection."""
        shown = len(self.sliced_code.splitlines())
        header = f"Source Context ({shown} of {self.total_lines} lines):"
        return f"{header}\n```python\n{self.sliced_code}\n```"


def _merge_ranges(
    ranges: list[tuple[int, int]],
    total_lines: int,
) -> list[tuple[int, int]]:
    clamped = [
        (max(1, start), min(total_lines, end))
        for start, end in ranges
        if start <= end and end >= 1 and start <= total_lines
    ]
    if not clamped:
        return []
    clamped.sort()
    merged = [clamped[0]]
    for start, end in clamped[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _node_span(node: ast.AST) -> tuple[int, int] | None:
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)
    if start is None or end is None:
        return None
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        dec_starts = [getattr(d, "lineno", start) for d in decorators]
        start = min([start, *dec_starts])
    return (start, end)


def _render_ranges(lines: list[str], ranges: list[tuple[int, int]]) -> str:
    chunks: list[str] = []
    prev_end = 0
    for start, end in ranges:
        if prev_end > 0 and start > prev_end + 1:
            chunks.append(f"# ... (lines {prev_end + 1}-{start - 1} omitted) ...")
        elif prev_end == 0 and start > 1:
            chunks.append(f"# ... (lines 1-{start - 1} omitted) ...")
        chunks.append("\n".join(lines[start - 1 : end]))
        prev_end = end

    if prev_end < len(lines):
        chunks.append(f"# ... (lines {prev_end + 1}-{len(lines)} omitted) ...")
    return "\n".join(chunks)


def slice_code(
    source: str,
    target_lines: int | list[int] | None = None,
    max_unsliced_lines: int = 80,
    context_window: int = 12,
    max_class_lines: int = 60,
) -> CodeSlice:
    """Slices a large script to imports and enclosing scopes around target lines."""
    normalized = source.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    total_lines = len(lines)

    if total_lines <= max_unsliced_lines or target_lines is None:
        return CodeSlice(
            sliced_code=normalized,
            included_ranges=[(1, total_lines)] if total_lines > 0 else [],
            was_sliced=False,
            total_lines=total_lines,
            syntax_fallback=False,
        )

    targets = [target_lines] if isinstance(target_lines, int) else list(target_lines)
    targets = [t for t in targets if 1 <= t <= total_lines]
    if not targets:
        return CodeSlice(
            sliced_code=normalized,
            included_ranges=[(1, total_lines)],
            was_sliced=False,
            total_lines=total_lines,
            syntax_fallback=False,
        )

    try:
        tree = ast.parse(normalized)
    except SyntaxError:
        raw_ranges = [(t - context_window, t + context_window) for t in targets]
        merged = _merge_ranges(raw_ranges, total_lines)
        return CodeSlice(
            sliced_code=_render_ranges(lines, merged),
            included_ranges=merged,
            was_sliced=True,
            total_lines=total_lines,
            syntax_fallback=True,
        )

    selected_ranges: list[tuple[int, int]] = []

    # 1. Preserve all top-level imports so the model knows available modules/aliases
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            span = _node_span(node)
            if span:
                selected_ranges.append(span)

    # 2. Extract enclosing class/function scope for each target line
    for t in targets:
        matched_top = False
        for node in tree.body:
            span = _node_span(node)
            if span is None or not (span[0] <= t <= span[1]):
                continue
            matched_top = True

            if isinstance(node, ast.ClassDef):
                class_len = span[1] - span[0] + 1
                if class_len <= max_class_lines:
                    selected_ranges.append(span)
                else:
                    selected_ranges.append((span[0], node.lineno))
                    hit_member = False
                    for item in node.body:
                        item_span = _node_span(item)
                        if item_span is None:
                            continue
                        is_init = (
                            isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and item.name == "__init__"
                        )
                        contains_target = item_span[0] <= t <= item_span[1]
                        if is_init or contains_target:
                            selected_ranges.append(item_span)
                            if contains_target:
                                hit_member = True
                    if not hit_member:
                        selected_ranges.append((t - context_window, t + context_window))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                selected_ranges.append(span)
            else:
                selected_ranges.append((t - context_window, t + context_window))
            break

        if not matched_top:
            selected_ranges.append((t - context_window, t + context_window))

    merged = _merge_ranges(selected_ranges, total_lines)
    was_sliced = not (len(merged) == 1 and merged[0] == (1, total_lines))
    sliced_text = _render_ranges(lines, merged) if was_sliced else normalized

    return CodeSlice(
        sliced_code=sliced_text,
        included_ranges=merged,
        was_sliced=was_sliced,
        total_lines=total_lines,
        syntax_fallback=False,
    )