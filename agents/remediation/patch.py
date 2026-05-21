# agents/remediation/patch.py
import difflib
from typing import Tuple


def detect_base_indent(code: str) -> str:
    for line in code.split("\n"):
        if line.strip():
            return line[: len(line) - len(line.lstrip())]
    return ""


def normalize_indent(code: str) -> str:
    base = detect_base_indent(code)
    if not base:
        return code
    result = []
    for line in code.split("\n"):
        if line.startswith(base):
            result.append(line[len(base):])
        elif not line.strip():
            result.append("")
        else:
            result.append(line)
    return "\n".join(result)


def reapply_indent(code: str, base_indent: str) -> str:
    if not base_indent:
        return code
    result = []
    for line in code.split("\n"):
        if line.strip():
            result.append(base_indent + line)
        else:
            result.append(line)
    return "\n".join(result)


def apply_patch(file_content: str, line_start: int, line_end: int,
                fixed_block: str) -> str:
    """Replace lines [line_start, line_end] (1-indexed, inclusive) with fixed_block.

    Preserves the indentation style of the original block.
    """
    original_lines = file_content.split("\n")
    original_block = "\n".join(original_lines[line_start - 1: line_end])
    base_indent = detect_base_indent(original_block)

    normalized = normalize_indent(fixed_block)
    reindented = reapply_indent(normalized, base_indent)

    new_lines = (
        original_lines[: line_start - 1]
        + reindented.split("\n")
        + original_lines[line_end:]
    )
    return "\n".join(new_lines)


def create_unified_diff(original_content: str, new_content: str,
                        file_path: str, line_start: int, line_end: int) -> str:
    diff = difflib.unified_diff(
        original_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
    )
    return "".join(diff)
