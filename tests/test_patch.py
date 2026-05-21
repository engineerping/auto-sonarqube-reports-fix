# tests/test_patch.py
import pytest
from agents.remediation.patch import (
    detect_base_indent,
    normalize_indent,
    reapply_indent,
    apply_patch,
    create_unified_diff,
)


def test_detect_base_indent_4spaces():
    code = "    public void method() {\n        String s = \"hello\";\n    }"
    assert detect_base_indent(code) == "    "


def test_detect_base_indent_8spaces():
    code = "        String status = \"PENDING\";"
    assert detect_base_indent(code) == "        "


def test_detect_base_indent_tabs():
    code = "\t\tString s = \"hello\";"
    assert detect_base_indent(code) == "\t\t"


def test_detect_base_indent_no_indent():
    code = "public void method() {}"
    assert detect_base_indent(code) == ""


def test_normalize_removes_base_indent():
    code = "    line1\n    line2\n    line3"
    result = normalize_indent(code)
    assert result == "line1\nline2\nline3"


def test_normalize_preserves_relative_indent():
    code = "    public void m() {\n        return 1;\n    }"
    result = normalize_indent(code)
    assert result == "public void m() {\n    return 1;\n}"


def test_reapply_adds_indent():
    code = "public void m() {\n    return 1;\n}"
    result = reapply_indent(code, "    ")
    assert result == "    public void m() {\n        return 1;\n    }"


def test_reapply_with_tabs():
    code = "public void m() {\n    return 1;\n}"
    result = reapply_indent(code, "\t")
    assert result.startswith("\t")
    assert "\t    return 1;" in result


def test_apply_patch_replaces_lines():
    file_content = (
        "public class Foo {\n"
        "    public void bar() {\n"
        "        String x = \"dup\";\n"
        "        String y = \"dup\";\n"
        "    }\n"
        "}"
    )
    fixed_block = 'public void bar() {\n    private static final String DUP = "dup";\n    String x = DUP;\n    String y = DUP;\n}'
    result = apply_patch(file_content, line_start=2, line_end=5, fixed_block=fixed_block)
    assert "DUP" in result
    assert "    private static final" in result  # indentation preserved
    assert "public class Foo {" in result         # surrounding code intact


def test_create_unified_diff_produces_diff():
    original = "public class Foo {\n    String x = \"dup\";\n}"
    fixed_block = "public class Foo {\n    private static final String DUP = \"dup\";\n    String x = DUP;\n}"
    diff = create_unified_diff(original, fixed_block, "Foo.java", line_start=1, line_end=3)
    assert "---" in diff
    assert "+++" in diff
    assert "-    String x" in diff
