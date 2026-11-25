from torchmend.core.ast_slicer import slice_code


def test_short_script_is_not_sliced():
    code = "import torch\nx = 1\nprint(x)\n"
    result = slice_code(code, target_lines=2)
    assert result.was_sliced is False
    assert result.syntax_fallback is False
    assert result.sliced_code == code


def test_large_class_keeps_imports_init_and_crashing_method():
    filler_methods = "\n".join(
        [f"    def helper_{i}(self):\n        return {i}\n" for i in range(25)]
    )
    code = (
        "import math\n"
        "import torch\n\n"
        "def unrelated_util():\n"
        "    return 42\n\n"
        "class BigNet:\n"
        "    def __init__(self):\n"
        "        self.fc = 'Linear(128, 10)'\n\n"
        f"{filler_methods}\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
    )
    crash_line = next(
        idx
        for idx, line in enumerate(code.splitlines(), start=1)
        if "return self.fc(x)" in line
    )

    result = slice_code(
        code,
        target_lines=crash_line,
        max_unsliced_lines=20,
        max_class_lines=30,
    )
    assert result.was_sliced is True
    assert "import torch" in result.sliced_code
    assert "def __init__(self):" in result.sliced_code
    assert "def forward(self, x):" in result.sliced_code
    assert "def helper_10(self):" not in result.sliced_code
    assert "unrelated_util" not in result.sliced_code
    assert "```python" in result.to_prompt_block()


def test_multiple_target_lines_include_both_functions():
    filler = "\n".join([f"def pad_{i}():\n    return {i}\n" for i in range(20)])
    code = (
        "import os\n\n"
        "def first_bug():\n"
        "    return 1 / 0\n\n"
        f"{filler}\n"
        "def second_bug():\n"
        "    return None.attr\n"
    )
    lines = code.splitlines()
    line_a = next(i for i, text in enumerate(lines, 1) if "1 / 0" in text)
    line_b = next(i for i, text in enumerate(lines, 1) if "None.attr" in text)

    result = slice_code(code, target_lines=[line_a, line_b], max_unsliced_lines=20)
    assert result.was_sliced is True
    assert "import os" in result.sliced_code
    assert "def first_bug():" in result.sliced_code
    assert "def second_bug():" in result.sliced_code
    assert "def pad_10():" not in result.sliced_code


def test_syntax_error_falls_back_to_line_window():
    code = "\n".join([f"x_{i} = {i}" for i in range(60)] + ["def broken(:", "    pass"])
    result = slice_code(code, target_lines=61, max_unsliced_lines=20, context_window=4)
    assert result.was_sliced is True
    assert result.syntax_fallback is True
    assert "def broken(:" in result.sliced_code
    assert "x_0 = 0" not in result.sliced_code