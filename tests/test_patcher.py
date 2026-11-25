from pathlib import Path

from torchmend.core.patcher import apply_patch, apply_patch_to_file, parse_patch_blocks


def test_apply_single_patch_generates_diff():
    original = (
        "class Net:\n"
        "    def __init__(self):\n"
        "        self.fc = Linear(128, 10)\n\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
    )
    llm_output = (
        "Fix the input dimension mismatch:\n"
        "```\n"
        "<<<<<<< SEARCH\n"
        "        self.fc = Linear(128, 10)\n"
        "=======\n"
        "        self.fc = Linear(64, 10)\n"
        ">>>>>>> REPLACE\n"
        "```\n"
    )
    result = apply_patch(original, llm_output, filename="model.py")
    assert result.success is True
    assert result.applied_blocks == 1
    assert result.patched_code is not None
    assert "Linear(64, 10)" in result.patched_code
    assert "-        self.fc = Linear(128, 10)" in result.diff
    assert "+        self.fc = Linear(64, 10)" in result.diff


def test_apply_pure_deletion_removes_line_cleanly():
    original = (
        "def forward(self, x):\n"
        "    x = self.fc(x)\n"
        "    x = F.softmax(x, dim=1)\n"
        "    return x\n"
    )
    llm_output = (
        "<<<<<<< SEARCH\n"
        "    x = F.softmax(x, dim=1)\n"
        "=======\n"
        ">>>>>>> REPLACE\n"
    )
    result = apply_patch(original, llm_output)
    assert result.success is True
    assert result.patched_code == (
        "def forward(self, x):\n"
        "    x = self.fc(x)\n"
        "    return x\n"
    )


def test_apply_multiple_blocks_in_one_response():
    original = "a = 1\nb = 2\nc = a + b\n"
    llm_output = (
        "<<<<<<< SEARCH\n"
        "a = 1\n"
        "=======\n"
        "a = 10\n"
        ">>>>>>> REPLACE\n"
        "\n"
        "<<<<<<< SEARCH\n"
        "b = 2\n"
        "=======\n"
        "b = 20\n"
        ">>>>>>> REPLACE\n"
    )
    blocks = parse_patch_blocks(llm_output)
    assert len(blocks) == 2
    result = apply_patch(original, llm_output)
    assert result.success is True
    assert result.applied_blocks == 2
    assert result.patched_code == "a = 10\nb = 20\nc = a + b\n"


def test_rejects_ambiguous_search_block():
    original = "x = 1\ny = 2\nx = 1\n"
    llm_output = "<<<<<<< SEARCH\nx = 1\n=======\nx = 9\n>>>>>>> REPLACE"
    result = apply_patch(original, llm_output)
    assert result.success is False
    assert result.error is not None
    assert "ambiguous" in result.error


def test_rejects_patch_that_breaks_python_syntax():
    original = "def add(a, b):\n    return a + b\n"
    llm_output = (
        "<<<<<<< SEARCH\n"
        "    return a + b\n"
        "=======\n"
        "    return a + (b\n"
        ">>>>>>> REPLACE"
    )
    result = apply_patch(original, llm_output)
    assert result.success is False
    assert result.error is not None
    assert "syntax check" in result.error


def test_apply_patch_to_file_respects_dry_run(tmp_path: Path):
    script = tmp_path / "train.py"
    script.write_text("lr = 0.1\n", encoding="utf-8")
    llm_output = "<<<<<<< SEARCH\nlr = 0.1\n=======\nlr = 0.001\n>>>>>>> REPLACE"

    dry_res = apply_patch_to_file(script, llm_output, dry_run=True)
    assert dry_res.success is True
    assert script.read_text(encoding="utf-8") == "lr = 0.1\n"

    write_res = apply_patch_to_file(script, llm_output, dry_run=False)
    assert write_res.success is True
    assert script.read_text(encoding="utf-8") == "lr = 0.001\n"