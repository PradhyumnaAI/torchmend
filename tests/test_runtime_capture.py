from pathlib import Path

from torchmend.core.runtime_capture import capture_code, capture_file


def test_clean_script_succeeds():
    report = capture_code("print('ok')\nx = 40 + 2\n")
    assert report.success is True
    assert report.timed_out is False
    assert report.stdout.strip() == "ok"
    assert report.exception_type is None


def test_python_crash_captures_line_function_and_locals():
    code = """
def divide(a, b):
    items = [1, 2, 3]
    return a / b

divide(10, 0)
"""
    report = capture_code(code)
    assert report.success is False
    assert report.exception_type == "ZeroDivisionError"
    assert report.crash_function == "divide"
    assert report.crash_line == 4
    assert report.locals_snapshot["a"]["value"] == 10
    assert report.locals_snapshot["b"]["value"] == 0
    assert report.locals_snapshot["items"] == {"kind": "container", "type": "list", "len": 3}


def test_tensor_and_nn_module_snapshot_and_prompt_block():
    code = """
class FakeTensor:
    __module__ = "torch"
    def __init__(self, shape, dtype="torch.float32", device="cpu", requires_grad=True):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.requires_grad = requires_grad

class FakeNet:
    def __init__(self):
        self._modules = {"fc": "Linear(in_features=128, out_features=10)"}
    def named_children(self):
        return self._modules.items()
    def forward(self, x):
        raise RuntimeError("mat1 and mat2 shapes cannot be multiplied (32x64 and 128x10)")

net = FakeNet()
x = FakeTensor((32, 64))
net.forward(x)
"""
    report = capture_code(code)
    assert report.success is False
    assert report.exception_type == "RuntimeError"
    assert report.crash_function == "forward"
    assert report.locals_snapshot["x"]["kind"] == "tensor"
    assert report.locals_snapshot["x"]["shape"] == [32, 64]
    assert report.locals_snapshot["self"]["kind"] == "nn_module"
    assert "fc" in report.locals_snapshot["self"]["children"]

    prompt_block = report.to_prompt_block()
    assert "Tensor(shape=[32, 64]" in prompt_block
    assert "FakeNet layers=" in prompt_block


def test_syntax_error_captured_cleanly():
    report = capture_code("def broken(:\n    pass\n")
    assert report.success is False
    assert report.exception_type == "SyntaxError"
    assert report.crash_line == 1


def test_infinite_loop_hits_timeout():
    report = capture_code("import time\ntime.sleep(5)\n", timeout_sec=0.5)
    assert report.success is False
    assert report.timed_out is True
    assert report.exception_type == "TimeoutError"


def test_capture_file_runs_real_path(tmp_path: Path):
    script = tmp_path / "demo.py"
    script.write_text("x = [1, 2]\nprint(x[5])\n", encoding="utf-8")
    report = capture_file(script)
    assert report.success is False
    assert report.exception_type == "IndexError"
    assert report.crash_line == 2
    assert report.locals_snapshot["x"]["len"] == 2