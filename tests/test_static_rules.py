from torchmend.core.static_rules import analyze_code, analyze_pytorch_ast


def test_pt001_double_softmax_in_forward():
    code = """
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def forward(self, x):
        return F.softmax(x, dim=-1)

criterion = nn.CrossEntropyLoss()
"""
    rule_ids = [i.rule_id for i in analyze_pytorch_ast(code)]
    assert "PT001" in rule_ids


def test_pt001_allows_softmax_in_predict():
    code = """
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def forward(self, x):
        return x

criterion = nn.CrossEntropyLoss()

def predict(model, x):
    return F.softmax(model(x), dim=-1)
"""
    assert analyze_pytorch_ast(code) == []


def test_pt002_and_pt005_nested_training_loop_no_duplicates():
    code = """
total_loss = 0.0
for epoch in range(5):
    for x, y in loader:
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss
"""
    rule_ids = [i.rule_id for i in analyze_pytorch_ast(code)]
    assert rule_ids == ["PT002", "PT005"]


def test_clean_training_loop_has_zero_issues():
    code = """
total_loss = 0.0
for epoch in range(5):
    for x, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
"""
    assert analyze_pytorch_ast(code) == []


def test_pt003_eval_without_no_grad():
    code = """
def evaluate(model, loader):
    model.eval()
    for x, y in loader:
        _ = model(x)
"""
    assert [i.rule_id for i in analyze_pytorch_ast(code)] == ["PT003"]


def test_pt003_clean_with_decorator():
    code = """
@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    for x, y in loader:
        _ = model(x)
"""
    assert analyze_pytorch_ast(code) == []


def test_pt004_inplace_in_forward():
    code = """
class Net(nn.Module):
    def forward(self, x):
        x.relu_()
        return x
"""
    assert [i.rule_id for i in analyze_pytorch_ast(code)] == ["PT004"]


def test_syntax_error_handled_gracefully():
    code = "def broken(:\n    pass\n"
    issues = analyze_code(code)
    assert len(issues) == 1
    assert issues[0].rule_id == "PY000"


def test_ruff_catches_undefined_name():
    code = "x = undefined_var + 1\n"
    rule_ids = [i.rule_id for i in analyze_code(code, include_ruff=True)]
    assert "F821" in rule_ids