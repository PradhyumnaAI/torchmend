# TorchMend

[![CI](https://github.com/PradhyumnaAI/torchmend/actions/workflows/ci.yml/badge.svg)](https://github.com/PradhyumnaAI/torchmend/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

**TorchMend** is a runtime-augmented debugger and automated patcher for Python and PyTorch.

Standard coding LLMs debug PyTorch scripts blindly—they try to calculate `Conv2d`/`Linear` tensor shapes in their heads and often miss silent training bugs (like missing `zero_grad()` or accumulating graph-attached loss tensors). TorchMend solves this by combining **static PyTorch AST checks**, **live crash-frame tensor telemetry**, **AST context slicing**, and **verified Search-and-Replace patching**—ultimately powering a compact, distilled student coder model trained strictly on execution-verified fixes.

---

## Architecture & Pipeline

```text
User PyTorch Script
        │
        ├──► 1. Static Analyser (static_rules.py)
        │       └── PyTorch AST rules (PT001–PT005) + Ruff lint checks
        │
        └──► 2. Runtime Capture Harness (runtime_capture.py)
                └── Isolated subprocess + deepest user frame + live Tensor/nn.Module snapshot
                        │
                        ▼
             3. AST Context Slicer (ast_slicer.py)          [Planned]
                └── Extracts imports, crashing class/method & local context
                        │
                        ▼
             4. Repair Model (API LLM / Student LoRA)       [Planned]
                └── Generates strict SEARCH / REPLACE edits from grounded state
                        │
                        ▼
             5. Safe Patcher & Verification Loop (patcher.py) [Planned]
                └── Syntax check (ast.parse) + unified diff + re-run verification
```

---

## Current Progress & Roadmap

| Phase / Module | Component | Purpose | Status |
|---|---|---|---|
| **Module 1** | `core/static_rules.py` | Detects silent PyTorch bugs via AST (`PT001`–`PT005`) + integrates `ruff` | Merged (`main`) — 9 tests |
| **Module 2** | `core/runtime_capture.py` | Executes scripts in a timed subprocess and snapshots live `Tensor` shapes, dtypes, devices, NaN/Inf flags, and `nn.Module` specs | Merged (PR #1) — 6 tests |
| **CI Pipeline** | `.github/workflows/ci.yml` | Runs `ruff check` and `pytest` across Python 3.12 & 3.14 on every PR | Merged (PR #2) |
| **Module 3** | `core/ast_slicer.py` + `core/patcher.py` | Slices large files to token-efficient crash contexts; applies & verifies `SEARCH/REPLACE` diffs | Up next |
| **Module 4** | `core/pipeline.py` + `cli/` | End-to-end debug-and-verify loop connecting Modules 1–3 to local/cloud LLMs via CLI | Planned |
| **Phase 2** | Student LLM Distillation | Synthetic PyTorch bug suite → teacher generation with automated rejection sampling (Modules 1–3) → LoRA fine-tuning of a compact coder model → before/after eval | Planned |

---

## Implemented Static PyTorch Rules (`static_rules.py`)

| Rule ID | Severity | What It Catches |
|---|---|---|
| **`PT001`** | Error | `softmax` / `log_softmax` inside `forward()` or `__init__()` when `CrossEntropyLoss` is used (double softmax bug) |
| **`PT002`** | Error | Training loop calling `.backward()` and `.step()` without `.zero_grad()` |
| **`PT003`** | Warning | Evaluation function calling `.eval()` with a loop but missing `@torch.no_grad()`, `@torch.inference_mode()`, or `with torch.no_grad():` |
| **`PT004`** | Warning | In-place tensor ops (e.g., `.relu_()`, `.add_()`) inside `forward()` that can break autograd |
| **`PT005`** | Warning | Accumulating a graph-attached `loss` tensor directly (`total_loss += loss` instead of `loss.item()`), causing VRAM leaks |
| **`PY000`** | Error | Python syntax errors caught gracefully before AST traversal |
| **`Ruff`** | Lint | Standard Python checks (`F821` undefined names, unused imports, etc.) via `ruff` |

---

## Project Structure

```text
torchmend/
├── .github/
│   └── workflows/
│       └── ci.yml                  # Automated Ruff + Pytest checks on every PR
├── src/
│   └── torchmend/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── static_rules.py     # Module 1: PyTorch AST rules + Ruff wrapper
│       │   ├── runtime_capture.py  # Module 2: Isolated crash harness & tensor snapshot
│       │   ├── ast_slicer.py       # Module 3 (Planned): AST context extraction
│       │   ├── patcher.py          # Module 3 (Planned): Safe SEARCH/REPLACE patcher
│       │   └── pipeline.py         # Module 4 (Planned): End-to-end repair orchestrator
│       └── cli/
│           └── __init__.py         # Module 4 (Planned): Command-line entry point
├── tests/
│   ├── test_static_rules.py        # 9 unit tests for static analysis
│   └── test_runtime_capture.py     # 6 unit tests for crash capture & snapshots
├── .gitignore
├── pyproject.toml                  # Package metadata, dependencies & pytest config
└── README.md
```

---

## Local Setup (For Collaborators)

### 1. Clone the repository
```bash
git clone [https://github.com/PradhyumnaAI/torchmend.git](https://github.com/PradhyumnaAI/torchmend.git)
cd torchmend
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell blocks script activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.)*

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install in editable mode with dev tools
```bash
pip install -e ".[dev]"
```

### 4. Verify your setup
```bash
ruff check src tests
pytest -v
```
You should see `All checks passed!` from Ruff and **15 passed** from Pytest.

---

## Contribution & Git Workflow

We keep `main` clean and deployable at all times. **Do not push directly to `main`.**

1. **Sync `main` and create a feature branch:**
   ```bash
   git switch main
   git pull origin main
   git switch -c feat/<short-feature-name>
   ```
   *(Use prefixes like `feat/`, `fix/`, `docs/`, `test/`, or `ci/`.)*

2. **Write code + unit tests** for every new function or edge case under `tests/`.

3. **Run local checks before committing:**
   ```bash
   ruff check src tests
   pytest -v
   ```

4. **Commit and push your branch:**
   ```bash
   git add <files>
   git commit -m "feat: short summary of change"
   git push -u origin feat/<short-feature-name>
   ```

5. **Open a Pull Request** targeting `main`. Wait for both GitHub Actions CI checks (`Python 3.12` and `Python 3.14`) to turn green, review the diff, and merge.

6. **Clean up locally after merge:**
   ```bash
   git switch main
   git pull origin main
   git branch -d feat/<short-feature-name>
   git fetch --prune
   ```