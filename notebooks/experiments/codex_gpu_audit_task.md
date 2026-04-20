# Codex Task — Audit why profiling ran on CPU instead of GPU

Read this file and work on the repository accordingly.

## Mission

Your goal is to audit the SmartGrid repository and determine **exactly why the profiling/benchmark workflow executed on CPU instead of GPU**, even though the local project virtual environment clearly sees CUDA.

Do not waste time on broad optimization yet.  
First, find the exact cause of the CPU fallback during profiling, prove it, and fix it.

---

## Proven facts

The local environment sees CUDA correctly.

### Direct Python in project venv
```bash
python -c "import sys, torch; print('exe:', sys.executable); print('torch:', torch.__version__); print('torch cuda build:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count()); print('device 0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

Observed result:
- executable: `.venv/bin/python`
- torch: `2.11.0+cu128`
- torch cuda build: `12.8`
- cuda available: `True`
- device count: `1`
- device 0: `NVIDIA GeForce RTX 3080 Laptop GPU`

### `uv run` in the same project
```bash
uv run python -c "import sys, torch; print('exe:', sys.executable); print('torch:', torch.__version__); print('torch cuda build:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count()); print('device 0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

Observed result:
- executable: `.venv/bin/python3`
- torch: `2.11.0+cu128`
- torch cuda build: `12.8`
- cuda available: `True`
- device count: `1`

So:
- the repository virtualenv is correct
- torch is a CUDA build
- CUDA is visible from both `python` and `uv run python`

---

## But profiling reports show the opposite

The profiling run recorded:
- `cuda_available: false`
- `selected_device: cpu`

The profiling command used `--device auto`, not `--device cpu`.

The generated training run summary also says:
- `device: "cpu"`
- `cuda_available: false`
- `selected_device: "cpu"`

So this is not just a reporting mistake.  
The actual profiled training run really executed on CPU.

This means there is an inconsistency between:
- the normal local runtime environment, which sees CUDA
- and the profiling workflow, which somehow ran in a CPU-only context

---

## Main hypothesis

The most likely cause is that the profiling workflow introduced by previous changes is modifying or losing the environment when launching subprocesses.

The primary suspect is:
- `scripts/profile_pipeline.py`

Secondary suspects:
- any logic added for `--profile` in training code
- any wrapper or helper that reconstructs environment variables
- any use of `subprocess.run(..., env=...)`
- any accidental forcing of CPU during profiling
- any accidental override of `CUDA_VISIBLE_DEVICES`

---

## Your objectives

### 1. Audit all profiling-related modifications
Inspect all modified / untracked / newly added files related to profiling, benchmarking, or wrapper execution.

Prioritize:
- `scripts/profile_pipeline.py`
- `scripts/train_consumption.py`
- `src/smartgrid/cli/train_consumption.py`
- `src/smartgrid/training/trainer.py`
- any shared profiling/timing utility module
- any file that uses:
  - `subprocess.run`
  - `subprocess.Popen`
  - `env=...`
  - `os.environ`
  - `CUDA_VISIBLE_DEVICES`
  - `torch.cuda.is_available()`
  - device selection overrides

You must determine:
- where CUDA visibility is lost
- whether it happens in the wrapper script or in the actual training path
- whether profiling mode explicitly or implicitly forces CPU

### 2. Reproduce the issue carefully
Run the minimal checks needed to isolate the bug.

You must compare at least:

#### A. direct train run without wrapper
Example:
```bash
uv run python scripts/train_consumption.py --config configs/consumption/mlp_baseline.yaml --analysis-days 1 --promote --profile
```

#### B. profiling wrapper run
Example:
```bash
uv run python scripts/profile_pipeline.py --config configs/consumption/mlp_baseline.yaml --analysis-days 1 --predict-target-date 2026-01-15 --replay-start-date 2026-01-01 --replay-end-date 2026-01-07 --device auto
```

For both, capture and compare:
- `sys.executable`
- `torch.__version__`
- `torch.version.cuda`
- `torch.cuda.is_available()`
- `torch.cuda.device_count()`
- `torch.cuda.get_device_name(0)` if available
- selected training device
- relevant environment variables, especially `CUDA_VISIBLE_DEVICES`

### 3. Add a preflight diagnostic if needed
If current scripts do not expose enough information, add a lightweight and explicit preflight diagnostic.

It should print or record:
- executable path
- python version
- torch version
- torch CUDA build
- `torch.cuda.is_available()`
- device count
- device names
- `CUDA_VISIBLE_DEVICES`
- final selected device
- whether the run is using profiling mode

This diagnostic should be safe on CPU-only environments too.

### 4. Identify the exact root cause
Do not stop at symptoms.  
Find the precise mechanism.

Examples of acceptable root causes:
- wrapper passes `env={...}` without inheriting `os.environ.copy()`
- wrapper clears `CUDA_VISIBLE_DEVICES`
- profiling mode explicitly forces CPU
- `--profile` path bypasses normal device selection
- environment differences between subprocesses
- incorrect use of device argument propagation

You must state clearly:
- what the bug is
- where it is in code
- why it causes CPU fallback
- how to fix it

### 5. Fix it cleanly
Implement the minimum correct fix so that profiling runs respect the normal device selection behavior.

Requirements:
- preserve CPU/GPU portability
- do not make the project GPU-only
- do not break standard training/inference logic
- profiling mode must remain compatible with CPU-only environments
- if the user requests `--device auto`, CUDA should be used when available
- if the user requests `--device cpu`, CPU must still be honored

If the bug is in environment forwarding, use the correct inheritance strategy.

---

## Important constraints

- Do **not** redesign the model or training logic yet.
- Do **not** do large refactors unrelated to this bug.
- Do **not** assume the business forecasting logic should change.
- Do **not** introduce forecast-on-forecast recursion.
- Keep fixes targeted and easy to review.

---

## Deliverables

At the end, provide:

### A. Root-cause explanation
A concise but precise explanation of:
- why profiling ran on CPU
- where the bug was
- how you proved it

### B. Code changes
List exactly which files you modified.

### C. Commands run
List the exact commands used for diagnosis and verification.

### D. Verification result
Show that after the fix:
- direct training sees CUDA correctly
- profiling workflow also sees CUDA correctly
- `--device auto` now selects CUDA when available

### E. Final status
Explicitly say whether the bug is fully fixed.

---

## Extra validation

If possible, after fixing, rerun one short profiling job and confirm that:
- `cuda_available: true`
- `selected_device: cuda`
- GPU name is correctly detected

Do not optimize performance yet unless it is required to verify the fix.

---

## Practical search checklist

Search for and inspect all occurrences of:
```bash
CUDA_VISIBLE_DEVICES
subprocess.run
subprocess.Popen
env=
os.environ
torch.cuda.is_available
selected_device
device_request
--device
--profile
profile_pipeline
```

---

## Priority

This is a debugging task, not a performance tuning task.
Your top priority is to find and fix the CPU-vs-GPU inconsistency in the profiling path.
