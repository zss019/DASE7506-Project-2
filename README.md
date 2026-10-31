# DASE7506 Mini-project 2 — Reliable Answer Selection

Student ID: **3036837930**. Submitted model: the frozen **Top 2 matched-OOF numeric ensemble**. Official test result: **387/467 = 82.86937901498929%**; validation: **752/798 = 94.23558897243107%**.

Read [the report](report/report.pdf), [AI/reuse disclosure](DISCLOSURE.md), and [experiment records](results/experiments.json). The published checkpoint is the evaluated artifact; reproducing its score does not require training. The official evaluator and parser in `code/` are unchanged. This repository contains no benchmark data; obtain it from the course repository.

## Installation and official data

Use Python 3.12. Run from this repository's root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
git clone https://github.com/xudongwu-0/DASE7506-MP2.git ../DASE7506-MP2-official
export MP2_DATA="$(cd ../DASE7506-MP2-official/code/data && pwd)"
python code/check_data.py --data-dir "$MP2_DATA"
python scripts/verify_manifest.py
```

`requirements.txt` pins the CPU inference dependencies. No PyTorch, external model, API key, or internet connection is required during prediction/scoring. Data identity is checked against the supplied manifest. See `results/official_integrity.json` for the exact official file hashes used.

## Reproduce the checkpoint result

Each output directory must be new:

```bash
python scripts/evaluate.py --data-dir "$MP2_DATA" --split validation --output runs/validation
python scripts/evaluate.py --data-dir "$MP2_DATA" --split test --output runs/test
```

The wrapper launches the official predictor and independent scorer as fresh CPU processes, sets all three thread environment variables to 4, disables CUDA, and records external wall time in `runtime.json`. The test `score.json` should contain `correct: 387`, `n: 467`, and `accuracy: 82.86937901498929`.

Equivalent direct commands:

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
python code/predict.py --method code/answer_ensemble_fast_v2.py --checkpoint checkpoint --split test --data-dir "$MP2_DATA" --output runs/test-predictions.json
python code/score_predictions.py --split test --data-dir "$MP2_DATA" --predictions runs/test-predictions.json --output runs/test-score.json
```

For instructor audit inputs use the same predictor with `--split audit --data-dir /path/to/audit_inputs`. The prediction function consumes only the passed bank/arrays; IDs are not model features. Stored result predictions are evidence only and are never inference inputs.

## Train the final configuration from scratch

Training was verified with Python 3.12.3, NumPy 2.4.6, SciPy 1.16.3, PyTorch 2.12.1+cu130 and one RTX 3080 Ti. The training command reads only supplied training inputs/references, uses no cached OOF arrays or pretrained checkpoints, and fits all 44 model components. Use a CUDA GPU within the course's RTX 4090 limit. CPU inference remains independent of PyTorch.

```bash
python -m venv .venv-train
source .venv-train/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python code/train_final.py --data-dir "$MP2_DATA" --output retrained/model
python scripts/evaluate.py --data-dir "$MP2_DATA" --checkpoint retrained/model --split test --output runs/retrained-test
```

Use a fresh `--output` path. Training writes a sibling `model-training.json` with seeds, fold timings, hardware and versions. The three neural seeds are 17/29/43, with 15/10/10 epochs. OOF partition seeds are 20261031/20261107/20261203, each using three folds stratified by candidate answer agreement. The third partition retains a historical floating-point blend/recovery operation for numerical compatibility; it changes no training information.

Full retraining was checked on the recorded CUDA stack: base parameters were bitwise identical; residual parameters differed by at most 4.04e-9, with identical complete validation and test predictions. Different hardware/library builds may produce numerical differences. The exact submitted checkpoint, not a newly retrained weight file, is the score reproduction artifact. See `results/training_reproduction.json`.

## Reproduce the fixed-weight validation ablation

```bash
python scripts/ablate_validation.py --data-dir "$MP2_DATA" --output runs/ablation.json
python scripts/compare_weights.py checkpoint retrained/model
```

The ablation does not refit weights or read test data.

## Files and resources

- `checkpoint/`: all additional inference assets, 2,077,521 bytes (about 1.98 MiB).
- `code/`: inference, unchanged official evaluator/parser, and self-contained training implementation.
- `report/report.pdf`: five-page experimental report; source in `report/report_source.json`.
- `results/`: official scores, fresh-process timings, matched retraining evidence, validation ablations and search summaries.
- `MANIFEST.json`: SHA-256 hashes of the release's files. `checkpoint.zip` is a separate distribution of the same inference assets.
- `LICENSE.txt` and `NOTICE.txt`: retained course-data attribution; data is not redistributed here.

Measured test prediction plus scoring: 5.35 seconds on a Xeon Gold 6248, using four CPU threads; the instructor's reference Xeon Platinum 8457C determines eligibility. Independent full retraining took about 99 seconds on the recorded environment. Search costs and adaptive-validation limitations are documented in the report. GPU training was verified in the existing pinned training environment; CPU checkpoint evaluation was verified in a newly created virtual environment.

## Rebuild the PDF

```bash
python -m pip install reportlab==4.4.10
python scripts/build_report.py
```

## Release and course entry

The release tag is `submission-3036837930-v1`. Code and checkpoint links should point to that immutable tag or its commit. The existing course score entry is [Issue #381](https://github.com/xudongwu-0/xudongwu-0.github.io/issues/381). Link submission is separate from publishing this repository and must be completed through the course website using the original GitHub account. See `SUBMISSION.json` for the intended links and exact scorer value.
