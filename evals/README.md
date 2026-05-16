# Experiment-Aware Evaluation Harness

## Overview

This directory contains the experiment-aware evaluation harness for the CARLA Perception Lab's LiDAR 3D detection pipeline. The harness is designed to **safely evaluate future experiments** aimed at improving mAP, without allowing regressions or collapse to go undetected.

**Scope**: Simulator-only perception. No real-road deployment. No RL driving agents.

## Current Global Best

| Metric | Value |
|---|---|
| mAP | 0.4591 |
| Vehicle AP | 0.4736 |
| Pedestrian AP | 0.4445 |
| Result file | `output/detection_3d_canonical_mix_v3/eval_results_fusion_rescore_best.json` |

**Target**: mAP ≥ 0.60

## Architecture

```
evals/
├── experiment_config.yaml      # Fixed baseline, targets, thresholds (PROTECTED)
├── experiment_registry.jsonl   # Append-only experiment log
├── check_env.py                # Environment guard
├── check_dataset_quality.py    # Dataset structural validation
├── sweep_checkpoints.py        # Checkpoint sweep with metric extraction
├── select_best_checkpoint.py   # Best checkpoint selection + collapse detection
├── compare_experiments.py      # Experiment-vs-baseline comparison
├── score.py                    # Scoring utilities (PROTECTED)
├── gating.py                   # Pass/fail gate (PROTECTED)
├── config.yaml                 # Legacy config (references experiment_config.yaml)
├── benchmark.py                # Benchmark utilities
├── protected_cases/            # Protected test data (PROTECTED)
└── README.md                   # This file
```

## Workflow

### 1. Environment Check
```bash
python evals/check_env.py --expected-env pcdet
```

### 2. Dataset Quality Check
```bash
python evals/check_dataset_quality.py \
  --experiment-id EXP001 \
  --dataset data/processed/pcdet_format_ped_strict20_easy4_v1
```

### 3. Checkpoint Sweep

**Scan existing eval results:**
```bash
python evals/sweep_checkpoints.py \
  --experiment-id EXP001 \
  --scan-existing output/detection_3d_ped_strict20_easy4_v1 \
  --start-epoch 51 --end-epoch 80
```

**Run new evaluations:**
```bash
python evals/sweep_checkpoints.py \
  --experiment-id EXP001 \
  --checkpoint-dir repos/OpenPCDet/output/.../ckpt \
  --config configs/carla_lidar_ped_strict20_easy4_v1.yaml \
  --start-epoch 51 --end-epoch 80 \
  --conda-env pcdet
```

### 4. Best Checkpoint Selection
```bash
python evals/select_best_checkpoint.py \
  --experiment-id EXP001 \
  --sweep outputs/benchmarks/checkpoint_sweep_EXP001.json
```

### 5. Experiment Comparison
```bash
python evals/compare_experiments.py \
  --experiment-id EXP001 \
  --current outputs/benchmarks/best_checkpoint_EXP001.json \
  --baseline output/detection_3d_canonical_mix_v3/eval_results_fusion_rescore_best.json
```

### 6. Gating
```bash
# Provisional: pass if improves baseline by min_delta
python evals/gating.py --mode provisional --experiment-id EXP001

# Target: pass only if mAP >= 0.60
python evals/gating.py --mode target --experiment-id EXP001
```

## Stop Conditions

**Stop success**: mAP ≥ 0.60, Vehicle AP ≥ 0.45, Pedestrian AP ≥ 0.45, all checks pass.

**Stop fail**: Max trials reached, no improvement for 2 consecutive trials, collapse detected, wrong env, 0-sample dataset, modified protected files.

## Known Failed Experiments

| Experiment | Best mAP | Status |
|---|---|---|
| EASY4_FAILED (ped_strict20_easy4_v1) | 0.0212 | COLLAPSED |

## Protected Files

Do not modify without documentation:
- `evals/experiment_config.yaml`
- `evals/score.py`
- `evals/gating.py`
- `evals/protected_cases/`
- Ground-truth labels
- Benchmark expected outputs
