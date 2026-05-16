# Agent Experiment Protocol

## Purpose

This document defines the protocol for proposing, running, evaluating, and recording LiDAR 3D detection experiments in the CARLA Perception Lab. All agents (human or AI) must follow this protocol to maintain reproducibility and prevent regressions.

**Scope**: Simulator-only perception (CARLA). Not a full autonomous driving stack.

---

## 1. Current Global Best

| Metric | Value |
|--------|-------|
| **mAP** | **0.4591** |
| Vehicle AP | 0.4736 |
| Pedestrian AP | 0.4445 |
| Result file | `output/detection_3d_canonical_mix_v3/eval_results_fusion_rescore_best.json` |
| Method | Fusion rescore of epoch 7 (Vehicle) + epoch 5 (Pedestrian) |

**Target**: mAP ≥ 0.60 with Vehicle AP ≥ 0.45 and Pedestrian AP ≥ 0.45.

---

## 2. How to Propose a New Experiment

Before starting any training or evaluation:

1. **Define a hypothesis**: What change do you expect to improve mAP?
2. **Choose an experiment ID**: Use format `EXP_<short_name>` (e.g., `EXP_ANCHOR_TUNE_V1`).
3. **Verify prerequisites**:
   - Dataset exists and passes quality check
   - Correct conda environment (`pcdet`)
   - Config is validated
   - No more than `max_trials_per_plan` (5) active trials
4. **Do NOT**:
   - Modify protected paths without documentation
   - Start training inside a CARLA session
   - Train on a 0-sample or extremely small dataset
   - Exceed `max_epochs_per_trial` (80)

---

## 3. How to Register an Experiment

Append a JSON line to `evals/experiment_registry.jsonl`:

```json
{
  "experiment_id": "EXP_MY_EXPERIMENT",
  "timestamp": "2026-05-16T00:00:00Z",
  "git_commit": "<commit hash>",
  "dataset_path": "data/processed/<dataset_name>",
  "config_path": "configs/<config>.yaml",
  "checkpoint_dir": "repos/OpenPCDet/output/.../ckpt",
  "checkpoint_range": "1-80",
  "eval_command": "scripts/run_lidar_det.py",
  "conda_env": "pcdet",
  "numpy_version": "<version>",
  "pcdet_import_ok": true,
  "best_epoch": null,
  "best_result_file": null,
  "mAP": null,
  "Vehicle_AP": null,
  "Pedestrian_AP": null,
  "status": "registered",
  "failure_reason": null,
  "notes": "Description of hypothesis and changes."
}
```

**Never delete or overwrite old records.** Update the record by appending a new line with the same `experiment_id` and updated fields.

---

## 4. How to Check Environment

```bash
python evals/check_env.py --expected-env pcdet
```

This verifies:
- Correct conda environment is active
- Required packages: `easydict`, `numpy`, `torch`
- Optional: `pcdet` importable
- Writes: `outputs/benchmarks/env_check.json`

**If this fails**, do not proceed with evaluation.

---

## 5. How to Check Dataset Quality

```bash
python evals/check_dataset_quality.py \
  --experiment-id EXP001 \
  --dataset data/processed/pcdet_format_<name>
```

This checks:
- Dataset path exists
- `custom_infos_train.pkl`, `custom_infos_val.pkl` exist
- `ImageSets/train.txt`, `ImageSets/val.txt` exist and are non-empty
- Label and point cloud file counts match
- Class distribution from label files
- Writes: `outputs/benchmarks/dataset_quality_<experiment_id>.json`

**Fail conditions**: 0 samples, missing val split, missing info files.

---

## 6. How to Sweep Checkpoints

**Scan existing results** (no training needed):
```bash
python evals/sweep_checkpoints.py \
  --experiment-id EXP001 \
  --scan-existing output/detection_3d_<name> \
  --start-epoch 1 --end-epoch 80
```

**Run new evaluations**:
```bash
python evals/sweep_checkpoints.py \
  --experiment-id EXP001 \
  --checkpoint-dir repos/OpenPCDet/output/.../ckpt \
  --config configs/<config>.yaml \
  --start-epoch 1 --end-epoch 80 \
  --conda-env pcdet
```

Writes: `outputs/benchmarks/checkpoint_sweep_<experiment_id>.{json,csv}`

---

## 7. How to Select Best Checkpoint

```bash
python evals/select_best_checkpoint.py \
  --experiment-id EXP001 \
  --sweep outputs/benchmarks/checkpoint_sweep_EXP001.json
```

Selection criteria:
1. Highest mAP
2. Tie-break: highest Pedestrian AP
3. Tie-break: highest Vehicle AP

Writes: `outputs/benchmarks/best_checkpoint_<experiment_id>.json`

---

## 8. How to Compare Against Baseline

```bash
python evals/compare_experiments.py \
  --experiment-id EXP001 \
  --current outputs/benchmarks/best_checkpoint_EXP001.json \
  --baseline output/detection_3d_canonical_mix_v3/eval_results_fusion_rescore_best.json
```

Reports: absolute/relative delta mAP, class-level deltas, pass/fail decision.

Writes: `outputs/benchmarks/experiment_comparison_<experiment_id>.json`

---

## 9. When to Stop

### Stop Success
- mAP ≥ 0.60
- Vehicle AP ≥ 0.45
- Pedestrian AP ≥ 0.45
- No protected paths modified
- Dataset quality passed
- Environment check passed

### Stop Fail
- `max_trials_per_plan` (5) reached
- `max_no_improve_trials` (2) consecutive non-improving trials
- Experiment collapse detected (mAP < 50% of baseline)
- Wrong conda environment
- 0-sample dataset
- Modified protected eval files
- Score lower than baseline by more than tolerance

---

## 10. When NOT to Train

- Do not train if the environment check fails
- Do not train if dataset quality check fails
- Do not train on datasets with fewer than ~50 samples without explicit approval
- Do not train while CARLA simulator is running (GPU memory conflict on RTX 4060)
- Do not start a new training run to "fix" a collapsed experiment without a fundamentally different approach

---

## 11. Why the easy4_v1 Branch Collapsed

The `pcdet_format_ped_strict20_easy4_v1` experiment collapsed because:

1. **Tiny dataset**: Only 20 samples total (16 train / 4 val)
2. **Augmentation disabled**: `DISABLE_AUG_LIST` contained all augmentors
3. **Narrow point cloud range**: `[-16.0, -25.6, -3.0, 48.0, 25.6, 2.0]`
4. **Pedestrian anchor mismatch**: Bottom height at -3.05 was likely incorrect
5. **Best mAP = 0.0212**: Far below baseline of 0.4591 (95.4% worse)
6. **Only 4 validation frames** with 18 GT boxes total — statistically unreliable

**Conclusion**: This branch must NOT replace the global best. Any future experiment must use a substantially larger and better-configured dataset.

---

## 12. Why mAP > 0.6 Must Be Validated Against Fixed Split and Metric

To prevent gaming or accidental inflation:

- All experiments must be evaluated on the **same validation split** as the baseline
- The evaluation metric must be **BEV_AP_custom** as defined in the baseline
- Changing the val split, IoU thresholds, or score thresholds invalidates the comparison
- The `experiment_config.yaml` defines the fixed baseline and is a **protected file**
- Any claim of mAP > 0.6 must be reproducible from the checkpoint + config + dataset

---

## 13. Full Pipeline Example

```bash
# Step 1: Environment check
python evals/check_env.py --expected-env pcdet

# Step 2: Dataset quality
python evals/check_dataset_quality.py \
  --experiment-id EXP_NEW \
  --dataset data/processed/pcdet_format_<name>

# Step 3: Checkpoint sweep (scan existing)
python evals/sweep_checkpoints.py \
  --experiment-id EXP_NEW \
  --scan-existing output/detection_3d_<name> \
  --start-epoch 1 --end-epoch 80

# Step 4: Select best
python evals/select_best_checkpoint.py \
  --experiment-id EXP_NEW \
  --sweep outputs/benchmarks/checkpoint_sweep_EXP_NEW.json

# Step 5: Compare
python evals/compare_experiments.py \
  --experiment-id EXP_NEW \
  --current outputs/benchmarks/best_checkpoint_EXP_NEW.json

# Step 6: Gate
python evals/gating.py --mode provisional --experiment-id EXP_NEW
python evals/gating.py --mode target --experiment-id EXP_NEW
```
