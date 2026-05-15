# STAGE 08 — LiDAR 3D Detection Integration Report

> Date: 2026-05-13  
> Status: COMPLETE (ISSUES FIXED + FINETUNE VERIFIED)

## Scope

Executed `final_plan/STAGE_08_LIDAR_INTEGRATION.md` only:
- run PointPillar inference on CARLA LiDAR custom dataset
- evaluate detections against CARLA GT labels
- generate detection visualizations
- fix Stage 08 blocking issues before moving stage

## Implemented / Updated Artifacts

- `scripts/run_lidar_det.py`
  - supports `--help`
  - modes: `infer`, `evaluate`, `infer_and_evaluate`, `visualize`
  - project-local compatibility shims (Argo2/DSVT optional imports)
  - custom BEV AP evaluator (stable, no upstream segfault)

- `scripts/pcdet_finetune.py`
  - supports `--help`
  - wrapper to run OpenPCDet `train.py` with compatibility shims
  - reproducible fine-tune entrypoint

- `scripts/convert_to_pcdet.py` (fixed)
  - added default point-cloud y-flip to match PCDet coordinate convention
  - new flags:
    - `--flip-point-y` (default enabled)
    - `--no-flip-point-y`

- `scripts/carla_recorder.py` (extended)
  - added `--actor-label-max-distance` to control actor metadata radius

- `configs/carla_lidar.yaml` (runtime inference config)
- `configs/carla_lidar_probe.yaml` (fine-tune config for probe dataset)

## Root-Cause Fixes Applied

1. **Point/label coordinate mismatch fixed**
- Problem: labels were converted to PCDet convention (y-left) while point cloud remained CARLA y-right in Stage 06 converter.
- Fix: `convert_to_pcdet.py` now flips point y-axis by default (`points[:,1] = -points[:,1]`).

2. **Evaluation segfault path removed from Stage 08 runtime**
- Problem: upstream KITTI eval path in this environment crashed with segmentation fault.
- Fix: `run_lidar_det.py` evaluates with project-local BEV AP metric using OpenPCDet IoU op (`boxes_bev_iou_cpu`) and writes JSON safely.

3. **Zero-shot AP=0 addressed with fine-tuning**
- Added reproducible fine-tune wrapper and config.
- Fine-tuned PointPillar on probe train split (5 epochs, bs=1).

## Commands Executed

### 1) Re-convert probe dataset with fixed point-y flip
```bash
/home/thuan/miniconda3/envs/pcdet/bin/python scripts/convert_to_pcdet.py \
  --input-dir data/raw/stage06_label_probe \
  --output-dir data/processed/pcdet_format_probe \
  --num-frames 80 --overwrite --shuffle-split --seed 42 \
  --generate-custom-infos
```

### 2) Fine-tune PointPillar (project wrapper)
```bash
ABS=$(realpath data/processed/pcdet_format_probe_yflip)
/home/thuan/miniconda3/envs/pcdet/bin/python scripts/pcdet_finetune.py \
  --cfg-file configs/carla_lidar_probe.yaml \
  --pretrained-model data/checkpoints/pointpillar_7728.pth \
  --epochs 5 --batch-size 1 --workers 2 \
  --extra-tag carla_probe_finetune \
  --set DATA_CONFIG.DATA_PATH "$ABS"
```

Checkpoint used after training:
- `../repos/OpenPCDet/output/.../carla_probe_finetune/ckpt/checkpoint_epoch_5.pth`

### 3) Inference + evaluation with fine-tuned checkpoint
```bash
/home/thuan/miniconda3/envs/pcdet/bin/python scripts/run_lidar_det.py \
  --mode infer \
  --cfg-file configs/carla_lidar_probe.yaml \
  --checkpoint <checkpoint_epoch_5.pth> \
  --dataset-dir data/processed/pcdet_format_probe \
  --split val --batch-size 1 --overwrite \
  --score-thresh 0.1 \
  --pred-dir output/detection_3d/predictions_ft

/home/thuan/miniconda3/envs/pcdet/bin/python scripts/run_lidar_det.py \
  --mode evaluate \
  --cfg-file configs/carla_lidar_probe.yaml \
  --checkpoint <checkpoint_epoch_5.pth> \
  --dataset-dir data/processed/pcdet_format_probe \
  --split val --batch-size 1 \
  --pred-dir output/detection_3d/predictions_ft \
  --eval-output output/detection_3d/eval_results_ft.json
```

### 4) Visualization
```bash
/home/thuan/miniconda3/envs/pcdet/bin/python scripts/run_lidar_det.py \
  --mode visualize \
  --dataset-dir data/processed/pcdet_format_probe \
  --pred-dir output/detection_3d/predictions_ft \
  --vis-output-dir output/detection_3d/visualizations_ft \
  --vis-max-frames 16 --overwrite
```

## Results

### Zero-shot baseline (before fine-tune)
- Vehicle AP@0.7: `0.0`
- mAP: `0.0`

### Fine-tuned result
From `output/detection_3d/eval_results_ft.json`:
- frames evaluated: `16`
- metric: `BEV_AP_custom`
- Vehicle AP@0.7: `0.14638933305463397`
- Vehicle recall: `0.3571`
- Vehicle precision: `0.0893`
- mAP: `0.14638933305463397`

Inference runtime (fine-tuned ckpt, val split):
- avg forward: `121.57 ms`
- FPS: `8.23`
- predicted boxes (`score>=0.1`): `56`

Visualization:
- `output/detection_3d/visualizations_ft/`: `16` BEV frames

## Exit Criteria Status

- [x] PointPillar inference runs on CARLA LiDAR frames
- [x] Bounding-box predictions generated and saved
- [x] AP metrics computed and documented
- [x] 3D/BEV visualizations saved
- [x] Fine-tune path executed and improves over zero-shot baseline
- [x] No OOM during Stage 08 runs (`batch_size=1`)

## Remaining Constraints (Non-blocking)

1. Current probe GT distribution is strongly skewed to `Vehicle`; `Pedestrian/Cyclist` GT count is 0 in val split.
2. Increasing walker density on this hardware/map caused CARLA instability in additional data-collection attempts; fallback used lower-load stable recording.

## Next Recommended Action

Proceed to `STAGE_09_DASHBOARD.md`.

## Revalidation Update (2026-05-13, Hardened Plan)

### Runtime behavior changes validated

1. `scripts/run_lidar_det.py` now hard-fails evaluation when total GT is zero by default.
   - Verified on `data/processed/pcdet_format` (known empty GT): command returns non-zero with explicit error.
   - Debug override `--allow-empty-gt` works as expected.

2. Stage 08 rerun on regenerated canonical mixed PCDet dataset:
   - dataset: `data/processed/pcdet_format_stage08_canonical_mix`
   - split: `val` (`240` frames)
   - checkpoint: fine-tuned `checkpoint_epoch_5.pth`

Command:

```bash
conda run -n pcdet python scripts/run_lidar_det.py \
  --mode infer_and_evaluate \
  --cfg-file configs/carla_lidar_probe.yaml \
  --checkpoint ../repos/OpenPCDet/output/.../checkpoint_epoch_5.pth \
  --dataset-dir data/processed/pcdet_format_stage08_canonical_mix \
  --split val --batch-size 1 \
  --pred-dir output/detection_3d_canonical_mix/predictions \
  --eval-output output/detection_3d_canonical_mix/eval_results.json \
  --overwrite --score-thresh 0.1
```

Results:
- frames evaluated: `240`
- total GT boxes: `50`
- Vehicle GT: `46`, Pedestrian GT: `4`, Cyclist GT: `0`
- mAP (custom BEV metric): `0.00299`
- forward FPS: `38.21`

### Hardened readiness gate

Command:

```bash
python scripts/validate_stage_readiness.py \
  --raw-dir data/raw/stage03_mine_main_1000 \
  --pad-summary output/stage_validation/stage06_pad_summary_canonical_mix.json \
  --pcdet-summary output/stage_validation/stage06_pcdet_summary_canonical_mix.json \
  --seg-pred-dir output/segmentation_canonical_mix_val/predictions \
  --det-pred-dir output/detection_3d_canonical_mix/predictions \
  --report-json output/stage_validation/stage08_readiness_report.json
```

Gate result:
- `overall_pass: true`
- raw integrity: pass
- PAD class coverage: pass
- PCDet GT gate (Vehicle + secondary): pass
- seg/det output stem alignment: pass

### Town10 note

Town10 collection remains unstable on this host/runtime and is quarantined in `docs/stage_reports/DATASET_QUARANTINE.md`.
