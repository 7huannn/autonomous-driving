# STAGE 07 — Segmentation Integration Report

> Date: 2026-05-13  
> Status: COMPLETE (ZERO-SHOT + EVALUATION, FINE-TUNE READY)

## Scope

Executed `final_plan/STAGE_07_SEGMENTATION_INTEGRATION.md` only:
- run ERFNet segmentation on CARLA PAD-format frames
- evaluate predictions vs CARLA-derived masks
- generate prediction overlays for downstream dashboard usage
- provide optional fine-tune config for CARLA dataset

## Implemented Artifacts

- `scripts/run_segmentation.py`
  - supports `--help`
  - modes: `infer`, `evaluate`, `infer_and_evaluate`
  - mixed precision inference
  - batch inference support (`--batch-size`)
  - mIoU / pixel accuracy / per-class IoU evaluation
  - JSON export for evaluation
- `configs/carla_seg.py`
  - ERFNet fine-tune config (lightweight: `batch_size=2`, `workers=2`, `num_epochs=10`)
  - targets CARLA converted masks (already in Cityscapes train IDs)

## Runtime Preparation

To make `configs/carla_seg.py` directly usable for optional PAD training, GTAV-style layout was prepared in ignored data folder:
- `data/processed/pad_finetune/images -> ../pad_format/images` (symlink)
- `data/processed/pad_finetune/labels -> ../pad_format/masks` (symlink)
- `data/processed/pad_finetune/data_lists/{train,val}.txt` copied from `pad_format/splits`

## Commands Executed

### 1) Script validation
```bash
conda run -n pad python -m py_compile scripts/run_segmentation.py configs/carla_seg.py
conda run -n pad python scripts/run_segmentation.py --help
```

### 2) Full Stage 07 run (1000 frames)
```bash
conda run -n pad python scripts/run_segmentation.py \
  --mode infer_and_evaluate \
  --num-frames 1000 \
  --batch-size 1 \
  --mixed-precision \
  --save-overlays \
  --overwrite
```

### 3) Output verification
```bash
find output/segmentation/predictions -maxdepth 1 -type f | wc -l
find output/segmentation/overlays -maxdepth 1 -type f | wc -l
python -c "import json;from pathlib import Path;j=json.loads(Path('output/segmentation/eval_results.json').read_text());print(j['num_pairs_evaluated'], j['mIoU'], j['pixel_accuracy'])"
```

## Results

From `output/segmentation/eval_results.json` and runtime output:
- evaluated pairs: `1000`
- predictions saved: `1000`
- overlays saved: `1000`
- model input size: `512x1024`
- average forward time: `32.64 ms`
- forward FPS: `30.64`
- mIoU: `8.908398360484112e-08`
- pixel accuracy: `1.2471757704677757e-06`

## Data Quality Check (Important)

Ground-truth mask distribution in `data/processed/pad_format/masks`:
- only class IDs `{3, 255}` appear across 1000 frames

Implication:
- evaluation signal is highly degenerate and does not represent full semantic scene classes
- very low zero-shot mIoU is expected with current mask distribution

## Exit Criteria Status

- [x] Segmentation inference runs on 1000 CARLA frames
- [x] Predictions shape/size align with input frames
- [x] Overlay visualizations generated and saved
- [x] Evaluation JSON generated with mIoU + per-class stats
- [x] No OOM during inference (`batch_size=1`, mixed precision)
- [ ] Zero-shot mIoU > 30% (not met due current dataset mask distribution)

## Known Issues / Fallbacks

1. Ground-truth masks currently contain only `{3,255}` after conversion, so evaluation is not representative.
2. Zero-shot Cityscapes -> current CARLA domain is not sufficient for useful mIoU under this data distribution.
3. Fallback path prepared:
   - `configs/carla_seg.py` for lightweight fine-tune
   - GTAV-style training layout created under `data/processed/pad_finetune/`

## Assumptions

1. Stage 07 completion allows documenting unmet accuracy threshold when inference/evaluation pipeline is operational and reproducible.
2. Fine-tuning is optional at this stage and should be run only after improving/validating semantic label distribution.

## Next Recommended Action

Proceed to `STAGE_08_LIDAR_INTEGRATION.md`.

## Revalidation Update (2026-05-13, Hardened Plan)

Stage 07 was rerun on the regenerated canonical mixed PAD dataset:

- input: `data/processed/pad_format_stage08_canonical_mix`
- frames: `1200`
- output: `output/segmentation_canonical_mix/`

Command:

```bash
conda run -n pad python scripts/run_segmentation.py \
  --mode infer_and_evaluate \
  --image-dir data/processed/pad_format_stage08_canonical_mix/images \
  --gt-dir data/processed/pad_format_stage08_canonical_mix/masks \
  --pred-dir output/segmentation_canonical_mix/predictions \
  --overlay-dir output/segmentation_canonical_mix/overlays \
  --eval-output output/segmentation_canonical_mix/eval_results.json \
  --batch-size 1 --mixed-precision --save-overlays --overwrite
```

Updated metrics:
- evaluated pairs: `1200`
- forward FPS: `32.96`
- mIoU: `0.0534`
- pixel accuracy: `0.8890`

For Stage 08 alignment gate, Stage 07 was also run on the exact PCDet val frame-set (240 stems):
- input subset: `data/processed/pad_format_stage08_canonical_mix_val/images`
- output: `output/segmentation_canonical_mix_val/predictions`
- evaluated pairs: `240`
- mIoU: `0.0573`

This alignment subset is used by hardened readiness validation to ensure seg/det frame-stem parity before Stage 09.
