# Offline Deployment Guide

## Scope
This deployment package runs local offline inference and dashboard rendering on pre-recorded CARLA data. It is simulator-only.

## Prerequisites
- NVIDIA GPU (tested on RTX 4060 Laptop 8GB)
- NVIDIA driver and CUDA runtime
- Conda environments: `carla-client`, `pad`, `pcdet`
- Existing checkpoints in `data/checkpoints/`

## One-Command Setup + Run
```bash
bash scripts/deploy_local.sh
```

## Manual Flow
```bash
# 1) Prepare 100-frame raw subset with normalized IDs
python scripts/prepare_demo_data.py \
  --input-dir data/raw/recording_001 \
  --output-dir deploy/demo_data \
  --num-frames 100 \
  --overwrite

# 2) Run offline demo pipeline
bash run_demo.sh --input-dir deploy/demo_data --output-dir output/demo --overwrite
```

## Docker Option
```bash
docker compose build
docker compose run --rm carla-perception-lab-dashboard
```

## Outputs
- `output/demo/segmentation/predictions/`
- `output/demo/detection_3d/predictions/`
- `output/demo/dashboard/demo_video.mp4`
- `output/demo/dashboard/dashboard_report.json`

## Notes
- `run_demo.sh` runs tasks sequentially to keep VRAM usage bounded.
- If environment export is unavailable, run `scripts/deploy_local.sh --skip-freeze`.
- Docker path is intended for dashboard/runtime helper usage; full inference pipeline uses host conda envs.
- This stage does not claim real-road deployment suitability.
