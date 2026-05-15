# Stage 11 Deployment

## Goal
Provide a reproducible local offline inference/deployment path for the perception pipeline without requiring a running CARLA simulator.

## Deliverables
- `Dockerfile`
- `docker-compose.yml`
- `scripts/deploy_local.sh`
- `run_demo.sh`
- `scripts/prepare_demo_data.py`
- `scripts/make_dashboard.py`
- `deploy/DEPLOY_README.md`

## Quick Start
```bash
# Prepare demo subset + optional env freeze + full offline run
bash scripts/deploy_local.sh

# Or run demo directly on prepared subset
bash run_demo.sh --input-dir deploy/demo_data --output-dir output/demo --overwrite
```

## Deployment Test Commands
```bash
python scripts/run_segmentation.py --help
python scripts/make_dashboard.py --help
bash run_demo.sh --help
bash scripts/deploy_local.sh --help
```

## Runtime Design
1. `prepare_demo_data.py`
- Creates aligned demo subset from recorded raw data.
- Normalizes frame IDs to `000000...` so segmentation/detection/dashboard stems match.

2. `run_demo.sh`
- Converts raw demo data to PAD and OpenPCDet formats.
- Runs segmentation inference.
- Runs OpenPCDet inference over train + val split to cover all frame IDs.
- Builds dashboard video and frame dump.

3. `deploy_local.sh`
- Orchestrates Stage 11 artifact prep.
- Optionally exports conda/pip lock files.
- Optionally runs full offline demo.

## Docker Notes
- `Dockerfile` + `docker-compose.yml` provide a containerized dashboard/runtime helper path.
- Full inference deployment remains host-conda based (`run_demo.sh`, `deploy_local.sh`).
- Compose service name: `carla-perception-lab-dashboard`.

## Constraints and Safety
- Local offline simulator-data inference only.
- No real vehicle/road deployment claims.
- Sequential execution to reduce VRAM pressure on 8GB GPUs.
