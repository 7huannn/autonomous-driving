# STAGE 11 — Lightweight Deployment Report

> Date: 2026-05-13  
> Status: COMPLETE (LOCAL OFFLINE DEPLOYMENT PATH ADDED)

## Scope
Implemented a reproducible local offline deployment/inference workflow that does not require an active CARLA server.

## Implemented Artifacts
- `Dockerfile`
- `docker-compose.yml`
- `scripts/deploy_local.sh`
- `run_demo.sh`
- `scripts/prepare_demo_data.py`
- `scripts/make_dashboard.py`
- `docs/deployment.md`
- `deploy/DEPLOY_README.md`
- `requirements/deploy.txt`

## Key Design Choices
1. Demo raw subset normalization:
- `prepare_demo_data.py` copies a compact frame subset and rewrites stem IDs to `000000...`.
- This guarantees stem parity across raw data, PAD predictions, and OpenPCDet predictions in dashboard rendering.

2. Sequential low-VRAM execution:
- `run_demo.sh` runs conversion -> segmentation -> detection -> dashboard in order.
- Avoids concurrent GPU load on 8GB VRAM hardware.

3. Detection split coverage:
- OpenPCDet inference runs on both `train` and `val` splits to produce predictions for all frame IDs in demo subset.

## Deployment Commands
```bash
# prepare deploy assets + run demo
bash scripts/deploy_local.sh

# run only offline demo
bash run_demo.sh --input-dir deploy/demo_data --output-dir output/demo --overwrite
```

## Exit Criteria Status
- [x] Deployment artifacts created (`Dockerfile`, `docker-compose.yml`, deploy scripts/docs)
- [x] CLI deployment entrypoint provided (`run_demo.sh`)
- [x] Stage test target supported (`python scripts/make_dashboard.py --help`)
- [x] Deployment documentation added (`docs/deployment.md`, `deploy/DEPLOY_README.md`)
- [x] Scope kept simulator/offline only (no real-road claims)

## Notes
- Environment lock files are exported by `scripts/deploy_local.sh` when conda envs are available.
- Container workflow is provided as optional dashboard/runtime helper path (`docker compose`).
- Full offline inference path is executed via host conda envs (`run_demo.sh`).
