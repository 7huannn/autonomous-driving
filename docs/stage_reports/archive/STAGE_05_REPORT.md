# STAGE 05 — OpenPCDet Smoke Test Report

> Date: 2026-05-13  
> Status: COMPLETE (WITH ENV/IMPORT FALLBACKS)

## Scope

Executed `final_plan/STAGE_05_OPENPCDET_SMOKE_TEST.md` only:
- verify OpenPCDet runtime in `pcdet` env
- verify required CUDA extension imports
- run PointPillar inference on a sample LiDAR `.npy`
- produce smoke-test visualization artifact
- document VRAM and latency/FPS metrics

## Implemented Artifacts

- `scripts/pcdet_smoke_test.py`
  - supports `--help`
  - reproducible runner for stage 05:
    - checkpoint download via `gdown`
    - synthetic LiDAR sample generation
    - CUDA extension verification
    - headless PointPillar inference
    - BEV visualization export to PNG
    - metrics summary to JSON
- `docs/pcdet_smoke_test.png`
- `docs/stage_reports/archive/STAGE_05_REPORT.md`

## Validation Performed

1. `conda run -n pcdet python -m py_compile scripts/pcdet_smoke_test.py`
   - PASS
2. `conda run -n pcdet python scripts/pcdet_smoke_test.py --help`
   - PASS
3. `conda run -n pcdet python scripts/pcdet_smoke_test.py`
   - initial run encountered import-path/runtime blockers (documented below)
4. `conda run -n pcdet python scripts/pcdet_smoke_test.py --skip-download`
   - PASS (after fallback handling)
5. Output checks:
   - summary JSON present: `output/pcdet_smoke_test/stage05_summary.json`
   - screenshot present: `docs/pcdet_smoke_test.png`

## Result Metrics

From `output/pcdet_smoke_test/stage05_summary.json`:

- Input points used by model: `29804`
- Predicted boxes drawn: `60`
- Max score: `0.3092`
- Mean score: `0.1712`
- VRAM baseline: `665 MiB`
- VRAM peak: `1281 MiB`
- VRAM delta: `616 MiB` (< 3 GB guardrail)

Timing:
- Total script inference stage (includes setup/load): `8.2872 s`
- Model setup/load: `1.2012 s`
- First forward path time: `1.57 s` (includes first-pass overhead)
- Steady-state benchmark (warmup=2, repeats=5):
  - avg forward latency: `25.26 ms`
  - forward FPS: `39.60`

## Exit Criteria Status

- [x] OpenPCDet CUDA extensions load successfully
- [x] PointPillar checkpoint loads
- [x] Demo-style inference runs on sample `.npy`
- [x] Predictions produced (non-empty boxes)
- [x] VRAM measured and `< 3 GB`
- [x] Inference latency measured; steady-state `< 100 ms/frame`
- [x] Result screenshot saved

## Fallbacks / Known Issues

1. Upstream OpenPCDet snapshot uses repo-scoped imports and includes optional modules that break this env even for PointPillar smoke test:
   - `Argo2Dataset` chain requires `av2` with Python typing incompatible on Python 3.8
   - DSVT module path in this snapshot triggers invalid relative import at import-time
2. Fallback used in project-local script (no upstream file modifications):
   - inject stub module for `repos.OpenPCDet.pcdet.datasets.argo2.argo2_dataset`
   - inject stub module for `repos.OpenPCDet.pcdet.models.backbones_3d.dsvt`
   - this keeps PointPillar path runnable while avoiding unrelated optional branches
3. `torch.meshgrid` warning appears during runtime (non-blocking)

## Assumptions

1. Stage 05 acceptance allows project-local runtime shims for optional upstream modules, as long as PointPillar inference path is validated and upstream repo source is not modified.
2. Synthetic point cloud is acceptable for smoke validation; prediction quality is not interpreted as accuracy benchmark.

## Next Recommended Action

Proceed to `STAGE_06_DATASET_INTERFACE.md`.
