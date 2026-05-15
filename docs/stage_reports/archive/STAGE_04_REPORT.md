# STAGE 04 — PAD Smoke Test Report

> Date: 2026-05-13  
> Status: COMPLETE (WITH ENV FALLBACKS)

## Scope

Executed `final_plan/STAGE_04_PAD_SMOKE_TEST.md` only:
- verify PAD runtime on `pad` env
- run ERFNet Cityscapes inference on sample images
- save visual output screenshot for docs
- collect model size, FPS, and VRAM usage
- keep all new project code under `carla-perception-lab`

## Implemented Artifacts

- `scripts/pad_smoke_test.py`
  - supports `--help`
  - reproducible stage-04 runner for:
    - asset download (checkpoint, encoder weight, test images)
    - ERFNet inference (`tools/vis/seg_img_dir.py`)
    - profiling (`tools/profiling.py`) at configurable resolutions
    - VRAM peak monitoring via `nvidia-smi`
    - summary export to JSON
- `docs/pad_smoke_test.png`
  - screenshot artifact copied from inference result
- `docs/stage_reports/STAGE_04_REPORT.md`

## Runtime Environment Fixes (Fallbacks Applied)

To make upstream PAD scripts runnable on this host/env combination, the following compatibility fixes were required in the **`pad` conda env**:

1. `protobuf` mismatch with TensorBoard-generated proto files
   - fix: `protobuf==3.20.3`
2. `numpy` deprecation (`np.float`) breaking old `scikit-learn` path
   - fix: `numpy==1.23.5`
3. `Pillow>=10` removed `Image.LINEAR` used by upstream transforms
   - fix: `Pillow<10` (installed `Pillow==9.5.0`)
4. missing tool for Google Drive downloads
   - fix: install `gdown`

These are environment-level fallbacks only; no upstream source files were modified.

## Validation Performed

### CLI / Script Checks

1. `conda run -n pad python -m py_compile scripts/pad_smoke_test.py`
   - PASS
2. `conda run -n pad python scripts/pad_smoke_test.py --help`
   - PASS
3. `conda run -n pad python scripts/pad_smoke_test.py --dry-run --skip-download --skip-profiling --mixed-precision`
   - PASS

### Asset Download Checks

4. Download ERFNet checkpoint:
   - `data/checkpoints/erfnet_cityscapes_512x1024_20200918.pt`
5. Download ERFNet encoder pretrain:
   - `data/checkpoints/erfnet_encoder_pretrained.pth.tar`
6. Download and unzip PAD test images:
   - `data/processed/PAD_test_images.zip`
   - `data/processed/pad_test_images/...`

### Inference + Profiling Execution

7. End-to-end stage script:
   - `conda run -n pad python scripts/pad_smoke_test.py --skip-download --mixed-precision`
   - PASS
8. Output checks:
   - generated segmentation images: `11`
   - screenshot exported: `docs/pad_smoke_test.png`
9. Visual/output sanity:
   - input shape: `(1024, 2048, 3)`
   - output shape: `(1024, 2048, 3)`
   - mean absolute pixel difference (input vs output): `45.17` (overlay is non-trivial)

## Measured Results

From `output/pad_smoke_test/stage04_summary.json`:

### Inference (Cityscapes munster set, 11 images)

- elapsed: `13.873s`
- baseline VRAM: `651 MiB`
- peak VRAM: `2418 MiB`
- VRAM delta: `1767 MiB`

### Profiling

- `512x1024`
  - GPU FPS: `56.61`
  - FLOPs: `60.11 G`
  - Params: `2.07 M`
  - peak VRAM: `2288 MiB` (delta `1641 MiB`)
- `720x1280`
  - GPU FPS: `30.59`
  - FLOPs: `105.66 G`
  - Params: `2.07 M`
  - peak VRAM: `2390 MiB` (delta `1741 MiB`)

## Exit Criteria Status

- [x] ERFNet inference produces segmentation outputs on Cityscapes sample
- [x] Result screenshot saved to `docs/pad_smoke_test.png`
- [x] FPS measured and documented
- [x] VRAM usage measured and `< 3 GB` peak on this run
- [x] No OOM during smoke inference/profiling
- [x] Reproducible script with `--help` provided

## Assumptions

1. Stage 04 allows environment dependency pinning in `pad` env when upstream compatibility issues block execution.
2. Upstream PAD repo remains read-only for source code; only runtime commands are executed against it.

## Known Issues

1. Upstream PAD scripts emit non-blocking warnings:
   - `mmcv` deprecation notice
   - LaneATT NMS compile warning
2. `seg_img_dir.py` requires ERFNet encoder pretrain path to exist even in inference usage with checkpoint; stage script handles this by providing `model.pretrained_weights=...` via `--cfg-options`.

## Next Recommended Action

Proceed to `STAGE_05_OPENPCDET_SMOKE_TEST.md`.
