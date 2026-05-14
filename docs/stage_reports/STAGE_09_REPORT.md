# STAGE 09 — Dashboard Video Report

> Date: 2026-05-13  
> Status: COMPLETE (CANONICAL INPUTS + STABLE OUTPUTS)

## Scope

Implemented and executed Stage 09 dashboard generation on canonical aligned inputs prepared in Stage 03-08 hardening.

## Implemented Artifacts

- `scripts/dashboard.py`
  - strict stem alignment gate by default (`rgb == seg == det == metadata`)
  - composes dashboard canvas with:
    - RGB + segmentation overlay
    - projected 3D detection boxes on camera view
    - LiDAR BEV with detection boxes
    - info panel (frame index, timestamp, class counts, target FPS)
  - supports modes: `frames`, `video`, `both`
  - writes run report JSON

- Outputs:
  - `output/dashboard/frames/*.png` (240 frames)
  - `output/dashboard/dashboard_video.mp4`
  - `output/dashboard/dashboard_report.json`
  - `output/dashboard/demo.gif`
  - `docs/dashboard_screenshot.png`

## Command Executed

```bash
python scripts/dashboard.py \
  --mode both \
  --overwrite \
  --fps 10 \
  --report-json output/dashboard/dashboard_report.json
```

## Runtime Inputs (resolved)

- RGB: `data/raw/recording_001/rgb`
- Segmentation predictions: `output/segmentation/predictions`
- Detection predictions: `output/detection_3d/predictions`
- LiDAR: `data/raw/recording_001/lidar`
- Metadata: `data/raw/recording_001/metadata`
- Calibration: `data/raw/recording_001/calib/sensors.json`

(These aliases were pre-bound to canonical Stage08-ready artifacts.)

## Results

From `output/dashboard/dashboard_report.json`:

- selected aligned frames: `240`
- rendered frames: `240`
- target FPS: `10.0`
- render throughput (CPU): `35.09 FPS`
- total detections rendered: `348`
- projected 3D boxes drawn: `293`
- frames containing projected boxes: `153`

Generated media validation:

- `output/dashboard/dashboard_video.mp4`
  - frame count: `240`
  - resolution: `1280x720`
  - FPS: `10.0`
  - size: `8,897,235 bytes`
- `output/dashboard/frames`: `240` PNG images (`1280x720`)
- `output/dashboard/demo.gif`: `2,695,043 bytes` (< 10 MB)

## Notes / Fallbacks

- `ffmpeg` is not installed on this host (`command not found`), so GIF was generated via Pillow from rendered PNG frames.
- OpenCV attempted hardware H264 codec first and fell back automatically to a working codec (`mp4v`) without impacting output validity.

## Exit Criteria Status

- [x] Dashboard renders every aligned frame without runtime errors
- [x] Segmentation overlay visible in main view
- [x] 3D detections shown in BEV and projected camera overlay
- [x] Info panel updates per-frame metadata and counts
- [x] MP4 output valid and playable
- [x] Demo GIF generated for documentation

## Next Action

Proceed to Stage 10 benchmark/evaluation using the same canonical alias set.
