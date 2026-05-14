# STAGE 10 — Benchmark Report

> Date: 2026-05-13  
> Status: COMPLETE (BENCHMARK + REPORT GENERATED)

## Scope

Executed Stage 10 benchmark/evaluation on canonical aligned artifacts from Stage 03-09.

## Implemented Artifacts

- `scripts/benchmark.py`
  - modes: `benchmark`, `vram`, `report`
  - supports both models: `segmentation`, `detection`
  - warmup + repeated runs
  - wall latency/FPS + model forward latency/FPS
  - peak VRAM polling via `nvidia-smi`
  - checkpoint size and parameter count extraction
  - markdown report generation

- Outputs:
  - `output/benchmark/seg_benchmark.json`
  - `output/benchmark/det_benchmark.json`
  - `output/benchmark/benchmark_results.json`
  - `output/benchmark/benchmark_report.md`

## Commands Executed

```bash
# refresh accuracy jsons used by benchmark
conda run -n pad python scripts/run_segmentation.py \
  --mode evaluate \
  --gt-dir data/processed/pad_format_stage08_canonical_mix_val/masks \
  --pred-dir output/segmentation/predictions \
  --eval-output output/segmentation_canonical_mix_val/eval_results.json

conda run -n pcdet python scripts/run_lidar_det.py \
  --mode evaluate \
  --cfg-file configs/carla_lidar_probe.yaml \
  --checkpoint ../repos/OpenPCDet/output/.../checkpoint_epoch_5.pth \
  --dataset-dir data/processed/pcdet_format_stage08_canonical_mix \
  --split val --batch-size 1 \
  --pred-dir output/detection_3d/predictions \
  --eval-output output/detection_3d_canonical_mix/eval_results.json

# stage 10 benchmark
python scripts/benchmark.py \
  --mode benchmark --model segmentation \
  --num-frames 100 --warmup-frames 10 --repeats 3 \
  --output output/benchmark/seg_benchmark.json

python scripts/benchmark.py \
  --mode benchmark --model detection \
  --num-frames 100 --warmup-frames 10 --repeats 3 \
  --output output/benchmark/det_benchmark.json

python scripts/benchmark.py \
  --mode report \
  --seg-results output/benchmark/seg_benchmark.json \
  --det-results output/benchmark/det_benchmark.json \
  --report-output output/benchmark/benchmark_report.md \
  --output output/benchmark/benchmark_results.json
```

## Key Results

From `output/benchmark/benchmark_report.md`:

- Segmentation (ERFNet)
  - params: `2.073M`
  - checkpoint: `8.03 MB`
  - wall FPS: `9.38 ± 0.12`
  - wall latency: `106.58 ± 1.38 ms/frame`
  - model forward FPS: `22.11 ± 0.46`
  - model forward latency: `45.24 ± 0.93 ms`
  - peak VRAM: `2273 MiB`
  - mIoU: `0.0573`

- Detection (PointPillar, fine-tuned ckpt)
  - params: `4.841M`
  - checkpoint: `55.40 MB`
  - wall FPS: `8.07 ± 0.05`
  - wall latency: `123.96 ± 0.79 ms/frame`
  - model forward FPS: `28.07 ± 0.22`
  - model forward latency: `35.63 ± 0.28 ms`
  - peak VRAM: `2444 MiB`
  - mAP (custom BEV): `0.0030`

## Exit Criteria Status

- [x] Real benchmark numbers generated for both pipelines
- [x] Repeatability check included (`repeats=3`, std reported)
- [x] Peak VRAM captured during runtime
- [x] Accuracy metrics included from canonical eval artifacts
- [x] Markdown report generated successfully
- [x] Both models individually below 8GB VRAM budget on this host

## Notes

- Wall FPS/latency includes end-to-end wrapper overhead (I/O + preprocessing + inference + write).
- Forward FPS/latency reflects model forward-only timing reported by Stage07/Stage08 wrappers.
