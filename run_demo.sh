#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CARLA_ENV="carla-client"
PAD_ENV="pad"
PCDET_ENV="pcdet"

INPUT_DIR="deploy/demo_data"
OUTPUT_DIR="output/demo"
OVERWRITE=0

usage() {
  cat <<EOF
Usage: $0 [options]

Run offline perception demo without CARLA runtime.

Options:
  --input-dir PATH       Raw demo input directory (default: ${INPUT_DIR})
  --output-dir PATH      Output root (default: ${OUTPUT_DIR})
  --carla-env NAME       Conda env for conversion/dashboard (default: ${CARLA_ENV})
  --pad-env NAME         Conda env for segmentation (default: ${PAD_ENV})
  --pcdet-env NAME       Conda env for detection (default: ${PCDET_ENV})
  --overwrite            Replace existing output directory
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --carla-env)
      CARLA_ENV="$2"
      shift 2
      ;;
    --pad-env)
      PAD_ENV="$2"
      shift 2
      ;;
    --pcdet-env)
      PCDET_ENV="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda command not found." >&2
  exit 1
fi

INPUT_DIR="$(realpath "$INPUT_DIR")"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Error: input dir not found: $INPUT_DIR" >&2
  exit 1
fi

if [[ $OVERWRITE -eq 1 && -d "$OUTPUT_DIR" ]]; then
  rm -rf "$OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR"

PAD_DATASET_DIR="$OUTPUT_DIR/pad_format"
PCDET_DATASET_DIR="$OUTPUT_DIR/pcdet_format"
SEG_DIR="$OUTPUT_DIR/segmentation"
DET_DIR="$OUTPUT_DIR/detection_3d"
DASHBOARD_DIR="$OUTPUT_DIR/dashboard"

echo "=== CARLA Perception Lab: Offline Demo ==="
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"

echo "[1/5] Convert raw demo data to PAD format"
conda run -n "$CARLA_ENV" python scripts/convert_to_pad.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$PAD_DATASET_DIR" \
  --train-ratio 0.8 \
  --overwrite

echo "[2/5] Convert raw demo data to OpenPCDet format"
conda run -n "$PCDET_ENV" python scripts/convert_to_pcdet.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$PCDET_DATASET_DIR" \
  --train-ratio 0.8 \
  --overwrite \
  --sensor-config configs/sensor_config.yaml \
  --generate-custom-infos

echo "[3/5] Segmentation inference (ERFNet)"
conda run -n "$PAD_ENV" python scripts/run_segmentation.py \
  --mode infer \
  --image-dir "$PAD_DATASET_DIR/images" \
  --pred-dir "$SEG_DIR/predictions" \
  --overlay-dir "$SEG_DIR/overlays" \
  --save-overlays \
  --batch-size 1 \
  --overwrite

echo "[4/5] 3D detection inference (PointPillar)"
conda run -n "$PCDET_ENV" python scripts/run_lidar_det.py \
  --mode infer \
  --cfg-file configs/carla_lidar_probe.yaml \
  --checkpoint data/checkpoints/pointpillar_7728.pth \
  --dataset-dir "$PCDET_DATASET_DIR" \
  --split train \
  --batch-size 1 \
  --pred-dir "$DET_DIR/predictions" \
  --overwrite

conda run -n "$PCDET_ENV" python scripts/run_lidar_det.py \
  --mode infer \
  --cfg-file configs/carla_lidar_probe.yaml \
  --checkpoint data/checkpoints/pointpillar_7728.pth \
  --dataset-dir "$PCDET_DATASET_DIR" \
  --split val \
  --batch-size 1 \
  --pred-dir "$DET_DIR/predictions"

echo "[5/5] Dashboard rendering"
conda run -n "$CARLA_ENV" python scripts/make_dashboard.py \
  --mode both \
  --rgb-dir "$INPUT_DIR/rgb" \
  --seg-dir "$SEG_DIR/predictions" \
  --det-dir "$DET_DIR/predictions" \
  --lidar-dir "$INPUT_DIR/lidar" \
  --metadata-dir "$INPUT_DIR/metadata" \
  --calib-json "$INPUT_DIR/calib/sensors.json" \
  --output "$DASHBOARD_DIR/demo_video.mp4" \
  --output-dir "$DASHBOARD_DIR/frames" \
  --report-json "$DASHBOARD_DIR/dashboard_report.json" \
  --fps 10 \
  --overwrite

echo "=== Done ==="
echo "Dashboard video: $DASHBOARD_DIR/demo_video.mp4"
