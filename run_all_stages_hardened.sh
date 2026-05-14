#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CARLA_ENV="carla-client"
PAD_ENV="pad"
PCDET_ENV="pcdet"
SKIP_STAGE03=0
MAIN_FRAMES=1000
AUX_FRAMES=200
TM_PORT=10000
SEG_FPS_TARGET=10

RAW_MAIN_DIR="data/raw/stage03_mine_main_1000"
RAW_AUX_DIR="data/raw/stage03_mine_aux_200_traffic_far"
PAD_OUT_DIR="data/processed/pad_format_stage08_canonical_mix"
PAD_VAL_DIR="data/processed/pad_format_stage08_canonical_mix_val"
PCDET_OUT_DIR="data/processed/pcdet_format_stage08_canonical_mix"

SEG_FULL_DIR="output/segmentation_canonical_mix"
SEG_VAL_DIR="output/segmentation_canonical_mix_val"
DET_DIR="output/detection_3d_canonical_mix"

STAGE_VALIDATION_DIR="output/stage_validation"
DASHBOARD_DIR="output/dashboard"

DET_CHECKPOINT_FT="../repos/OpenPCDet/output/media/thuan/Workspace/Hoc_Ki_Cuoi/autonomous-driving/carla-perception-lab/configs/carla_lidar_probe/carla_probe_finetune/ckpt/checkpoint_epoch_5.pth"
DET_CHECKPOINT_BASE="data/checkpoints/pointpillar_7728.pth"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_cmd() {
  log "RUN: $*"
  "$@"
}

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --skip-stage03           Skip Stage 03 recording and reuse existing raw datasets
  --main-frames N          Frame target for main Mine recording (default: ${MAIN_FRAMES})
  --aux-frames N           Frame target for auxiliary Mine recording (default: ${AUX_FRAMES})
  --tm-port N              Traffic Manager port (default: ${TM_PORT})
  --carla-env NAME         Conda env for CARLA recorder (default: ${CARLA_ENV})
  --pad-env NAME           Conda env for PAD segmentation (default: ${PAD_ENV})
  --pcdet-env NAME         Conda env for OpenPCDet detection (default: ${PCDET_ENV})
  --seg-fps N              Dashboard target FPS (default: ${SEG_FPS_TARGET})
  -h, --help               Show this help

This script rebuilds hardened Stage 03->09 artifacts end-to-end:
  1) record stable Mine raw datasets (main+aux)
  2) convert PAD/PCDet datasets with strict overwrite + manifests
  3) run Stage07 segmentation (full + val-aligned subset)
  4) run Stage08 LiDAR detection (val split)
  5) run readiness hard-gate
  6) prepare Stage09 canonical aliases
  7) render Stage09 dashboard video/frames
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-stage03)
      SKIP_STAGE03=1
      shift
      ;;
    --main-frames)
      MAIN_FRAMES="$2"
      shift 2
      ;;
    --aux-frames)
      AUX_FRAMES="$2"
      shift 2
      ;;
    --tm-port)
      TM_PORT="$2"
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
    --seg-fps)
      SEG_FPS_TARGET="$2"
      shift 2
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

if [[ -f "$DET_CHECKPOINT_FT" ]]; then
  DET_CHECKPOINT="$DET_CHECKPOINT_FT"
else
  DET_CHECKPOINT="$DET_CHECKPOINT_BASE"
fi

mkdir -p "$STAGE_VALIDATION_DIR" "$DASHBOARD_DIR"

if [[ "$SKIP_STAGE03" -eq 0 ]]; then
  log "Stage03: recording stable Mine main dataset"
  run_cmd conda run -n "$CARLA_ENV" python scripts/carla_recorder.py \
    --config configs/sensor_config.yaml \
    --map Mine_01 \
    --output-dir "$RAW_MAIN_DIR" \
    --num-frames "$MAIN_FRAMES" \
    --timeout 20 \
    --tm-port "$TM_PORT" \
    --traffic-vehicles 0 \
    --traffic-walkers 0 \
    --prefer-cyclist-vehicles 0 \
    --overwrite

  log "Stage03: recording auxiliary Mine dataset for detection labels"
  run_cmd conda run -n "$CARLA_ENV" python scripts/carla_recorder.py \
    --config configs/sensor_config.yaml \
    --map Mine_01 \
    --output-dir "$RAW_AUX_DIR" \
    --num-frames "$AUX_FRAMES" \
    --timeout 20 \
    --tm-port "$TM_PORT" \
    --traffic-vehicles 15 \
    --traffic-walkers 20 \
    --prefer-cyclist-vehicles 5 \
    --actor-label-max-distance 200 \
    --overwrite
else
  log "Stage03 skipped by flag; existing datasets will be reused"
fi

log "Stage06: converting to PAD canonical mix"
run_cmd python scripts/convert_to_pad.py \
  --input-dirs "$RAW_MAIN_DIR" "$RAW_AUX_DIR" \
  --output-dir "$PAD_OUT_DIR" \
  --num-frames 0 \
  --shuffle-split \
  --seed 42 \
  --overwrite \
  --mapping-profile carla_010 \
  --summary-json "$STAGE_VALIDATION_DIR/stage06_pad_summary_canonical_mix.json" \
  --manifest-json "$STAGE_VALIDATION_DIR/stage06_pad_manifest_canonical_mix.json"

log "Stage06: converting to PCDet canonical mix"
run_cmd conda run -n "$PCDET_ENV" python scripts/convert_to_pcdet.py \
  --input-dirs "$RAW_MAIN_DIR" "$RAW_AUX_DIR" \
  --output-dir "$PCDET_OUT_DIR" \
  --num-frames 0 \
  --shuffle-split \
  --seed 42 \
  --overwrite \
  --sensor-config configs/sensor_config.yaml \
  --summary-json "$STAGE_VALIDATION_DIR/stage06_pcdet_summary_canonical_mix.json" \
  --manifest-json "$STAGE_VALIDATION_DIR/stage06_pcdet_manifest_canonical_mix.json" \
  --generate-custom-infos

log "Stage07 prep: building PAD val subset aligned to PCDet val split"
run_cmd python - <<'PY'
import shutil
from pathlib import Path

split_file = Path("data/processed/pcdet_format_stage08_canonical_mix/ImageSets/val.txt")
src_root = Path("data/processed/pad_format_stage08_canonical_mix")
dst_root = Path("data/processed/pad_format_stage08_canonical_mix_val")

ids = [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]
if not ids:
    raise RuntimeError(f"Empty split file: {split_file}")

if dst_root.exists():
    shutil.rmtree(dst_root)
(dst_root / "images").mkdir(parents=True, exist_ok=False)
(dst_root / "masks").mkdir(parents=True, exist_ok=False)

for stem in ids:
    for modal in ("images", "masks"):
        src = src_root / modal / f"{stem}.png"
        if not src.exists():
            raise FileNotFoundError(f"Missing source file: {src}")
        dst = dst_root / modal / src.name
        dst.symlink_to(src.resolve())

print({"val_count": len(ids), "dst": str(dst_root.resolve())})
PY

log "Stage07: running segmentation on full canonical mix"
run_cmd conda run -n "$PAD_ENV" python scripts/run_segmentation.py \
  --mode infer_and_evaluate \
  --image-dir "$PAD_OUT_DIR/images" \
  --gt-dir "$PAD_OUT_DIR/masks" \
  --pred-dir "$SEG_FULL_DIR/predictions" \
  --overlay-dir "$SEG_FULL_DIR/overlays" \
  --eval-output "$SEG_FULL_DIR/eval_results.json" \
  --batch-size 1 \
  --mixed-precision \
  --save-overlays \
  --overwrite

log "Stage07: running segmentation on val-aligned subset"
run_cmd conda run -n "$PAD_ENV" python scripts/run_segmentation.py \
  --mode infer_and_evaluate \
  --image-dir "$PAD_VAL_DIR/images" \
  --gt-dir "$PAD_VAL_DIR/masks" \
  --pred-dir "$SEG_VAL_DIR/predictions" \
  --overlay-dir "$SEG_VAL_DIR/overlays" \
  --eval-output "$SEG_VAL_DIR/eval_results.json" \
  --batch-size 1 \
  --mixed-precision \
  --save-overlays \
  --overwrite

log "Stage08: running LiDAR detection on canonical mix val split"
run_cmd conda run -n "$PCDET_ENV" python scripts/run_lidar_det.py \
  --mode infer_and_evaluate \
  --cfg-file configs/carla_lidar_probe.yaml \
  --checkpoint "$DET_CHECKPOINT" \
  --dataset-dir "$PCDET_OUT_DIR" \
  --split val \
  --batch-size 1 \
  --pred-dir "$DET_DIR/predictions" \
  --eval-output "$DET_DIR/eval_results.json" \
  --overwrite \
  --score-thresh 0.1

log "Stage08 gate: validating readiness hard checks"
run_cmd python scripts/validate_stage_readiness.py \
  --raw-dir "$RAW_MAIN_DIR" \
  --pad-summary "$STAGE_VALIDATION_DIR/stage06_pad_summary_canonical_mix.json" \
  --pcdet-summary "$STAGE_VALIDATION_DIR/stage06_pcdet_summary_canonical_mix.json" \
  --seg-pred-dir "$SEG_VAL_DIR/predictions" \
  --det-pred-dir "$DET_DIR/predictions" \
  --report-json "$STAGE_VALIDATION_DIR/stage08_readiness_report.json"

log "Stage09 prep: creating canonical raw subset + aliasing default paths"
run_cmd python scripts/prepare_stage09_inputs.py \
  --overwrite \
  --apply-default-aliases \
  --report-json "$STAGE_VALIDATION_DIR/stage09_preparation_report.json"

log "Stage09: rendering dashboard video + frames"
run_cmd python scripts/dashboard.py \
  --mode both \
  --fps "$SEG_FPS_TARGET" \
  --overwrite \
  --report-json "$DASHBOARD_DIR/dashboard_report.json"

log "DONE: Hardened Stage03->09 pipeline completed"
log "Key outputs:"
log "  - $STAGE_VALIDATION_DIR/stage08_readiness_report.json"
log "  - $STAGE_VALIDATION_DIR/stage09_preparation_report.json"
log "  - $DASHBOARD_DIR/dashboard_video.mp4"
log "  - $DASHBOARD_DIR/dashboard_report.json"
