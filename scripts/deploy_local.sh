#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CARLA_ENV="carla-client"
PAD_ENV="pad"
PCDET_ENV="pcdet"

SOURCE_RAW="data/raw/recording_001"
DEMO_RAW="deploy/demo_data"
DEMO_FRAMES=100
DEMO_OUTPUT="output/demo"

SKIP_FREEZE=0
SKIP_RUN=0

usage() {
  cat <<EOF
Usage: $0 [options]

Prepare Stage 11 local offline deployment assets.

Options:
  --source-raw PATH       Raw dataset source for demo subset (default: ${SOURCE_RAW})
  --demo-raw PATH         Demo subset output directory (default: ${DEMO_RAW})
  --demo-frames N         Number of demo frames (default: ${DEMO_FRAMES})
  --demo-output PATH      Offline demo output root (default: ${DEMO_OUTPUT})
  --carla-env NAME        Conda env for CARLA client scripts (default: ${CARLA_ENV})
  --pad-env NAME          Conda env for PAD scripts (default: ${PAD_ENV})
  --pcdet-env NAME        Conda env for OpenPCDet scripts (default: ${PCDET_ENV})
  --skip-freeze           Do not export conda/pip environment lock files
  --skip-run              Do not execute run_demo.sh
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-raw)
      SOURCE_RAW="$2"
      shift 2
      ;;
    --demo-raw)
      DEMO_RAW="$2"
      shift 2
      ;;
    --demo-frames)
      DEMO_FRAMES="$2"
      shift 2
      ;;
    --demo-output)
      DEMO_OUTPUT="$2"
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
    --skip-freeze)
      SKIP_FREEZE=1
      shift
      ;;
    --skip-run)
      SKIP_RUN=1
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

mkdir -p deploy

echo "[deploy] prepare demo raw subset"
python scripts/prepare_demo_data.py \
  --input-dir "$SOURCE_RAW" \
  --output-dir "$DEMO_RAW" \
  --num-frames "$DEMO_FRAMES" \
  --overwrite \
  --report-json deploy/demo_data_report.json

if [[ $SKIP_FREEZE -eq 0 ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "[deploy] warning: conda not found; skip env freeze"
  else
    echo "[deploy] export conda env specs"
    conda env export -n "$CARLA_ENV" > deploy/env_${CARLA_ENV}.yml
    conda env export -n "$PAD_ENV" > deploy/env_${PAD_ENV}.yml
    conda env export -n "$PCDET_ENV" > deploy/env_${PCDET_ENV}.yml

    echo "[deploy] export pip freeze snapshots"
    conda run -n "$CARLA_ENV" pip freeze > deploy/requirements_${CARLA_ENV}_frozen.txt
    conda run -n "$PAD_ENV" pip freeze > deploy/requirements_${PAD_ENV}_frozen.txt
    conda run -n "$PCDET_ENV" pip freeze > deploy/requirements_${PCDET_ENV}_frozen.txt
  fi
fi

if [[ $SKIP_RUN -eq 0 ]]; then
  echo "[deploy] run offline demo pipeline"
  bash run_demo.sh \
    --input-dir "$DEMO_RAW" \
    --output-dir "$DEMO_OUTPUT" \
    --carla-env "$CARLA_ENV" \
    --pad-env "$PAD_ENV" \
    --pcdet-env "$PCDET_ENV" \
    --overwrite
else
  echo "[deploy] skip demo run (--skip-run)"
fi

echo "[deploy] done"
