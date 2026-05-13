#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  scripts/setup_envs_stage01.sh [options]
  scripts/setup_envs_stage01.sh --help

Options:
  --repos-dir PATH         Upstream repos directory (default: ../repos)
  --skip-docker            Skip pulling carlasim/carla:0.10.0
  --skip-carla-client      Skip creating/installing carla-client env
  --skip-pad               Skip creating/installing pad env
  --skip-pcdet             Skip creating/installing pcdet env
  --spconv-package NAME    spconv package for pcdet env (default: spconv-cu116)
  --dry-run                Print commands only
  --help                   Show this help message

Notes:
  - This script follows final_plan/STAGE_01_ENV_SETUP.md.
  - It does not modify files under ../repos.
  - CARLA 0.10.0 Python wheel must come from CARLA package or Docker extraction.
EOF
}

REPOS_DIR="../repos"
SKIP_DOCKER=0
SKIP_CARLA_CLIENT=0
SKIP_PAD=0
SKIP_PCDET=0
SPCONV_PACKAGE="spconv-cu116"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repos-dir)
      shift
      REPOS_DIR="${1:-}"
      [[ -n "$REPOS_DIR" ]] || { echo "Missing value for --repos-dir" >&2; exit 2; }
      ;;
    --skip-docker) SKIP_DOCKER=1 ;;
    --skip-carla-client) SKIP_CARLA_CLIENT=1 ;;
    --skip-pad) SKIP_PAD=1 ;;
    --skip-pcdet) SKIP_PCDET=1 ;;
    --spconv-package)
      shift
      SPCONV_PACKAGE="${1:-}"
      [[ -n "$SPCONV_PACKAGE" ]] || { echo "Missing value for --spconv-package" >&2; exit 2; }
      ;;
    --dry-run) DRY_RUN=1 ;;
    --help|-h) show_help; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; show_help; exit 2 ;;
  esac
  shift
done

run_cmd() {
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] $*"
  else
    echo "[RUN] $*"
    eval "$*"
  fi
}

if [[ $SKIP_DOCKER -eq 0 ]]; then
  run_cmd "docker pull carlasim/carla:0.10.0"
fi

if [[ $SKIP_CARLA_CLIENT -eq 0 ]]; then
  run_cmd "conda create -n carla-client python=3.10 -y"
  run_cmd "conda run -n carla-client pip install pygame numpy opencv-python open3d psutil requests"
  echo "[INFO] CARLA wheel install step is manual (source wheel from CARLA package or Docker)."
fi

if [[ $SKIP_PAD -eq 0 ]]; then
  run_cmd "conda create -n pad python=3.8 -y"
  run_cmd "conda install -n pad -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge"
  run_cmd "conda run -n pad pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html"
  run_cmd "conda run -n pad pip install -r \"$REPOS_DIR/pytorch-auto-drive/requirements.txt\""
fi

if [[ $SKIP_PCDET -eq 0 ]]; then
  run_cmd "conda create -n pcdet python=3.8 -y"
  run_cmd "conda install -n pcdet -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia"
  run_cmd "conda run -n pcdet pip install \"$SPCONV_PACKAGE\""
  run_cmd "conda run -n pcdet pip install -r \"$REPOS_DIR/OpenPCDet/requirements.txt\""
  run_cmd "conda run -n pcdet pip install open3d"
  run_cmd "cd \"$REPOS_DIR/OpenPCDet\" && conda run -n pcdet python setup.py develop"
fi

echo
echo "Setup script finished. Run scripts/check_envs.sh to verify."
