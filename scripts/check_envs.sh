#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  scripts/check_envs.sh [--repos-dir PATH] [--strict]
  scripts/check_envs.sh --help

Options:
  --repos-dir PATH   Upstream repos directory (default: ../repos)
  --strict           Return non-zero if any check fails
  --help             Show this help message

Checks:
  1) Host prerequisites: conda, nvidia-smi, docker
  2) Conda env existence: carla-client, pad, pcdet
  3) Python import checks inside each env
  4) CARLA docker image availability: carlasim/carla:0.10.0
EOF
}

REPOS_DIR="../repos"
STRICT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repos-dir)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --repos-dir" >&2; exit 2; }
      REPOS_DIR="$1"
      ;;
    --strict)
      STRICT=1
      ;;
    --help|-h)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      show_help
      exit 2
      ;;
  esac
  shift
done

PASS=0
WARN=0
FAIL=0

ok()   { PASS=$((PASS + 1)); echo "[PASS] $*"; }
warn() { WARN=$((WARN + 1)); echo "[WARN] $*"; }
fail() { FAIL=$((FAIL + 1)); echo "[FAIL] $*"; }

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

check_cmd() {
  local cmd="$1"
  if have_cmd "$cmd"; then
    ok "command available: $cmd"
  else
    fail "command missing: $cmd"
  fi
}

check_env_exists() {
  local env_name="$1"
  if conda env list | awk '{print $1}' | grep -Fxq "$env_name"; then
    ok "conda env exists: $env_name"
    return 0
  fi
  fail "conda env missing: $env_name"
  return 1
}

check_imports() {
  local env_name="$1"
  local code="$2"
  local desc="$3"

  if conda run -n "$env_name" python -c "$code" >/dev/null 2>&1; then
    ok "$env_name imports OK: $desc"
  else
    fail "$env_name import check failed: $desc"
  fi
}

echo "== Stage 01 Environment Verification =="
echo "repos dir: $REPOS_DIR"

check_cmd conda
check_cmd nvidia-smi
check_cmd docker

if have_cmd nvidia-smi; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/[INFO] GPU /'
fi

for env_name in carla-client pad pcdet; do
  check_env_exists "$env_name"
done

if conda env list | awk '{print $1}' | grep -Fxq "carla-client"; then
  check_imports "carla-client" \
    "import pygame, numpy, cv2, psutil, requests; print('ok')" \
    "pygame numpy cv2 psutil requests"
  check_imports "carla-client" \
    "import carla; print(carla.__file__)" \
    "carla"
fi

if conda env list | awk '{print $1}' | grep -Fxq "pad"; then
  check_imports "pad" \
    "import torch, mmcv, timm, scipy, sklearn; print(torch.__version__)" \
    "torch mmcv timm scipy sklearn"
  if conda run -n "pad" python -c "import torch; assert torch.cuda.is_available(); print('cuda ok')" >/dev/null 2>&1; then
    ok "pad cuda available"
  else
    warn "pad cuda not available"
  fi
fi

if conda env list | awk '{print $1}' | grep -Fxq "pcdet"; then
  check_imports "pcdet" \
    "import torch, spconv, pcdet; print(torch.__version__)" \
    "torch spconv pcdet"
  if conda run -n "pcdet" python -c "import torch; assert torch.cuda.is_available(); print('cuda ok')" >/dev/null 2>&1; then
    ok "pcdet cuda available"
  else
    warn "pcdet cuda not available"
  fi
fi

if docker image inspect carlasim/carla:0.10.0 >/dev/null 2>&1; then
  ok "docker image present: carlasim/carla:0.10.0"
else
  warn "docker image missing: carlasim/carla:0.10.0"
fi

if [[ -d "$REPOS_DIR" ]]; then
  ok "repos dir exists: $REPOS_DIR"
else
  fail "repos dir missing: $REPOS_DIR"
fi

echo
echo "Summary: PASS=$PASS WARN=$WARN FAIL=$FAIL"

if [[ $STRICT -eq 1 && ( $FAIL -gt 0 || $WARN -gt 0 ) ]]; then
  exit 1
fi

if [[ $FAIL -gt 0 ]]; then
  exit 1
fi
