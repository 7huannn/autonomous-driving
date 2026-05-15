# STAGE 01 — Environment Setup Report

> Date: 2026-05-08  
> Status: COMPLETE

## Scope

Executed `final_plan/STAGE_01_ENV_SETUP.md` only:
- Create envs `carla-client`, `pad`, `pcdet`
- Pull CARLA Docker image
- Install dependencies
- Verify with reproducible scripts/CLI

## Implemented Artifacts

- `.gitignore` created for data/output/checkpoints/build artifacts
- `scripts/check_envs.sh` created (supports `--help`)
- `scripts/setup_envs_stage01.sh` created (supports `--help`)

## Environment Results

### 1) `carla-client`
- Python 3.10
- Installed: `pygame numpy opencv-python open3d psutil requests`
- Installed CARLA wheel from Docker extraction:
  - `/tmp/carla_pythonapi_stage01/carla/dist/carla-0.10.0-cp310-cp310-linux_x86_64.whl`
- Verification: `import carla` PASS

### 2) `pad`
- Python 3.8
- Installed:
  - `torch==1.12.1+cu116`, `torchvision==0.13.1+cu116`, `torchaudio==0.12.1+cu116`
  - `mmcv-full==1.7.0`
  - `repos/pytorch-auto-drive/requirements.txt`
- Verification:
  - `import torch, mmcv, timm` PASS
  - `torch.cuda.is_available()` PASS (`True`)

### 3) `pcdet`
- Python 3.8
- Installed:
  - `torch==1.13.1+cu116`, `torchvision==0.14.1+cu116`, `torchaudio==0.13.1+cu116`
  - `spconv-cu116`
  - `repos/OpenPCDet/requirements.txt`, `open3d`
- Installed OpenPCDet in develop mode successfully
- Verification:
  - `import torch, spconv, pcdet` PASS
  - `torch.cuda.is_available()` PASS (`True`)

## CARLA Runtime Asset

- Docker image `carlasim/carla:0.10.0`: PASS (present locally)

## Tests Executed

1. `scripts/check_envs.sh --help`
2. `scripts/setup_envs_stage01.sh --help`
3. `scripts/check_envs.sh --repos-dir ../repos`
4. `conda run -n carla-client python -c "import pygame,numpy,cv2,psutil,requests; print('carla-client base deps ok')"`
5. `conda run -n carla-client python -c "import carla; print('carla import ok', carla.__file__)"`
6. `conda run -n pad python -c "import torch, mmcv, timm; assert torch.cuda.is_available(); print('pad OK', torch.__version__, mmcv.__version__, timm.__version__)"`
7. `conda run -n pcdet python -c "import torch, spconv, pcdet; print('torch', torch.__version__, torch.cuda.is_available()); print('spconv', spconv.__version__); print('pcdet OK')"`
8. `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv`

## Final Verification Summary

- `scripts/check_envs.sh --repos-dir ../repos`
- Result: `PASS=14 WARN=0 FAIL=0`

## Necessary Fallbacks / Notes

1. `carla==0.10.0` không có trên PyPI → cài từ wheel trích ra từ Docker image.
2. Torch runtime trong `pad/pcdet` ban đầu lỗi binary (`iJIT_NotifyEvent`) → cài lại từ PyTorch wheel index `cu116` để ổn định.
3. OpenPCDet extension build bị nghẽn ở `pointnet2_stack` với `voxel_query_gpu.cu`.
   - **Necessary upstream hotfix** (do compile blocker, phạm vi tối thiểu 1 file):
     - `repos/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/src/voxel_query_gpu.cu`
     - bỏ include `curand_kernel.h` và bỏ `curandState/curand_init` không dùng.
   - Build command dùng fallback an toàn phần cứng:
     - `CC/CXX` từ conda toolchain GCC 11
     - `TORCH_CUDA_ARCH_LIST=8.6`
     - `MAX_JOBS=1`

## Exit Criteria Status

- [x] All three conda envs activate without errors
- [x] `torch.cuda.is_available()` returns True in `pad` and `pcdet`
- [x] `import pcdet` works
- [x] `import mmcv` works
- [x] CARLA Docker image pulled
- [x] `.gitignore` configured
