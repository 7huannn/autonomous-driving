# STAGE 00 — Repo Audit Report

> **Stage**: 00  
> **Status**: ✅ COMPLETE  
> **Date**: 2026-05-07  
> **Scope**: Read-only audit of three upstream repos; no code changes; no GPU usage

---

## 1. Repo Verification Results

All three repos verified successfully via `scripts/verify_repos.py`:

| Repo | Branch | Commit | Date | Size | Status |
|------|--------|--------|------|------|--------|
| `carla` | `ue5-dev` | `ffd9d275cb07` | 2026-05-06 | 3.3 GB | ✅ PASS |
| `pytorch-auto-drive` | `master` | `137e63a9e6c3` | 2023-10-04 | 8.5 MB | ✅ PASS |
| `OpenPCDet` | `master` | `233f849829b6` | 2025-10-08 | 10 MB | ✅ PASS |

---

## 2. Deep Audit: CARLA Simulator

### 2.1 Install & Runtime

| Item | Value |
|------|-------|
| UE version | Unreal Engine 5.5 |
| CARLA release | 0.10.0 |
| Docker image | `carlasim/carla:0.10.0` |
| Docker (headless) | `docker run --device=nvidia.com/gpu=all --net=host carlasim/carla:0.10.0 bash CarlaUnreal.sh -RenderOffScreen -nosound` |
| Docker (display) | Add `--user=$(id -u):$(id -g) --env=DISPLAY=$DISPLAY --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"` |
| Docker prereq | NVIDIA Container Toolkit v2 + CDI: `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml && sudo systemctl restart docker` |
| Python client | `pip install carla-*-cp3*-linux_x86_64.whl` from `PythonAPI/dist/` |
| Client deps | `numpy<2.0,>=1.24.4`, `pygame`, `open3d` (Python ≤ 3.11), `Pillow` |
| TCP ports | 2000 (main), 2001 (stream) |
| VRAM rec. | 16 GB (official), ours is 8 GB → use `-quality-level=Low -ResX=1280 -ResY=720` |

### 2.2 Sensor Specifications (from `ref_sensors.md`)

| Sensor | Blueprint | Default Res | Output | Key Attributes |
|--------|-----------|-------------|--------|----------------|
| RGB Camera | `sensor.camera.rgb` | 800×600 | BGRA 32-bit | `image_size_x/y`, `fov=90`, `sensor_tick` |
| Semantic Seg | `sensor.camera.semantic_segmentation` | 800×600 | Class ID in red channel | Same as RGB; use `CityScapesPalette` converter |
| LiDAR | `sensor.lidar.ray_cast` | — | 4D float `[x,y,z,intensity]` | `channels=32`, `points_per_second=56000`, `rotation_frequency=10`, `range=10.0`, `upper_fov=10`, `lower_fov=-30` |
| Semantic LiDAR | `sensor.lidar.ray_cast_semantic` | — | `[x,y,z,cos_angle,obj_idx,sem_tag]` | Same geometry attrs as LiDAR; no intensity/dropoff |

### 2.3 Full Semantic Tag Table (23 classes)

| ID | Tag | CityScapes Color (R,G,B) |
|----|-----|--------------------------|
| 0 | Unlabeled | (0, 0, 0) |
| 1 | Building | (70, 70, 70) |
| 2 | Fence | (100, 40, 40) |
| 3 | Other | (55, 90, 80) |
| 4 | Pedestrian | (220, 20, 60) |
| 5 | Pole | (153, 153, 153) |
| 6 | RoadLine | (157, 234, 50) |
| 7 | Road | (128, 64, 128) |
| 8 | SideWalk | (244, 35, 232) |
| 9 | Vegetation | (107, 142, 35) |
| 10 | Vehicles | (0, 0, 142) |
| 11 | Wall | (102, 102, 156) |
| 12 | TrafficSign | (220, 220, 0) |
| 13 | Sky | (70, 130, 180) |
| 14 | Ground | (81, 0, 81) |
| 15 | Bridge | (150, 100, 100) |
| 16 | RailTrack | (230, 150, 140) |
| 17 | GuardRail | (180, 165, 180) |
| 18 | TrafficLight | (250, 170, 30) |
| 19 | Static | (110, 190, 160) |
| 20 | Dynamic | (170, 120, 50) |
| 21 | Water | (45, 60, 150) |
| 22 | Terrain | (145, 170, 100) |

### 2.4 Key Example Scripts

| Script | Purpose |
|--------|---------|
| `manual_control.py` | Pygame-based vehicle control with sensor display |
| `generate_traffic.py` | Spawn N vehicles + M walkers with autopilot |
| `sensor_synchronization.py` | Sync multiple sensors using `CarlaSensorMasterClock` |
| `open3d_lidar.py` | Real-time LiDAR point cloud visualization |
| `visualize_multiple_sensors.py` | Multi-camera display |
| `recorder_replay.py` | Replay recorded simulations |

### 2.5 Recorder API

```python
client.start_recorder("/path/recording.log")       # Start recording
client.start_recorder("/path/recording.log", True)  # + additional data (velocities, physics)
client.stop_recorder()                               # Stop
client.replay_file("recording.log", start, duration, camera_id)  # Replay
client.show_recorder_file_info("recording.log")      # Inspect
```

**Estimate**: 1h recording with 50 traffic lights + 100 vehicles ≈ 200 MB.

### 2.6 Coordinate System

CARLA uses **UE coordinates**: **x-forward, y-right, z-up** (left-hand convention).

> ⚠️ **Important**: Many visualization tools invert Y-axis. Sensor data is returned in local space.

---

## 3. Deep Audit: PytorchAutoDrive (PAD)

### 3.1 Install & Dependencies

| Item | Value |
|------|-------|
| Python | ≥ 3.6 (recommend 3.8) |
| PyTorch | ≥ 1.6 (recommend 1.12.1) |
| CUDA | ≥ 9.2 |
| mmcv-full | ≥ 1.3.5 (recommend 1.7.0 for torch 1.12/cu116) |
| timm | ==0.4.5 (pinned) |
| Other key deps | `scipy==1.5.4`, `scikit_learn==0.23.2`, `opencv-python==4.5.4.58`, `thop`, `ninja>=1.8.2` |

**Install order**: PyTorch → mmcv-full (pre-built wheel) → `pip install -r requirements.txt`

### 3.2 Segmentation Models (Performance on 2080 Ti)

| Model | Resolution | FPS | FLOPs (G) | Params (M) | Cityscapes mIoU |
|-------|-----------|-----|-----------|------------|-----------------|
| **ERFNet** ★ | 512×1024 | **85.51** | 60.11 | **2.07** | **72.47** |
| ENet | 512×1024 | 55.69 | 10.88 | 0.35 | 65.74 |
| FCN | 512×1024 | 12.06 | 865.69 | 51.95 | 68.20 |
| DeepLabV2 | 512×1024 | 12.93 | 722.37 | 43.90 | 72.12 |
| DeepLabV3 | 512×1024 | 10.26 | 966.61 | 58.63 | 74.67 |

★ **ERFNet is our primary choice**: best speed/accuracy tradeoff, 2M params, 85 FPS.

### 3.3 Checkpoint URLs (Segmentation)

| Model | Dataset | File | URL |
|-------|---------|------|-----|
| ERFNet | Cityscapes 512×1024 | `erfnet_cityscapes_512x1024_20200918.pt` | [Google Drive](https://drive.google.com/file/d/1uzBSboKD-Xt0K6VHd2aF561Cy13q9xRe/view?usp=sharing) |
| ENet | Cityscapes 512×1024 | `enet_cityscapes_512x1024.pt` | [Google Drive](https://drive.google.com/file/d/1oK2mKCetOtY8KFaKLjs7-jOMkxZjbIQD/view?usp=sharing) |
| ERFNet encoder | ImageNet pretrained | `erfnet_encoder_pretrained.pth.tar` | [Official Repo](https://github.com/Eromera/erfnet_pytorch/tree/master/trained_models) |

### 3.4 ERFNet Config Details (`cityscapes_512x1024.py`)

```python
train = dict(
    batch_size=10, workers=8, num_epochs=150,
    input_size=(512, 1024), num_classes=19,
    save_dir='./checkpoints',
)
test = dict(
    batch_size=1, workers=0, num_classes=19,
    original_size=(512, 1024),
)
model = dict(
    name='ERFNet', num_classes=19,
    dropout_1=0.03, dropout_2=0.3,
    pretrained_weights='erfnet_encoder_pretrained.pth.tar'
)
```

### 3.5 Key CLI Commands

```bash
# Training
python main_semseg.py --train --config=<config> --mixed-precision

# Validation
python main_semseg.py --val --config=<config> --checkpoint=<pt>

# Inference on images
python tools/vis/seg_img_dir.py --image-path=<dir> --pred --config=<config> --checkpoint=<pt> --save-path=<dir>

# Inference on video
python tools/vis/seg_video.py --video-path=<avi> --pred --config=<config> --checkpoint=<pt> --save-path=<avi>

# Profiling
python tools/profiling.py --mode=simple --config=<config> --height=720 --width=1280
```

### 3.6 Deployment Path

```
PyTorch (.pt) → ONNX (.onnx) → TensorRT (.engine)
```

- ERFNet: ✅ ONNX + TensorRT supported
- ENet: ❌ NOT supported for ONNX/TensorRT
- Requires separate venv for deployment (onnxruntime + tensorrt)

### 3.7 PAD Test Images

- URL: [PAD_test_images (129 MB)](https://drive.google.com/file/d/1XQvBS1uoHeIgUv7oDQ4Vp1tWYi0oAGhU/view?usp=sharing)
- Contains: Cityscapes, PASCAL VOC, TuSimple, CULane samples + videos

---

## 4. Deep Audit: OpenPCDet

### 4.1 Install & Dependencies

| Item | Value |
|------|-------|
| Python | ≥ 3.6 (recommend 3.8) |
| PyTorch | ≥ 1.1, tested ≤ 1.13 |
| CUDA | ≥ 9.0 |
| spconv | v2.x pip install (`spconv-cu116` or `spconv-cu118`) |
| CUDA extensions | 7 custom ops built via `python setup.py develop` |
| Key deps | `numba`, `SharedArray`, `open3d`, `easydict`, `pyyaml`, `tensorboardX`, `pyquaternion` |

### 4.2 CUDA Extensions Built by `setup.py`

1. `iou3d_nms` — 3D IoU + NMS operations
2. `roiaware_pool3d` — RoI-aware point cloud pooling
3. `pointnet2_stack` — PointNet++ set abstraction (stacked)
4. `pointnet2_batch` — PointNet++ set abstraction (batched)
5. `center_ops` — Center-based operations (CenterPoint)
6. `voxel_ops` — Voxelization operations
7. `bev_pool` — BEV pooling for multi-modal

> ⚠️ **Build requires `TORCH_CUDA_ARCH_LIST`**. For RTX 4060 (Ada Lovelace): set `"8.9"`.

### 4.3 3D Detection Models

| Model | Params | KITTI 3D AP (Car Mod.) | Train Time (8×V100) | Our Feasibility |
|-------|--------|----------------------|---------------------|-----------------|
| **PointPillar** ★ | 4.8M | 77.28 | ~1.2h | ✅ Best for 8GB VRAM |
| SECOND | ~5M | 78.62 | ~1.5h | ✅ OK |
| PV-RCNN | ~13M | 83.61 | ~5h | ⚠️ Tight VRAM |
| Voxel R-CNN | ~10M | 84.52 | ~4h | ⚠️ Tight VRAM |

★ **PointPillar is our primary choice**: fastest, simplest, fits in 8GB VRAM at bs=1.

### 4.4 Checkpoint URLs (3D Detection)

| Model | Dataset | URL |
|-------|---------|-----|
| PointPillar | KITTI | [pointpillar_7728.pth](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view) |
| SECOND | KITTI | [second_7862.pth](https://drive.google.com/file/d/1-01zsPOsqanZQqIIyy7FpNXStL3y4jdR/view) |
| PV-RCNN | KITTI | [pv_rcnn_8369.pth](https://drive.google.com/file/d/1lIOq4Hxr0W3qsX83ilQv0nk1Cls6KAr-/view) |

### 4.5 Custom Dataset Format

```
data/custom/
├── ImageSets/
│   ├── train.txt         # One frame ID per line: 000000\n000001\n...
│   └── val.txt
├── points/
│   └── 000000.npy        # shape: (N, 4), dtype: float32, columns: [x, y, z, intensity]
└── labels/
    └── 000000.txt        # One box per line: x y z dx dy dz heading_angle class_name
```

**Box format**: `[center_x, center_y, center_z, length, width, height, heading_angle, class_name]`

**Voxel size constraints** (for PointPillar):
- Point cloud range along x&y / voxel_size must be multiple of 16
- Point cloud range along z / voxel_size must be 40

**Default custom_dataset.yaml**:
- `POINT_CLOUD_RANGE: [-75.2, -75.2, -2, 75.2, 75.2, 4]`
- `VOXEL_SIZE: [0.1, 0.1, 0.15]`
- `CLASS_NAMES: ['Vehicle', 'Pedestrian', 'Cyclist']`
- `MAP_CLASS_TO_KITTI: {'Vehicle': 'Car', 'Pedestrian': 'Pedestrian', 'Cyclist': 'Cyclist'}`

### 4.6 Key CLI Commands

```bash
# Demo inference
python tools/demo.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt pointpillar.pth --data_path data.npy

# Training
python tools/train.py --cfg_file ${CONFIG} --batch_size 1

# Testing
python tools/test.py --cfg_file ${CONFIG} --batch_size 1 --ckpt ${CKPT}

# Create custom dataset infos
python -m pcdet.datasets.custom.custom_dataset create_custom_infos tools/cfgs/dataset_configs/custom_dataset.yaml
```

### 4.7 Docker Reference

`docker/cu116.Dockerfile`:
- Base: `nvidia/cuda:11.6.2-devel-ubuntu20.04`
- PyTorch: 1.13.1+cu116
- `TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"`
- Includes: OpenCV 4.2.0 with CUDA, spconv-cu116

---

## 5. Dependency Conflict Analysis

### 5.1 Cross-Repo Conflicts

| Dependency | CARLA client | PAD | OpenPCDet | Conflict? |
|------------|-------------|-----|-----------|-----------|
| Python | 3.10 | 3.8 | 3.8 | ⚠️ Separate envs needed |
| PyTorch | — | 1.12.1 | 1.13.1 | ⚠️ Different minor versions |
| CUDA | — | 11.6 | 11.6 | ✅ Compatible |
| numpy | <2.0 | (any) | (any) | ✅ |
| opencv-python | needed | ==4.5.4.58 | needed | ✅ |
| open3d | needed | — | needed | ✅ |
| mmcv-full | — | ≥1.3.5 | — | Isolated to `pad` |
| spconv | — | — | cu116 | Isolated to `pcdet` |
| scipy | — | ==1.5.4 | — | Isolated to `pad` |
| timm | — | ==0.4.5 | — | Isolated to `pad` |

**Decision**: 3 separate conda envs as planned. No conflicts within each env.

### 5.2 CUDA/GPU Compatibility

| Component | CUDA Requirement | Our System |
|-----------|-----------------|------------|
| CARLA 0.10.0 | Needs NVIDIA driver ≥ 550 | ✅ To verify |
| PAD (PyTorch 1.12.1+cu116) | CUDA 11.6 runtime | ✅ Compatible with driver ≥ 525 |
| OpenPCDet (PyTorch 1.13.1+cu116) | CUDA 11.6 runtime | ✅ Compatible with driver ≥ 525 |
| spconv-cu116 | CUDA 11.6 headers | ✅ If using conda cudatoolkit |

> **Note**: Conda-managed `cudatoolkit` bundles CUDA runtime, so system CUDA toolkit version doesn't matter for PyTorch. Only the NVIDIA **driver** version matters.

### 5.3 RTX 4060 CUDA Architecture

- Architecture: **Ada Lovelace**
- Compute capability: **8.9**
- OpenPCDet Dockerfile default `TORCH_CUDA_ARCH_LIST` ends at 8.6+PTX → **should still work via PTX JIT**, but setting `"8.9"` explicitly would be faster.

---

## 6. Coordinate System Analysis

### CARLA → OpenPCDet Coordinate Mapping

| Axis | CARLA (UE) | KITTI/OpenPCDet | Transform |
|------|-----------|-----------------|-----------|
| X | Forward ✅ | Forward | `x_pcdet = x_carla` |
| Y | Right | Left | `y_pcdet = -y_carla` ⚠️ |
| Z | Up ✅ | Up | `z_pcdet = z_carla` |

> ⚠️ **Y-axis negation required** when converting CARLA LiDAR data to OpenPCDet format.

### CARLA Semantic → Cityscapes Class Mapping

CARLA has **23 classes**, Cityscapes has **19 training classes**. Many map directly; some must be merged or set to `ignore (255)`. Full mapping defined in STAGE_06.

---

## 7. Checkpoint Download Catalog

| Model | Framework | Dataset | Size | Google Drive ID |
|-------|-----------|---------|------|-----------------|
| ERFNet | PAD | Cityscapes 512×1024 | ~8 MB | `1uzBSboKD-Xt0K6VHd2aF561Cy13q9xRe` |
| ERFNet encoder | PAD | ImageNet pretrained | ~6 MB | [Official repo](https://github.com/Eromera/erfnet_pytorch/tree/master/trained_models) |
| PointPillar | OpenPCDet | KITTI | ~18 MB | `1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm` |
| SECOND | OpenPCDet | KITTI | ~20 MB | `1-01zsPOsqanZQqIIyy7FpNXStL3y4jdR` |
| PV-RCNN | OpenPCDet | KITTI | ~50 MB | `1lIOq4Hxr0W3qsX83ilQv0nk1Cls6KAr-` |

---

## 8. Exit Criteria Checklist

- [x] All three repos read thoroughly (README, install docs, API docs, configs, examples)
- [x] Version requirements documented (Python, CUDA, PyTorch, key deps)
- [x] Input/output formats for each repo documented
- [x] Pretrained checkpoint download URLs identified (ERFNet, PointPillar, SECOND, PV-RCNN)
- [x] VRAM estimates documented (Section 9 of FINAL_PLAN.md)
- [x] FINAL_PLAN.md Section 1 is complete and accurate ✅
- [x] Dependency conflict risks identified between repos ✅
- [x] Verification script created and run ✅

---

## 9. Files Created

| File | Purpose |
|------|---------|
| `scripts/verify_repos.py` | Automated repo structure verification script |
| `docs/stage_reports/repo_audit_results.json` | Machine-readable verification results |
| `docs/stage_reports/STAGE_00_REPORT.md` | This report |

## 10. Files Modified

None. Stage 00 is read-only.

## 11. Tests Executed

| Test | Result |
|------|--------|
| `python scripts/verify_repos.py --repos-dir ../repos` | ✅ ALL REPOS VERIFIED |
| CARLA repo: 12/12 key files found | ✅ |
| PAD repo: 12/12 key files found | ✅ |
| OpenPCDet repo: 14/14 key files found | ✅ |

## 12. Known Issues

1. **CARLA 0.10.0 Docker image**: Not yet pulled — will be verified in Stage 02.
2. **PAD repo age**: Last commit was 2023-10-04 — potential compatibility issues with modern PyTorch. ERFNet itself is simple enough that this should not be a problem.
3. **OpenPCDet `TORCH_CUDA_ARCH_LIST`**: Default Dockerfile targets up to 8.6+PTX. RTX 4060 is 8.9 — will rely on PTX JIT or set explicitly in Stage 01.
4. **CARLA Python client wheel**: Not yet available locally — depends on Docker extraction or packaged release download in Stage 01.

## 13. Assumptions

1. We assume CARLA 0.10.0 Docker image is publicly available at `carlasim/carla:0.10.0`.
2. We assume NVIDIA driver ≥ 550 is installed (required for UE5 + CUDA 11.6).
3. We assume `conda` is available on the target machine.
4. We assume internet access is available during Stage 01 (env setup) but NOT during final deployment (Stage 11).

---

## Next Recommended Stage

**→ STAGE_01_ENV_SETUP.md**: Create 3 conda environments and install all dependencies.
