# CARLA Perception Lab — Agent Execution Plan

## 0. Project Intent

Build a practical autonomous-driving **perception** project using:

- CARLA as the simulator and synthetic data source.
- PytorchAutoDrive for lane / road / semantic segmentation.
- OpenPCDet for LiDAR 3D object detection.
- A new integration repo named `carla-perception-lab`.

The project must stay focused on **Computer Vision / perception**, not full end-to-end autonomous driving.

Final goal:

```text
CARLA simulated vehicle
→ RGB camera + semantic camera + LiDAR recording
→ lane/road segmentation
→ LiDAR 3D object detection
→ dashboard video
→ benchmark report
→ lightweight deployment/inference path
```

## 1. Hardware-Aware Constraints

Target machine profile:

```text
Laptop: ASUS ROG Zephyrus G16
CPU: Intel Core i7-13620H
GPU: NVIDIA RTX 4060 Laptop
RAM: 32GB
Storage: about 1.38TiB SSD total
OS: Linux Mint 22.2, Ubuntu 24.04 base
```

Hard constraints:

- Prefer CARLA packaged release or Docker over building CARLA from source.
- Do not run CARLA at 4K.
- Default simulation resolution: `1280x720`.
- Maximum initial resolution: `1920x1080`.
- Use 1 RGB camera, 1 semantic camera, and 1 LiDAR at first.
- Do not train large 3D models from scratch.
- OpenPCDet batch size should start at `1`.
- Do not run CARLA simulator and OpenPCDet training at the same time.
- Do not start with full nuScenes / Waymo training.
- Keep datasets small for MVP: `1000–5000` frames.
- Use separate conda environments for each major repo.
- Avoid modifying upstream repos directly unless absolutely necessary.
- All original code should live inside `carla-perception-lab`.

## 2. Non-Goals

Agents must not drift into these directions during the first version:

- Full autonomous driving stack.
- End-to-end policy learning.
- Reinforcement learning driving agent.
- Real-road vehicle deployment.
- CARLA source build unless packaged release or Docker fails.
- Custom real-world map creation.
- ROS / Autoware / Apollo integration.
- Training full-size OpenPCDet models on massive datasets.
- Building a production safety-critical driving system.

This is a simulator-only perception research / portfolio project.

## 3. Workspace Layout

Recommended local workspace:

```text
~/autonomous-driving/
├── repos/
│   ├── carla/
│   ├── pytorch-auto-drive/
│   └── OpenPCDet/
│
├── datasets/
│   ├── carla/
│   ├── kitti/
│   └── samples/
│
├── outputs/
│   ├── videos/
│   ├── visualizations/
│   ├── checkpoints/
│   └── benchmarks/
│
└── carla-perception-lab/
    ├── configs/
    ├── docs/
    ├── scripts/
    ├── src/
    ├── tests/
    ├── tools/
    ├── outputs/
    ├── requirements/
    ├── PLAN.md
    └── README.md
```

## 4. Environment Strategy

Use separate environments:

```text
carla-env   → CARLA Python API and recorder scripts
pad-env     → PytorchAutoDrive experiments
pcdet-env   → OpenPCDet demo/inference/fine-tune
lab-env     → dashboard, data conversion, benchmark utilities
```

Agents should never force all dependencies into one shared environment.

## 5. Repository Clone Commands

```bash
mkdir -p ~/autonomous-driving/repos
cd ~/autonomous-driving/repos

git clone https://github.com/carla-simulator/carla.git
git clone https://github.com/voldemortX/pytorch-auto-drive.git
git clone https://github.com/open-mmlab/OpenPCDet.git

cd ~/autonomous-driving
mkdir -p datasets/carla datasets/kitti datasets/samples
mkdir -p outputs/videos outputs/visualizations outputs/checkpoints outputs/benchmarks

mkdir -p carla-perception-lab/{configs,docs,scripts,src,tests,tools,outputs,requirements}
```

## 6. Stage Overview

| Stage | Name | Goal | Output |
|---|---|---|---|
| 0 | Repo audit | Read upstream docs and map commands | `docs/repo_audit.md` |
| 1 | Environment setup | Create envs and install minimum dependencies | `docs/setup_status.md` |
| 2 | CARLA smoke test | Run CARLA and control ego vehicle | screenshots + notes |
| 3 | CARLA recorder | Record RGB, semantic, LiDAR, metadata | `datasets/carla/raw/...` |
| 4 | PytorchAutoDrive smoke test | Run segmentation/lane inference | overlay images/videos |
| 5 | OpenPCDet smoke test | Run pretrained 3D detection demo | point cloud visualization |
| 6 | Dataset interface | Normalize CARLA output layout | `src/data/...` |
| 7 | Segmentation integration | Run segmentation on CARLA frames | masks + overlay videos |
| 8 | LiDAR integration | Run 3D detection on sample data | 3D boxes + BEV images |
| 9 | Dashboard | Combine RGB, segmentation, BEV, metrics | `outputs/videos/dashboard.mp4` |
| 10 | Benchmark | Measure FPS, latency, memory, weather impact | `outputs/benchmarks/report.md` |
| 11 | Lightweight deployment | Export / package inference path where feasible | deploy notes + scripts |
| 12 | Final README | Clean docs and reproducible commands | polished GitHub repo |

## 7. Stage Details

### Stage 0 — Repo Audit

Agent tasks:

1. Read these repositories locally:
   - `repos/carla`
   - `repos/pytorch-auto-drive`
   - `repos/OpenPCDet`
2. Identify official install commands, demo commands, and expected input/output formats.
3. Do not modify code yet.
4. Write:

```text
carla-perception-lab/docs/repo_audit.md
```

Required contents:

```text
- CARLA version / package choice
- CARLA Python API setup notes
- PytorchAutoDrive required Python/PyTorch versions
- PytorchAutoDrive inference command
- OpenPCDet install notes
- OpenPCDet demo command
- dependency risks
- disk usage estimate
- GPU/VRAM risks
```

Exit criteria:

```text
docs/repo_audit.md exists and includes exact commands to run each official demo.
```

### Stage 1 — Environment Setup

Agent tasks:

1. Create environment documentation.
2. Install only the minimum needed to run smoke tests.
3. Prefer stable versions over latest versions.

Output:

```text
carla-perception-lab/docs/setup_status.md
carla-perception-lab/requirements/lab.txt
```

Exit criteria:

```text
- Python imports work.
- `nvidia-smi` sees RTX GPU.
- No repo has been modified unnecessarily.
```

### Stage 2 — CARLA Smoke Test

Goal:

Run CARLA and confirm the simulator works on the target laptop.

Recommended settings:

```text
resolution: 1280x720
map: Town03 or Town10HD
weather: ClearNoon
sensors: minimal
```

Tasks:

1. Start CARLA packaged release or Docker.
2. Run manual control or a simple client script.
3. Record notes:
   - startup success/failure
   - FPS rough estimate
   - GPU memory usage
   - CPU/RAM usage
   - known issues

Output:

```text
docs/carla_smoke_test.md
outputs/visualizations/carla_smoke_test.png
```

Exit criteria:

```text
CARLA opens, ego vehicle can move, Python client connects.
```

### Stage 3 — CARLA Recorder

Goal:

Create original project code that records sensor data from CARLA.

Script:

```text
scripts/record_carla.py
```

Default command:

```bash
python scripts/record_carla.py \
  --town Town03 \
  --weather ClearNoon \
  --frames 1000 \
  --width 1280 \
  --height 720 \
  --output ../datasets/carla/town03_clear_001
```

Target dataset layout:

```text
datasets/carla/town03_clear_001/
├── rgb/
│   ├── 000000.png
│   └── ...
├── semantic/
│   ├── 000000.png
│   └── ...
├── lidar/
│   ├── 000000.bin
│   └── ...
├── ego_pose/
│   ├── 000000.json
│   └── ...
├── calib/
│   └── sensors.json
└── scenario.json
```

Requirements:

- Save metadata.
- Keep frame indices aligned.
- Avoid over-recording.
- Add `--max-disk-gb` guard if possible.
- Print FPS and disk usage while recording.

Exit criteria:

```text
1000 frames recorded without crashing and with aligned RGB/semantic/LiDAR frame IDs.
```

### Stage 4 — PytorchAutoDrive Smoke Test

Goal:

Run official PytorchAutoDrive inference before integration.

Tasks:

1. Follow upstream demo/inference instructions.
2. Use sample image/video first.
3. Produce overlay output.
4. Document exact command.

Output:

```text
docs/pytorch_auto_drive_smoke_test.md
outputs/visualizations/pad_demo_overlay.png
```

Exit criteria:

```text
Segmentation/lane overlay generated from a sample input.
```

### Stage 5 — OpenPCDet Smoke Test

Goal:

Run official OpenPCDet pretrained demo before integration.

Tasks:

1. Install OpenPCDet dependencies in `pcdet-env`.
2. Run quick demo with pretrained checkpoint and sample point cloud.
3. Export visualization image/screenshot.
4. Document exact command and any dependency fixes.

Output:

```text
docs/openpcdet_smoke_test.md
outputs/visualizations/openpcdet_demo.png
```

Exit criteria:

```text
OpenPCDet loads checkpoint and generates 3D predictions on sample data.
```

### Stage 6 — Dataset Interface

Goal:

Create a clean data interface for CARLA recordings.

Files:

```text
src/data/carla_dataset.py
src/data/io_utils.py
src/data/transforms.py
```

Functions:

```text
load_rgb(frame_id)
load_semantic(frame_id)
load_lidar(frame_id)
load_pose(frame_id)
load_calibration()
iter_frames()
```

Tests:

```text
tests/test_carla_dataset.py
```

Exit criteria:

```text
Dataset loader can iterate over recorded frames and validate alignment.
```

### Stage 7 — Segmentation Integration

Goal:

Run segmentation on CARLA RGB frames.

Script:

```text
scripts/run_segmentation.py
```

Command:

```bash
python scripts/run_segmentation.py \
  --input ../datasets/carla/town03_clear_001/rgb \
  --output outputs/segmentation/town03_clear_001 \
  --model erfnet \
  --save-overlay
```

Outputs:

```text
outputs/segmentation/town03_clear_001/
├── masks/
├── overlays/
└── metrics.json
```

Metrics:

```text
- FPS
- latency_ms_mean
- latency_ms_p95
- GPU memory peak
```

Exit criteria:

```text
Overlay images/video exist and benchmark JSON is saved.
```

### Stage 8 — LiDAR / OpenPCDet Integration

Goal:

Run 3D detection on LiDAR data.

First version:

- Use KITTI sample or OpenPCDet sample.
- Do not immediately require CARLA-to-KITTI conversion.

Second version:

- Add a minimal CARLA-to-KITTI-like converter.

Scripts:

```text
scripts/run_lidar_detection.py
tools/convert_carla_to_kitti.py
```

Outputs:

```text
outputs/lidar_detection/
├── predictions/
├── bev/
└── metrics.json
```

Exit criteria:

```text
3D boxes or BEV visualizations are generated for at least sample LiDAR frames.
```

### Stage 9 — Dashboard Video

Goal:

Create the main visual demo.

Script:

```text
scripts/make_dashboard.py
```

Dashboard layout:

```text
Top-left: RGB camera
Top-right: segmentation overlay
Bottom-left: LiDAR BEV
Bottom-right: metrics panel
```

Output:

```text
outputs/videos/dashboard_town03_clear.mp4
```

Required overlay text:

```text
Town
Weather
Frame ID
Segmentation FPS
LiDAR inference FPS
GPU memory
```

Exit criteria:

```text
Dashboard video plays and shows synchronized results.
```

### Stage 10 — Benchmark

Goal:

Show this is not just a demo.

Benchmark scenarios:

```text
Town03_ClearNoon
Town03_HardRainNoon
Town03_CloudyNoon
```

Optional later:

```text
Town10HD_ClearNoon
Night / low light
Fog
```

Benchmark table:

```text
scenario, resolution, sensors, seg_fps, lidar_fps, gpu_mem_gb, notes
```

Output:

```text
outputs/benchmarks/report.md
outputs/benchmarks/results.csv
```

Exit criteria:

```text
Benchmark report has at least 3 scenarios and clear failure notes.
```

### Stage 11 — Lightweight Deployment

Goal:

Add a practical deployment/inference story without pretending this is road-ready.

Allowed deployment scope:

```text
- local offline inference
- Docker image for dashboard/inference
- ONNX export for segmentation if supported
- TensorRT notes if feasible
- CLI scripts for reproducible inference
```

Not allowed:

```text
- real vehicle deployment
- road testing
- safety-critical claims
```

Deliverables:

```text
Dockerfile
docker-compose.yml
scripts/deploy_local.sh
docs/deployment.md
```

Deployment test:

```bash
python scripts/run_segmentation.py --help
python scripts/make_dashboard.py --help
```

Exit criteria:

```text
A new user can run offline inference/dashboard from documented commands.
```

### Stage 12 — Final README

README must include:

```text
- project goal
- architecture diagram
- hardware profile
- setup commands
- demo video/GIF
- dataset recording instructions
- segmentation instructions
- LiDAR detection instructions
- dashboard instructions
- benchmark results
- limitations
- safety note: simulator-only, not for real-road driving
```

Exit criteria:

```text
README lets another developer reproduce the MVP from scratch.
```

## 8. Agent Rules

Agents must follow these rules:

1. Never overwrite upstream repos.
2. Keep custom code inside `carla-perception-lab`.
3. After each stage, write a short stage report in `docs/stage_reports/`.
4. Every script must support `--help`.
5. Every long-running script must print progress.
6. Every generated output must go under `outputs/` or `datasets/`.
7. Do not assume unlimited VRAM.
8. Prefer small batch sizes.
9. Prefer CPU-safe preprocessing and GPU-only inference/training.
10. Do not train large models without explicit approval.
11. Do not use real-road deployment language.
12. Every stage must have a clear exit criterion before moving on.

## 9. Build / Test / Deploy Requirements

### Build

Minimum build means:

```text
- clone repos
- install envs
- run official smoke demos
- run project scripts
```

CARLA source build is not required for MVP.

### Test

Required tests:

```text
tests/test_carla_dataset.py
tests/test_file_layout.py
tests/test_config_load.py
tests/test_dashboard_inputs.py
```

Recommended commands:

```bash
pytest tests/
python scripts/record_carla.py --help
python scripts/run_segmentation.py --help
python scripts/run_lidar_detection.py --help
python scripts/make_dashboard.py --help
```

### Deploy

Deployment means **local offline deployment**, not real-world vehicle deployment.

Allowed deploy deliverables:

```text
- Dockerfile for integration repo
- local CLI runner
- documented conda envs
- exported segmentation model if feasible
- reproducible dashboard generation command
```

## 10. MVP Definition

The MVP is complete only when:

```text
- CARLA runs.
- 1000 aligned frames are recorded.
- PytorchAutoDrive produces segmentation/lane overlay.
- OpenPCDet demo/inference produces 3D visualization.
- Dashboard video is generated.
- Benchmark report exists.
- README has reproducible commands.
```

## 11. Final Project Definition

The final portfolio version is complete when:

```text
- 3 weather scenarios are benchmarked.
- CARLA dataset loader is tested.
- Segmentation integration works on CARLA frames.
- LiDAR detection path is documented and visualized.
- Dashboard video is polished.
- Local deployment path is documented.
- Limitations are clearly stated.
```

## 12. Suggested Timeline

| Time | Target |
|---|---|
| Week 1 | repo audit, env setup, CARLA smoke test |
| Week 2 | CARLA recorder + PytorchAutoDrive smoke test |
| Week 3 | OpenPCDet smoke test + dataset loader |
| Week 4 | segmentation integration + dashboard MVP |
| Week 5 | LiDAR integration + benchmark |
| Week 6 | deployment docs + README polish |

If the laptop overheats or VRAM becomes limiting, reduce scope in this order:

1. lower resolution to 720p
2. reduce sensors
3. reduce frame count
4. skip fine-tuning
5. run OpenPCDet only on sample/KITTI data
6. keep dashboard as the main deliverable
