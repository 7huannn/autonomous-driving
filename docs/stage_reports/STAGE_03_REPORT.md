# STAGE 03 — CARLA Recorder Report

> Date: 2026-05-12  
> Status: IMPLEMENTED / SMOKE VALIDATED / LONG-RUN CARLA OOM BLOCKER

## Scope

Executed `final_plan/STAGE_03_CARLA_RECORDER.md` only:
- create a CARLA recorder script inside `carla-perception-lab`
- create the sensor YAML config
- save synchronized RGB, semantic segmentation, LiDAR, and per-frame metadata
- provide reproducible CLI commands and `--help`
- avoid modifying upstream repos in `../repos`

## Implemented Artifacts

- `scripts/carla_recorder.py`
  - supports `--help`
  - supports `--check-config`
  - supports overrides for frames, output dir, camera resolution, traffic, walkers, LiDAR density, timeout, and Traffic Manager port
  - uses CARLA synchronous mode with `world.tick()`
  - saves:
    - `rgb/*.png`
    - `semantic/*.png`
    - `lidar/*.npy`
    - `metadata/*.json`
    - `recording_summary.json`
- `configs/sensor_config.yaml`
  - default Stage 03 sensor config
  - Traffic Manager default set to `10000` because Stage 02 found host ports `8000/9000` occupied by Portainer

## Validation Performed

### Static / CLI Checks

1. `conda run -n carla-client python -m py_compile scripts/carla_recorder.py`
   - PASS
2. `conda run -n carla-client python scripts/carla_recorder.py --help`
   - PASS
3. `conda run -n carla-client python scripts/carla_recorder.py --check-config`
   - PASS

### Runtime Smoke Check

CARLA server:

```bash
docker run --rm --net=host --gpus all --name carla-stage03-recorder \
  carlasim/carla:0.10.0 \
  bash CarlaUnreal.sh -RenderOffScreen -nosound -quality-level=Low -ResX=640 -ResY=360
```

Recorder smoke:

```bash
conda run -n carla-client python scripts/carla_recorder.py \
  --output-dir data/raw/stage03_smoke_10 \
  --num-frames 10 \
  --width 80 \
  --height 45 \
  --traffic-vehicles 0 \
  --traffic-walkers 0 \
  --lidar-pps 1000 \
  --overwrite \
  --timeout 20
```

Result:

- PASS: 10 synchronized frames recorded
- PASS: CARLA client reconnects after recording
- Final observed VRAM: `6144 MiB / 8188 MiB`

Output verification:

```text
counts {'rgb': 10, 'semantic': 10, 'lidar': 10, 'metadata': 10}
rgb_shape (45, 80, 3)
semantic_shape (45, 80, 3)
semantic_unique_red_count 18
lidar_shape (43, 4) float32
frame_sync_0 {'rgb_camera': 1016, 'semantic_camera': 1016, 'lidar': 1016}
timestamps_monotonic True
```

## Exit Criteria Status

- [x] Script connects to CARLA server
- [x] Ego vehicle spawns with autopilot
- [x] Three sensors attached and producing data
- [x] Synchronous mode aligns RGB, semantic, and LiDAR frame IDs
- [x] RGB PNG files are written
- [x] Semantic PNG files preserve class IDs in the red channel
- [x] LiDAR arrays are `(N, 4)` float32 `.npy` files
- [x] Metadata JSON includes `frame_id`, `timestamp`, `sensor_frames`, and ego transform state
- [x] Metadata timestamps are monotonically increasing in smoke run
- [ ] 1000 synchronized frames recorded
- [ ] CARLA server stable throughout long recording

## Known Issues / Fallbacks

1. Long runs with two camera sensors on CARLA 0.10.0 UE5 + Town10HD_Opt still trigger server-side Vulkan OOM on this 8GB VRAM laptop.
   - Observed with 100-frame attempts even at `80x45`.
   - CARLA log ends with `Out of memory on Vulkan`.
   - This blocks the Stage 03 full 1000-frame exit criterion.
2. The smoke run is valid for recorder code and data format, but not for full dataset collection.
3. Recommended fallback before moving to Stage 04:
   - retry Stage 03 with the packaged CARLA release instead of Docker, or
   - use a lighter CARLA map if available in the runtime package, or
   - record RGB/semantic/LiDAR in smaller staged passes only if strict frame synchronization is waived.

## Next Recommended Action

Stay on Stage 03 until the long-run CARLA OOM is resolved or explicitly waived. Do not move to Stage 04 under strict policy until a usable recording target is produced.
