# STAGE 03 — CARLA Recorder Report

> Date: 2026-05-13  
> Status: IMPLEMENTED / LONG-RUN VALIDATED (LIGHT PROFILE)

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

1. CARLA Docker `0.10.0` runtime on this host only exposes maps:
   - `Town10HD_Opt`
   - `Mine_01`
2. Previous `Town03` config was not available in runtime, and recorder did not apply map selection.
3. `Town10HD_Opt` remains unstable on this host (frequent `std::exception`/Vulkan pressure). `Mine_01` is stable for 1000-frame long run with reduced default load.
4. In Mine_01 long-runs, ego snapshot can disappear late in run (`state_valid=false` fallback in metadata for affected frames). Sensor synchronization and timestamp monotonicity stay valid.

## Next Recommended Action

Proceed to Stage 04 with the validated light Stage 03 profile (`Mine_01`, `320x180`, `LiDAR 20k pps`, no background traffic), if this dataset profile is acceptable for downstream tasks.

## Revalidation on 2026-05-13

### Recorder fixes applied

- Added `--map` CLI override and `carla.map` override support.
- Recorder now validates/loads a runtime-available map before recording.
- Restored robust cleanup:
  - `sensor.stop()`
  - batch actor destruction
  - restore TM/world synchronous settings in `finally`
- Added clearer runtime diagnostics (`world tick`, snapshot, sensor-sync, and write-stage errors).

### Config changes applied

- Default map changed to `Mine_01`.
- Default load reduced:
  - camera resolution `320x180`
  - LiDAR `points_per_second: 20000`
  - `traffic_vehicles: 0`

### Long-run result (default config after fixes)

Command:

```bash
conda run -n carla-client python scripts/carla_recorder.py \
  --output-dir data/raw/stage03_longrun_1000_snapshotmeta \
  --overwrite --timeout 20
```

Observed result:

- PASS: `1000/1000` frames written
- PASS: synchronized sensor frames (`rgb == semantic == lidar`) across run
- PASS: monotonic metadata timestamps
- PASS: clean process exit (no post-run abort)

Output check:

```text
counts {'rgb': 1000, 'semantic': 1000, 'lidar': 1000, 'metadata': 1000}
rgb_shape (180, 320, 3)
semantic_shape (180, 320, 3)
lidar_shape (829, 4) float32
timestamps_monotonic True
sync_all True
ego_state_invalid_frames 449
```

## Revalidation Update (2026-05-13, Hardened Plan)

Recorder contract was extended to support downstream Stage 09 requirements:

- New artifacts per recording:
  - `calib/sensors.json`
  - `scenario.json`
  - `dataset_complete.json`
- `recording_summary.json` now includes:
  - CARLA/runtime metadata
  - recording parameters
  - integrity section
  - `complete` marker
- Integrity marker is now hard-enforced:
  - if post-run integrity fails, recorder exits non-zero.

Additional runtime robustness/fixes:
- startup connect wait (`carla.startup_wait_seconds`, default 30s)
- optional weather preset from config
- optional cyclist-biased traffic spawn (`--prefer-cyclist-vehicles`)
- pedestrian fallback spawn when navmesh random locations are unavailable
- Cyclist class detection expanded to include motorcycle-type actors

### New validated recordings

1. Canonical main dataset:
- `data/raw/stage03_mine_main_1000`
- status: complete, aligned 1000 frames

2. Detection-label auxiliary dataset:
- `data/raw/stage03_mine_aux_200_traffic_far`
- status: complete, aligned 200 frames
- actor labels include `Vehicle` and `Pedestrian`

### Town10 collection status

On this host/runtime, repeated Town10 attempts remain unstable (native `std::exception`/abort before usable aligned output). These attempts are explicitly quarantined and excluded from progression checks.
