# STAGE 02 — CARLA Smoke Test Report

> Date: 2026-05-12  
> Status: COMPLETE / LOW-VRAM SENSOR PATH

## Scope

Executed `final_plan/STAGE_02_CARLA_SMOKE_TEST.md`:
- add a reusable CARLA connection smoke-test script
- add a runtime wrapper for upstream CARLA examples to avoid TM default-port failure on this host
- add bounded low-VRAM sensor visualization mode for this 8GB VRAM runtime
- verify script help/CLI behavior
- launch CARLA 0.10.0 in Docker headless low-quality mode
- connect the `carla-client` env to localhost:2000
- run VRAM check while CARLA is live
- run bounded `generate_traffic.py` tests

## Implemented Artifacts

- `scripts/test_carla_connection.py` — reusable client connection smoke test with `--help`
- `scripts/run_carla_example_tm_workaround.py` — runs upstream examples while remapping TM default ports (8000/9000 -> 10000), supports bounded runtime, and provides a safe sensor-visualization smoke path for `visualize_multiple_sensors.py`

## Validation Performed

### Script-level checks

1. `python carla-perception-lab/scripts/test_carla_connection.py --help`
   - PASS
2. `conda run -n carla-client python carla-perception-lab/scripts/test_carla_connection.py --timeout 1`
   - Expected fail when no server is running
   - Result: `[FAIL] Connection refused`

### Runtime checks attempted

3. `docker image inspect carlasim/carla:0.10.0`
   - PASS, image present locally
4. `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
   - PASS, Docker GPU runtime confirmed
5. `docker run --rm --net=host --gpus all carlasim/carla:0.10.0 bash CarlaUnreal.sh -RenderOffScreen -nosound -quality-level=Low -ResX=1280 -ResY=720`
   - PASS, CARLA container started and RPC ports 2000/2001 were listening
6. `conda run -n carla-client python carla-perception-lab/scripts/test_carla_connection.py --host localhost --port 2000 --timeout 20`
   - PASS
   - server_version: `0.10.0`
   - map_name: `Carla/Maps/Town10HD_Opt`
7. `nvidia-smi --query-gpu=memory.used,memory.total --format=csv`
   - PASS
   - observed while CARLA live: `2430 MiB / 8188 MiB`
8. Direct TM probe
   - FAIL on ports 8000/9000
   - PASS on ports 10000/10001/11000/12000
9. `generate_traffic.py` with TM workaround wrapper (fallback port 10000)
   - PASS (time-boxed run remains active, actor counts confirm spawn)
   - observed actor counts: vehicles=21, walkers=5
10. `manual_control.py` via wrapper (time-boxed)
   - PASS for smoke startup (process remained alive for timeout window)
11. `visualize_multiple_sensors.py` via wrapper safe mode
   - Command:
     `conda run -n carla-client env SDL_VIDEODRIVER=dummy python scripts/run_carla_example_tm_workaround.py --script ../repos/carla/PythonAPI/examples/visualize_multiple_sensors.py --safe-visualize-sensors --max-runtime 15 -- --res 320x180`
   - PASS
   - Result: `[PASS] safe visualize sensor smoke ran 15.0s with 2 sensors`
   - Client reconnected successfully after the sensor run, confirming the CARLA server stayed alive.
12. `generate_traffic.py` regression check via wrapper
   - Command:
     `conda run -n carla-client python scripts/run_carla_example_tm_workaround.py --script ../repos/carla/PythonAPI/examples/generate_traffic.py --tm-fallback-port 10000 --max-runtime 10 -- --number-of-vehicles 5 --number-of-walkers 0 --tm-port 8000`
   - PASS
   - Result: spawned and destroyed 5 vehicles; no native abort

## Results Summary

- CARLA Docker image is available locally
- Docker GPU runtime is now functional
- CARLA server launches in headless mode and RPC ports open
- `carla-client` connects successfully and retrieves world metadata
- VRAM usage during live CARLA run is within guardrail (`2430 MiB < 6 GiB target`)
- Traffic Manager default ports 8000/9000 are unstable on this host runtime, but fallback ports >=10000 work
- `generate_traffic.py` runs successfully with wrapper fallback port
- `manual_control.py` smoke startup works with wrapper fallback port
- Low-VRAM `visualize_multiple_sensors.py` safe path runs successfully without native abort
- Post-sensor and post-traffic client reconnect checks pass
- Final observed VRAM after validation: `5631 MiB / 8188 MiB`
- `dmesg` tail has existing boot/firmware/NVIDIA display warnings, but no new CARLA runtime GPU crash was observed in the final validation run
- No screenshot artifact was produced because final validation used headless/dummy display mode to stay within VRAM

## Known Issues / Fallbacks

1. Traffic Manager default ports 8000/9000 fail consistently on this runtime.
   - Workaround in project wrapper remaps to 10000.
2. The full upstream camera/LiDAR visualization configuration can exceed VRAM on this host with CARLA 0.10.0 UE5 + Town10HD_Opt.
   - Resolution: Stage 02 uses the wrapper's `--safe-visualize-sensors` path for bounded sensor callback validation.
   - Guidance for Stage 03: keep capture settings conservative first; avoid multiple simultaneous render cameras until the recorder is proven stable.

## Exit Criteria Status

- [x] Stage 02 connection script created
- [x] TM workaround script created
- [x] Script `--help` works
- [x] `carla-client` invocation path is validated
- [x] CARLA Docker image is present
- [x] CARLA server starts without crash
- [x] Client connects and retrieves world info
- [x] Traffic generation works (with documented TM workaround)
- [x] VRAM usage documented during live CARLA run
- [x] `manual_control.py` smoke startup works (with documented TM workaround)
- [x] `visualize_multiple_sensors.py` safe sensor path runs without native abort
- [x] No new CARLA runtime GPU crash observed in final `dmesg`/CARLA validation

## Next Recommended Action

Stage 02 is complete. Move to Stage 03 using the same low-VRAM guardrails: headless server, 640x360 or staged sensor bring-up first, TM fallback port 10000 if Portainer keeps ports 8000/9000 occupied.
