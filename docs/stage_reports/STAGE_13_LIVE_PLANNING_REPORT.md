# STAGE 13B — Live Closed-Loop EPLS (Simulator-Only)

> Date: 2026-05-20  
> Status: IMPLEMENTED (workflow + orchestration), live CARLA execution blocked in current shell env by dependency split

## Scope Mode
- `stage13.mode`: `live_planning` for `run_planning_agent.py`
- `stage13.mode`: `iterative` for `iterative_train.py`
- Both require explicit `--allow-stage13-control`.

## Implemented Components
- New live closed-loop planning script:
  - `scripts/run_planning_agent.py`
  - Uses trained VAE + MDRNN + RMHC/RHEA to plan `[steer, throttle, brake]` each tick.
  - Runs in synchronous CARLA mode only.
  - Writes Stage 13-compatible artifacts:
    - `rgb/`, `semantic/`, `lidar/`, `metadata/`
    - `recording_summary.json`, `dataset_complete.json`, `rollout_manifest.json`
    - `planning_metrics.json` (planner-vs-random predicted reward trace)
- New iterative loop orchestrator:
  - `scripts/iterative_train.py`
  - Per cycle:
    1. collect planner rollouts (live control)
    2. collect baseline rollouts (`autopilot_noise` / `autopilot` / `random`)
    3. train VAE
    4. encode latents
    5. train MDRNN
    6. run offline planner evaluation
  - Writes `cycle_XXX/cycle_summary.json` and top-level `iterative_summary.json`.
- Added Stage 13B smoke tests (dry-run orchestration):
  - `tests/test_stage13b_workflow.py`

## Verification Results

### Unit/integration tests
Command:
```bash
python3 -m pytest -q tests/test_world_model_reward.py tests/test_vae.py tests/test_mdrnn.py tests/test_planners.py tests/test_stage13b_workflow.py
```
Result:
- `12 passed`

### Script compile check
Command:
```bash
python3 -m py_compile scripts/run_planning_agent.py scripts/iterative_train.py tests/test_stage13b_workflow.py
```
Result:
- Passed.

### Stage 13B smoke tests (dry-run)
- `run_planning_agent.py --dry-run` (mode `live_planning`) passed.
- `iterative_train.py --dry-run` (mode `iterative`) passed and generated full command pipeline + summary JSON.

## Live Run Blocker (Current Environment)
Attempted live smoke (`run_planning_agent.py` without `--dry-run`) from current shell failed with:
- `RuntimeError: CARLA Python API import failed: No module named 'carla'`

Root cause on this host:
- `carla-client` env has `carla` but lacks `torch`.
- `pcdet`/base env has `torch` but lacks `carla`.
- pip `carla` package install in base fails due legacy package incompatibility.

This is an environment composition issue, not a Stage 13B code-path failure.

## Remaining Operational Requirements
To run Stage 13B live end-to-end:
1. Use a single runtime env containing `carla`, `torch`, `opencv-python`, `numpy`, `pyyaml`.
2. Launch CARLA server and keep synchronous mode enabled.
3. Run `run_planning_agent.py` with `--allow-stage13-control` and `stage13.mode=live_planning`.
4. Run `iterative_train.py` with `--allow-stage13-control` and `stage13.mode=iterative`.

## Notes
- Stage 13A remains unchanged.
- Stage 13B implementation reused existing reward/terminal, recorder serialization schema, and planner/model components with minimal code-surface expansion.

## Stage 14 Follow-Up
- Stage 14 report: `docs/stage_reports/STAGE_14_SIMULATOR_DEMO_REPORT.md`
- Live CARLA execution is now verified for planner/random/autopilot-noise 100-frame smoke runs, plus a one-cycle iterative collect/train/evaluate smoke run.
