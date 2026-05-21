# Stage 14 Simulator Demo Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a verified simulator-only CARLA demo with dynamic traffic, pedestrians, perception overlays, and Stage 13 EPLS live planning artifacts.

**Architecture:** Keep Stage 13A as the offline world-model path and add Stage 14 as the completion layer: live CARLA closed-loop smoke, baseline comparison, one iterative collect/train/evaluate smoke cycle, and final dashboard/demo artifacts. CARLA collection and model training may use different Python environments; configs must make live control opt-in and keep the default offline mode unchanged.

**Tech Stack:** CARLA Python API, PyTorch, OpenCV, NumPy, PyYAML, pytest, existing `src/world_model`, existing `scripts/*`, existing dashboard/video tooling.

---

## Operating Rules

- This project is simulator-only. Do not add real vehicle control, real-road deployment claims, ROS, Autoware, Apollo, or production autonomous-driving claims.
- Do not claim "complete", "passing", "paper reproduced", or "high accuracy" without fresh command output and artifact paths.
- Do not use destructive commands such as `git reset --hard`, `git checkout --`, or broad deletion commands.
- Do not revert or overwrite user changes. The worktree may already contain unrelated dirty files.
- Keep `configs/world_model.yaml` defaulting to `stage13.mode: offline`.
- Live CARLA control must require `--allow-stage13-control`.
- If CARLA server, GPU, or environment setup blocks a live run, document the exact command, stderr, and environment probe. Do not mark Stage 14 complete.

## Current Known State

- Stage 13A offline world model exists and has smoke artifacts:
  - `output/world_model/vae_realistic_lite/best.pth`
  - `output/world_model/mdrnn_realistic_lite/best.pth`
  - `output/world_model/demo_stage13_realistic_lite_triptych_real_recon_dream.mp4`
- Stage 13B workflow code exists but must be verified in the local worktree:
  - `scripts/run_planning_agent.py`
  - `scripts/iterative_train.py`
  - `tests/test_stage13b_workflow.py`
  - `docs/stage_reports/STAGE_13_LIVE_PLANNING_REPORT.md`
- The previous live blocker was environment split: one environment had `carla`, another had CUDA `torch`.
- The repository's perception metrics are operational but not high-accuracy by default. Do not claim high accuracy unless new evaluation supports it.

## File Ownership

Create or modify only these files unless a failing test proves another file must change:

- Create: `configs/world_model_live.yaml`
- Create: `configs/world_model_iterative.yaml`
- Modify: `scripts/iterative_train.py`
- Modify: `tests/test_stage13b_workflow.py`
- Create: `docs/stage_reports/STAGE_14_SIMULATOR_DEMO_REPORT.md`
- Modify: `docs/stage_reports/STAGE_13_LIVE_PLANNING_REPORT.md`
- Modify: `README.md`
- Optionally create: `scripts/make_stage14_demo.py` if existing dashboard scripts cannot render the required final video cleanly.

Do not modify these files unless a concrete failure requires it:

- `configs/world_model.yaml`
- `src/world_model/*`
- `scripts/run_planning_agent.py`
- existing Stage 03-12 reports

## Artifact Targets

Final artifacts must use stable paths:

- `data/raw/stage14_live_planner_100/`
- `data/raw/stage14_live_random_100/`
- `data/raw/stage14_live_autopilot_noise_100/`
- `data/raw/stage14_iterative_smoke/`
- `output/world_model/stage14_iterative_smoke/`
- `output/demo/stage14_simulator_demo.mp4`
- `output/demo/stage14_simulator_demo.gif`
- `docs/stage_reports/STAGE_14_SIMULATOR_DEMO_REPORT.md`

## Task 1: Baseline Audit

**Files:**
- Read: `docs/STAGE_13_EPLS_INTEGRATION_PLAN.md`
- Read: `docs/stage_reports/STAGE_13_REPORT.md`
- Read: `docs/stage_reports/STAGE_13_LIVE_PLANNING_REPORT.md`
- Read: `configs/world_model.yaml`
- Read: `scripts/run_planning_agent.py`
- Read: `scripts/iterative_train.py`
- Read: `tests/test_stage13b_workflow.py`

- [ ] **Step 1: Inspect dirty worktree**

Run:

```bash
git status --short
```

Expected: print the current dirty files. Do not revert them.

- [ ] **Step 2: Confirm Stage 13 docs and scripts exist**

Run:

```bash
test -f docs/STAGE_13_EPLS_INTEGRATION_PLAN.md
test -f docs/stage_reports/STAGE_13_REPORT.md
test -f scripts/run_planning_agent.py
test -f scripts/iterative_train.py
test -f tests/test_stage13b_workflow.py
```

Expected: exit code `0`.

- [ ] **Step 3: Run the Stage 13 test gate**

Run:

```bash
python3 -m pytest -q tests/test_world_model_reward.py tests/test_vae.py tests/test_mdrnn.py tests/test_planners.py tests/test_stage13b_workflow.py
```

Expected: all selected tests pass.

- [ ] **Step 4: If tests fail, stop Stage 14 execution and fix the failing test first**

Use the failing traceback to identify the smallest code path that broke. Re-run the same pytest command after each fix.

## Task 2: Add Explicit Live and Iterative Configs

**Files:**
- Create: `configs/world_model_live.yaml`
- Create: `configs/world_model_iterative.yaml`
- Test: `tests/test_stage13b_workflow.py`

- [ ] **Step 1: Create `configs/world_model_live.yaml`**

Write this file:

```yaml
stage13:
  mode: live_planning
  allow_control_default: false

rollout:
  action_order: [steer, throttle, brake]
  policy: autopilot_noise
  reward_version: carla_progress_v1
  terminal_version: carla_terminal_v1

reward:
  progress_weight: 1.0
  speed_weight: 0.05
  speed_norm_kmh: 40.0
  collision_weight: 5.0
  lane_invasion_penalty: 1.0
  offroad_penalty: 2.0
  steer_delta_weight: 0.02
  stuck_penalty: 0.5

terminal:
  collision_intensity_threshold: 0.75
  stuck_speed_kmh_threshold: 1.0
  stuck_frames_threshold: 20
  lane_invasion_frames_threshold: 5
  offroad_frames_threshold: 5

vae:
  image_size: 64
  channels: 3
  latent_size: 64
  batch_size: 32
  learning_rate: 0.0001
  epochs_smoke: 3
  epochs_final: 50
  kl_weight: 1.0

mdrnn:
  latent_size: 64
  action_size: 3
  hidden_size_smoke: 256
  hidden_size_final: 512
  num_gaussians: 5
  sequence_length_smoke: 100
  sequence_length_final: 250
  batch_size: 1
  learning_rate: 0.001
  epochs_smoke: 3
  epochs_final: 60

planner:
  action_order: [steer, throttle, brake]
  horizon: 20
  generations: 10
  generation_sweep: [5, 10, 15]
  horizon_sweep: [5, 10, 20]
  mutation_std:
    steer: 0.15
    throttle: 0.10
    brake: 0.10
  action_bounds:
    steer: [-1.0, 1.0]
    throttle: [0.0, 1.0]
    brake: [0.0, 1.0]
```

- [ ] **Step 2: Create `configs/world_model_iterative.yaml`**

Write the same content as `configs/world_model_live.yaml`, except:

```yaml
stage13:
  mode: iterative
  allow_control_default: false
```

- [ ] **Step 3: Verify the default config remains offline**

Run:

```bash
python3 - <<'PY'
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("configs/world_model.yaml").read_text(encoding="utf-8"))
assert cfg["stage13"]["mode"] == "offline"
print("world_model.yaml remains offline")
PY
```

Expected output contains:

```text
world_model.yaml remains offline
```

- [ ] **Step 4: Verify live config dry-run**

Run:

```bash
python3 scripts/run_planning_agent.py \
  --config configs/sensor_config_stage13_realistic_lite.yaml \
  --world-model-config configs/world_model_live.yaml \
  --vae-checkpoint output/world_model/vae_realistic_lite/best.pth \
  --mdrnn-checkpoint output/world_model/mdrnn_realistic_lite/best.pth \
  --planner rmhc \
  --policy planner \
  --output-dir output/world_model/stage14_live_config_dryrun \
  --max-frames 10 \
  --device cpu \
  --allow-stage13-control \
  --dry-run \
  --overwrite
```

Expected: command exits `0` and writes `output/world_model/stage14_live_config_dryrun/recording_summary.json`.

## Task 3: Support Split Python Environments in Iterative Orchestration

**Files:**
- Modify: `scripts/iterative_train.py`
- Modify: `tests/test_stage13b_workflow.py`

**Design:** `run_planning_agent.py` may need the CARLA environment, while training and offline planner evaluation need the PyTorch/CUDA environment. Add explicit `--carla-python` and `--train-python` flags. Keep `--python` as a compatibility alias that sets both when the new flags are not provided.

- [ ] **Step 1: Add parser arguments**

Modify `parse_args()` in `scripts/iterative_train.py` so it has these arguments:

```python
parser.add_argument("--python", type=str, default=None, help="Compatibility alias used for both CARLA and training commands")
parser.add_argument("--carla-python", type=str, default=None, help="Python executable for live CARLA rollout collection")
parser.add_argument("--train-python", type=str, default=None, help="Python executable for VAE/MDRNN/offline planner commands")
```

- [ ] **Step 2: Add executable resolver**

Add this function after `stage13_flags()`:

```python
def resolve_python_executables(args: argparse.Namespace) -> tuple[str, str]:
    default_python = args.python or sys.executable
    carla_python = args.carla_python or default_python
    train_python = args.train_python or default_python
    return carla_python, train_python
```

- [ ] **Step 3: Pass CARLA Python to rollout collection**

Change `collect_rollouts()` signature to accept `python_executable: str`:

```python
def collect_rollouts(
    args: argparse.Namespace,
    data_root: Path,
    cycle_index: int,
    policy: str,
    num_rollouts: int,
    vae_ckpt: Path,
    mdrnn_ckpt: Path,
    dry_run: bool,
    python_executable: str,
) -> list[Path]:
```

Inside its `cmd`, replace `args.python` with `python_executable`.

- [ ] **Step 4: Pass training Python to model commands**

At the start of `run_training_cycle()`, add:

```python
carla_python, train_python = resolve_python_executables(args)
```

Pass `python_executable=carla_python` to both `collect_rollouts()` calls.

Replace the first element of `vae_cmd`, `encode_cmd`, `mdrnn_cmd`, and `eval_cmd` with `train_python`.

- [ ] **Step 5: Record Python executables in summary**

Add these fields to `cycle_summary`:

```python
"carla_python": str(carla_python),
"train_python": str(train_python),
```

Add these fields to the top-level `report` in `main()`:

```python
"carla_python": str(resolve_python_executables(args)[0]),
"train_python": str(resolve_python_executables(args)[1]),
```

- [ ] **Step 6: Extend dry-run test coverage**

In `tests/test_stage13b_workflow.py`, add assertions to `test_iterative_train_dry_run()` after reading `summary`:

```python
assert "carla_python" in summary
assert "train_python" in summary
cycle = summary["cycle_summaries"][0]
assert "carla_python" in cycle
assert "train_python" in cycle
```

Add a new test:

```python
def test_iterative_train_split_python_dry_run(tmp_path: Path) -> None:
    sensor_cfg = tmp_path / "sensor.yaml"
    wm_cfg = tmp_path / "world_model.yaml"
    write_sensor_config(sensor_cfg)
    write_world_model_config(wm_cfg, mode="iterative")

    vae = tmp_path / "vae_init.pth"
    mdrnn = tmp_path / "mdrnn_init.pth"
    vae.write_text("stub", encoding="utf-8")
    mdrnn.write_text("stub", encoding="utf-8")

    data_root = tmp_path / "data"
    output_root = tmp_path / "output"

    proc = run(
        [
            sys.executable,
            "scripts/iterative_train.py",
            "--world-model-config",
            str(wm_cfg),
            "--sensor-config",
            str(sensor_cfg),
            "--initial-vae-checkpoint",
            str(vae),
            "--initial-mdrnn-checkpoint",
            str(mdrnn),
            "--cycles",
            "1",
            "--rollouts-per-cycle",
            "1",
            "--frames-per-rollout",
            "20",
            "--baseline-rollouts",
            "1",
            "--data-root",
            str(data_root),
            "--output-root",
            str(output_root),
            "--carla-python",
            "/tmp/carla-python",
            "--train-python",
            "/tmp/train-python",
            "--allow-stage13-control",
            "--dry-run",
        ]
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "/tmp/carla-python scripts/run_planning_agent.py" in proc.stdout
    assert "/tmp/train-python scripts/train_vae.py" in proc.stdout
    assert "/tmp/train-python scripts/encode_rollouts.py" in proc.stdout
    assert "/tmp/train-python scripts/train_mdrnn.py" in proc.stdout
    assert "/tmp/train-python scripts/run_planner.py" in proc.stdout
```

- [ ] **Step 7: Run targeted test**

Run:

```bash
python3 -m pytest -q tests/test_stage13b_workflow.py
```

Expected: all tests in the file pass.

- [ ] **Step 8: Run full Stage 13/14 regression gate**

Run:

```bash
python3 -m pytest -q tests/test_world_model_reward.py tests/test_vae.py tests/test_mdrnn.py tests/test_planners.py tests/test_stage13b_workflow.py
```

Expected: all selected tests pass.

## Task 4: Environment Probe

**Files:**
- Create or update: `docs/stage_reports/STAGE_14_SIMULATOR_DEMO_REPORT.md`

- [ ] **Step 1: Probe Python environments**

Run:

```bash
python3 - <<'PY'
mods = ["torch", "cv2", "numpy", "yaml", "carla"]
for m in mods:
    try:
        mod = __import__(m)
        print(f"base {m}: OK {getattr(mod, '__version__', '')}")
    except Exception as exc:
        print(f"base {m}: FAIL {type(exc).__name__}: {exc}")
PY
for env in carla-client pcdet pad; do
  echo "--- $env"
  for m in carla torch cv2 numpy yaml; do
    echo -n "$m: "
    conda run -n "$env" python -c "import $m; print(getattr($m, '__version__', 'OK'))" 2>&1 || true
  done
done
```

Expected: identify one Python executable for CARLA and one for training. Record the result in the Stage 14 report.

- [ ] **Step 2: Probe CARLA server**

Run:

```bash
pgrep -af 'Carla|CARLA|UE4|CarlaUE4' || true
ss -ltnp 2>/dev/null | rg ':2000|:2001|:2002' || true
```

Expected: CARLA server process and port `2000` are visible before live runs. If not visible, start CARLA using the local CARLA packaged release or ask the user to start it.

- [ ] **Step 3: Verify CARLA client connection**

Run with the CARLA Python executable found in Step 1:

```bash
conda run -n carla-client python scripts/test_carla_connection.py
```

Expected: connection succeeds. If this command fails because the server is not running, record it as an external blocker and do not continue to live execution.

## Task 5: Live Planner Smoke Run

**Files:**
- Read: `configs/sensor_config_stage13_realistic_lite.yaml`
- Read: `configs/world_model_live.yaml`
- Output: `data/raw/stage14_live_planner_100/`
- Output: `output/world_model/stage14_live_planner_100/` if planner metrics are emitted separately

- [ ] **Step 1: Run live planner for 100 frames**

Use the CARLA Python executable that can import both `carla` and enough `torch` to load the VAE/MDRNN. If only CPU torch is available in the CARLA environment, use `--device cpu`.

Run:

```bash
conda run -n carla-client python scripts/run_planning_agent.py \
  --config configs/sensor_config_stage13_realistic_lite.yaml \
  --world-model-config configs/world_model_live.yaml \
  --vae-checkpoint output/world_model/vae_realistic_lite/best.pth \
  --mdrnn-checkpoint output/world_model/mdrnn_realistic_lite/best.pth \
  --planner rmhc \
  --policy planner \
  --output-dir data/raw/stage14_live_planner_100 \
  --max-frames 100 \
  --seed 1400 \
  --device cpu \
  --allow-stage13-control \
  --overwrite
```

Expected: command exits `0`.

- [ ] **Step 2: Verify planner artifacts**

Run:

```bash
test -f data/raw/stage14_live_planner_100/dataset_complete.json
test -f data/raw/stage14_live_planner_100/recording_summary.json
test -f data/raw/stage14_live_planner_100/rollout_manifest.json
test -f data/raw/stage14_live_planner_100/planning_metrics.json
python3 - <<'PY'
import json
from pathlib import Path
root = Path("data/raw/stage14_live_planner_100")
complete = json.loads((root / "dataset_complete.json").read_text(encoding="utf-8"))
summary = json.loads((root / "recording_summary.json").read_text(encoding="utf-8"))
metrics = json.loads((root / "planning_metrics.json").read_text(encoding="utf-8"))
assert complete.get("complete") is True, complete
assert int(summary.get("frames_recorded", 0)) >= 100, summary
assert "policy" in metrics or "planner" in metrics or "frames" in metrics, metrics
print("planner live artifacts verified")
PY
```

Expected output contains:

```text
planner live artifacts verified
```

- [ ] **Step 3: If live planner fails, fix before continuing**

Use this triage order:

1. CARLA server missing or wrong port: start/restart CARLA and retry.
2. Python import error: use the correct env or adjust `--device cpu`.
3. Checkpoint load error: verify checkpoint file and model dimensions.
4. Sensor timeout: lower resolution, lower traffic, increase `sensor_timeout`, retry.
5. CARLA sync failure: ensure synchronous mode is enabled and no stale actors remain.

Record the failing command and stderr in `docs/stage_reports/STAGE_14_SIMULATOR_DEMO_REPORT.md` if the blocker is outside the repo.

## Task 6: Live Baseline Runs

**Files:**
- Output: `data/raw/stage14_live_random_100/`
- Output: `data/raw/stage14_live_autopilot_noise_100/`

- [ ] **Step 1: Run random baseline**

Run:

```bash
conda run -n carla-client python scripts/run_planning_agent.py \
  --config configs/sensor_config_stage13_realistic_lite.yaml \
  --world-model-config configs/world_model_live.yaml \
  --vae-checkpoint output/world_model/vae_realistic_lite/best.pth \
  --mdrnn-checkpoint output/world_model/mdrnn_realistic_lite/best.pth \
  --planner rmhc \
  --policy random \
  --output-dir data/raw/stage14_live_random_100 \
  --max-frames 100 \
  --seed 1401 \
  --device cpu \
  --allow-stage13-control \
  --overwrite
```

Expected: command exits `0`.

- [ ] **Step 2: Run autopilot-noise baseline**

Run:

```bash
conda run -n carla-client python scripts/run_planning_agent.py \
  --config configs/sensor_config_stage13_realistic_lite.yaml \
  --world-model-config configs/world_model_live.yaml \
  --vae-checkpoint output/world_model/vae_realistic_lite/best.pth \
  --mdrnn-checkpoint output/world_model/mdrnn_realistic_lite/best.pth \
  --planner rmhc \
  --policy autopilot_noise \
  --output-dir data/raw/stage14_live_autopilot_noise_100 \
  --max-frames 100 \
  --seed 1402 \
  --device cpu \
  --allow-stage13-control \
  --overwrite
```

Expected: command exits `0`.

- [ ] **Step 3: Verify baseline artifacts**

Run:

```bash
for d in data/raw/stage14_live_random_100 data/raw/stage14_live_autopilot_noise_100; do
  test -f "$d/dataset_complete.json"
  test -f "$d/recording_summary.json"
  test -f "$d/rollout_manifest.json"
  test -f "$d/planning_metrics.json"
done
python3 - <<'PY'
import json
from pathlib import Path
for name in ["stage14_live_random_100", "stage14_live_autopilot_noise_100"]:
    root = Path("data/raw") / name
    complete = json.loads((root / "dataset_complete.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "recording_summary.json").read_text(encoding="utf-8"))
    assert complete.get("complete") is True, name
    assert int(summary.get("frames_recorded", 0)) >= 100, name
print("baseline live artifacts verified")
PY
```

Expected output contains:

```text
baseline live artifacts verified
```

## Task 7: Compare Planner Against Baselines

**Files:**
- Create or update: `docs/stage_reports/STAGE_14_SIMULATOR_DEMO_REPORT.md`

- [ ] **Step 1: Extract comparable metrics**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

rows = []
for name in ["stage14_live_planner_100", "stage14_live_random_100", "stage14_live_autopilot_noise_100"]:
    root = Path("data/raw") / name
    summary = load(root / "recording_summary.json")
    metrics = load(root / "planning_metrics.json")
    wm = summary.get("world_model", {})
    rows.append({
        "run": name,
        "frames_recorded": summary.get("frames_recorded"),
        "episode_reward": wm.get("episode_reward"),
        "done_reason": wm.get("done_reason"),
        "collisions": wm.get("collisions"),
        "planner_metric_keys": sorted(metrics.keys()),
    })

out = Path("output/world_model/stage14_live_comparison.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(json.dumps(rows, indent=2))
print(f"wrote {out}")
PY
```

Expected: `output/world_model/stage14_live_comparison.json` exists.

- [ ] **Step 2: Decide the claim level**

Use the comparison result:

- If planner beats random on episode reward and has no worse collision/stuck outcome, report "planner beats random in a 100-frame simulator smoke run".
- If planner does not beat random, report "live planner executed successfully, but smoke metrics do not yet show performance improvement".
- If autopilot-noise beats planner, report it plainly and recommend data/model scaling before stronger claims.

Do not claim "paper reproduced" from a 100-frame smoke run.

## Task 8: Iterative Collect/Train/Evaluate Smoke Cycle

**Files:**
- Input: `configs/world_model_iterative.yaml`
- Output: `data/raw/stage14_iterative_smoke/`
- Output: `output/world_model/stage14_iterative_smoke/`

- [ ] **Step 1: Run one iterative smoke cycle**

Use the split Python flags from Task 3. Adjust the executable paths based on Task 4 environment probes.

Run:

```bash
conda run -n carla-client python scripts/iterative_train.py \
  --world-model-config configs/world_model_iterative.yaml \
  --sensor-config configs/sensor_config_stage13_realistic_lite.yaml \
  --initial-vae-checkpoint output/world_model/vae_realistic_lite/best.pth \
  --initial-mdrnn-checkpoint output/world_model/mdrnn_realistic_lite/best.pth \
  --planner rmhc \
  --cycles 1 \
  --rollouts-per-cycle 1 \
  --frames-per-rollout 100 \
  --baseline-policy autopilot_noise \
  --baseline-rollouts 1 \
  --data-root data/raw/stage14_iterative_smoke \
  --output-root output/world_model/stage14_iterative_smoke \
  --carla-python "$(conda run -n carla-client which python)" \
  --train-python "$(conda run -n pcdet which python)" \
  --device auto \
  --seed 1410 \
  --allow-stage13-control \
  --overwrite
```

If command substitution through `conda run` is unreliable in the local shell, resolve the paths first:

```bash
CARLA_PY=/home/thuan/miniconda3/envs/carla-client/bin/python
TRAIN_PY=/home/thuan/miniconda3/envs/pcdet/bin/python
conda run -n carla-client python scripts/iterative_train.py \
  --world-model-config configs/world_model_iterative.yaml \
  --sensor-config configs/sensor_config_stage13_realistic_lite.yaml \
  --initial-vae-checkpoint output/world_model/vae_realistic_lite/best.pth \
  --initial-mdrnn-checkpoint output/world_model/mdrnn_realistic_lite/best.pth \
  --planner rmhc \
  --cycles 1 \
  --rollouts-per-cycle 1 \
  --frames-per-rollout 100 \
  --baseline-policy autopilot_noise \
  --baseline-rollouts 1 \
  --data-root data/raw/stage14_iterative_smoke \
  --output-root output/world_model/stage14_iterative_smoke \
  --carla-python "$CARLA_PY" \
  --train-python "$TRAIN_PY" \
  --device auto \
  --seed 1410 \
  --allow-stage13-control \
  --overwrite
```

Expected: command exits `0` and writes `output/world_model/stage14_iterative_smoke/iterative_summary.json`.

- [ ] **Step 2: Verify iterative artifacts**

Run:

```bash
test -f output/world_model/stage14_iterative_smoke/iterative_summary.json
python3 - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path("output/world_model/stage14_iterative_smoke/iterative_summary.json").read_text(encoding="utf-8"))
assert summary["mode"] == "iterative", summary
assert summary["cycles"] == 1, summary
assert len(summary["cycle_summaries"]) == 1, summary
cycle = summary["cycle_summaries"][0]
for key in ["planner_rollouts", "baseline_rollouts", "vae_checkpoint", "mdrnn_checkpoint", "latents_dir", "planning_eval_dir"]:
    assert key in cycle, key
print("iterative smoke verified")
PY
```

Expected output contains:

```text
iterative smoke verified
```

## Task 9: Final Simulator Demo Video

**Files:**
- Read: `data/raw/stage14_live_planner_100/`
- Read: existing segmentation/detection/dashboard artifacts
- Create: `output/demo/stage14_simulator_demo.mp4`
- Create: `output/demo/stage14_simulator_demo.gif`
- Optionally create: `scripts/make_stage14_demo.py`

**Required video content:** The final video must show a CARLA road scene with dynamic actors, ego telemetry, and at least three of these panels:

- RGB camera
- semantic/lane/road visualization
- LiDAR BEV or 3D box visualization
- planner action trace `[steer, throttle, brake]`
- reward/progress/collision status
- VAE reconstruction or dream rollout panel

- [ ] **Step 1: Try existing dashboard script first**

Inspect:

```bash
python3 scripts/make_dashboard.py --help
python3 scripts/dashboard.py --help
```

If an existing script can render from `data/raw/stage14_live_planner_100`, use it. If not, create `scripts/make_stage14_demo.py`.

- [ ] **Step 2: Render MP4**

If using a new script, it must accept:

```text
--rollout-dir data/raw/stage14_live_planner_100
--output-video output/demo/stage14_simulator_demo.mp4
--fps 10
--overwrite
```

Run:

```bash
python3 scripts/make_stage14_demo.py \
  --rollout-dir data/raw/stage14_live_planner_100 \
  --output-video output/demo/stage14_simulator_demo.mp4 \
  --fps 10 \
  --overwrite
```

Expected: `output/demo/stage14_simulator_demo.mp4` exists and has nonzero size.

- [ ] **Step 3: Render GIF preview**

Run:

```bash
mkdir -p output/demo
ffmpeg -y -i output/demo/stage14_simulator_demo.mp4 \
  -vf "fps=10,scale=960:-1:flags=lanczos" \
  output/demo/stage14_simulator_demo.gif
```

Expected: `output/demo/stage14_simulator_demo.gif` exists and has nonzero size.

- [ ] **Step 4: Verify video artifacts**

Run:

```bash
test -s output/demo/stage14_simulator_demo.mp4
test -s output/demo/stage14_simulator_demo.gif
ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 output/demo/stage14_simulator_demo.mp4
```

Expected: `ffprobe` prints a positive duration.

## Task 10: Stage 14 Report and README Update

**Files:**
- Create or update: `docs/stage_reports/STAGE_14_SIMULATOR_DEMO_REPORT.md`
- Modify: `docs/stage_reports/STAGE_13_LIVE_PLANNING_REPORT.md`
- Modify: `README.md`

- [ ] **Step 1: Write Stage 14 report**

Create `docs/stage_reports/STAGE_14_SIMULATOR_DEMO_REPORT.md` with these sections:

```markdown
# STAGE 14 - Simulator Demo Completion

> Date: 2026-05-21
> Status: [VERIFIED / BLOCKED]

## Scope
- Simulator-only CARLA demo.
- No real-road vehicle control.
- No production autonomous-driving claim.

## Commands Run

## Environment Probe

## Live Planner Run

## Baseline Runs

## Planner vs Baseline Comparison

## Iterative Smoke Cycle

## Demo Video

## Verification

## Limitations

## Next Scale-Up Steps
```

Fill each section with exact commands, outputs, metrics, and artifact paths. If any required live command failed due to external setup, set `Status: BLOCKED` and include the exact blocker.

- [ ] **Step 2: Update Stage 13B report**

In `docs/stage_reports/STAGE_13_LIVE_PLANNING_REPORT.md`, add a short "Stage 14 follow-up" section linking to the Stage 14 report and stating whether live CARLA execution was verified.

- [ ] **Step 3: Update README demo section**

In `README.md`, add or update a short Stage 14 simulator demo note:

```markdown
### Stage 14 Simulator Demo

Stage 14 is a simulator-only CARLA demo that combines dynamic traffic/pedestrians, perception visualization, and optional Stage 13 EPLS live planning. It is not a real-road or production autonomous-driving system.

Report: `docs/stage_reports/STAGE_14_SIMULATOR_DEMO_REPORT.md`
Demo video: `output/demo/stage14_simulator_demo.mp4`
```

Only add the demo video line if the file exists.

## Task 11: Final Verification Gate

**Files:**
- Read all changed files
- Verify all required artifacts

- [ ] **Step 1: Run regression tests**

Run:

```bash
python3 -m pytest -q tests/test_world_model_reward.py tests/test_vae.py tests/test_mdrnn.py tests/test_planners.py tests/test_stage13b_workflow.py
```

Expected: all selected tests pass.

- [ ] **Step 2: Verify live and demo artifacts**

Run:

```bash
test -f data/raw/stage14_live_planner_100/dataset_complete.json
test -f data/raw/stage14_live_random_100/dataset_complete.json
test -f data/raw/stage14_live_autopilot_noise_100/dataset_complete.json
test -f output/world_model/stage14_live_comparison.json
test -f output/world_model/stage14_iterative_smoke/iterative_summary.json
test -s output/demo/stage14_simulator_demo.mp4
test -s output/demo/stage14_simulator_demo.gif
test -f docs/stage_reports/STAGE_14_SIMULATOR_DEMO_REPORT.md
```

Expected: exit code `0`.

- [ ] **Step 3: Verify report does not overclaim**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
report = Path("docs/stage_reports/STAGE_14_SIMULATOR_DEMO_REPORT.md").read_text(encoding="utf-8").lower()
for forbidden in ["real-road deployment", "production autonomous", "paper reproduced", "high accuracy"]:
    assert forbidden not in report, forbidden
assert "simulator-only" in report
print("report claim check passed")
PY
```

Expected output contains:

```text
report claim check passed
```

- [ ] **Step 4: Capture final git status**

Run:

```bash
git status --short
```

Expected: review changed files and mention which files were modified by the agent.

## Completion Definition

Stage 14 is complete only when all of these are true:

- The selected pytest command passes in the final verification gate.
- `run_planning_agent.py` completes a live CARLA planner run for at least 100 frames.
- Random and autopilot-noise live baselines complete with the same frame target.
- `output/world_model/stage14_live_comparison.json` compares planner and baselines.
- One iterative collect/train/evaluate smoke cycle completes, or a non-repo external blocker is documented with stderr and environment probe.
- `output/demo/stage14_simulator_demo.mp4` and `.gif` exist and are playable.
- `docs/stage_reports/STAGE_14_SIMULATOR_DEMO_REPORT.md` records commands, metrics, artifacts, limitations, and simulator-only scope.

If any item is missing, report Stage 14 as `BLOCKED` or `PARTIAL`, not complete.

## Agent Handoff Prompt

Use this prompt for future agents:

```text
You are working in `/media/thuan/Workspace/autonomous-paper/carla-perception-lab`.

Read and execute `docs/STAGE_14_SIMULATOR_DEMO_COMPLETION_PLAN.md` task-by-task.

Do not claim completion without running the verification commands in Task 11.
Do not modify `configs/world_model.yaml` away from offline mode.
Keep the work simulator-only: no real-road control, no production autonomous-driving claim.
If CARLA server or environment setup blocks live execution, document the exact command, stderr, and environment probe in `docs/stage_reports/STAGE_14_SIMULATOR_DEMO_REPORT.md`.

Final response format:
- Summary
- Verification
- Artifacts
- Metrics
- Limitations
- Next steps
```

