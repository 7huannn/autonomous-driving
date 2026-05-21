# STAGE 14 - Simulator Demo Completion

> Date: 2026-05-21  
> Status: PARTIAL - smoke/demo artifacts verified; single Town10+traffic+walker hero demo blocked by CARLA Vulkan OOM

## Scope
- simulator-only CARLA demo.
- No real-road vehicle control.
- No production-driving claim.

## Commands Run
### Pre-fix test gate
```bash
python3 -m pytest -q tests/test_stage13b_workflow.py
python3 -m pytest -q tests/test_world_model_reward.py tests/test_vae.py tests/test_mdrnn.py tests/test_planners.py tests/test_stage13b_workflow.py
```
Result:
- `tests/test_stage13b_workflow.py`: `3 passed`
- regression gate: `13 passed`

### Environment probe
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
pgrep -af 'Carla|CARLA|UE4|CarlaUE4' || true
ss -ltnp 2>/dev/null | rg ':2000|:2001|:2002' || true
conda run -n carla-client python scripts/test_carla_connection.py
```
Connection result:
- first attempt: `[FAIL] Connection refused`
- CARLA startup command:
```bash
docker run -d --rm --name carla-stage14 --net=host --gpus all carlasim/carla:0.10.0 \
  bash CarlaUnreal.sh -RenderOffScreen -nosound -quality-level=Low -ResX=1280 -ResY=720
```
- retry: `[PASS] CARLA connection established`

### Stage 14 live runs
Planner:
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

Random baseline:
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

Autopilot-noise baseline:
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

### Stage 14 iterative smoke
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
  --carla-python /home/thuan/miniconda3/envs/carla-client/bin/python \
  --train-python /home/thuan/miniconda3/envs/pcdet/bin/python \
  --device auto \
  --seed 1410 \
  --allow-stage13-control \
  --overwrite
```
Result:
- first attempt failed with:
  - `ValueError: No sequences available for MDRNN dataset`
- fix: cap MDRNN sequence length in `scripts/iterative_train.py` to `min(sequence_length_smoke, frames_per_rollout - 1)`
- rerun passed: `[PASS] iterative_train complete`

### Stage 14 comparison and demo rendering
```bash
python3 scripts/analyze_stage14_runs.py \
  --planner-dir data/raw/stage14_live_planner_100 \
  --random-dir data/raw/stage14_live_random_100 \
  --autopilot-noise-dir data/raw/stage14_live_autopilot_noise_100 \
  --output-json output/world_model/stage14_live_comparison.json \
  --output-md output/world_model/stage14_live_comparison.md
```

```bash
python3 scripts/make_stage14_demo.py \
  --rollout-dir data/raw/stage14_live_planner_100 \
  --output-video output/demo/stage14_simulator_demo.mp4 \
  --fps 10 \
  --overwrite
```

GIF conversion attempt:
```bash
ffmpeg -y -i output/demo/stage14_simulator_demo.mp4 \
  -vf "fps=10,scale=960:-1:flags=lanczos" \
  output/demo/stage14_simulator_demo.gif
```
Result:
- failed: `/bin/bash: ffmpeg: command not found`
- fallback: Python/PIL GIF conversion produced `output/demo/stage14_simulator_demo.gif`

## Environment Probe
- Base `python3`:
  - `torch 2.10.0+cu128`, `cv2 4.13.0`, `numpy 2.3.4`, `yaml 6.0.3`, `carla` missing.
- `carla-client`:
  - `carla OK`, `torch 2.12.0+cpu`, `cv2 4.13.0`, `numpy 2.2.6`, `yaml 6.0.3`.
- `pcdet`:
  - `carla` missing, `torch 1.13.1+cu116`, `cv2 4.13.0`, `numpy 1.24.4`, `yaml 6.0.3`.
- `pad`:
  - `carla` missing, `torch 1.12.1+cu116`, `cv2 4.5.4-dev`, `numpy 1.23.5`, `yaml 6.0.3`.

## Live Planner Run
- Artifacts:
  - `data/raw/stage14_live_planner_100/dataset_complete.json`
  - `data/raw/stage14_live_planner_100/recording_summary.json`
  - `data/raw/stage14_live_planner_100/rollout_manifest.json`
  - `data/raw/stage14_live_planner_100/planning_metrics.json`
- Key values from `recording_summary.json`:
  - `frames_recorded=100`
  - `episode_reward=-2.0713543493020588`
  - `done_reason=stuck`

## Baseline Runs
- `data/raw/stage14_live_random_100/recording_summary.json`:
  - `frames_recorded=100`
  - `episode_reward=-11.633845462228424`
  - `done_reason=rollout_end`
- `data/raw/stage14_live_autopilot_noise_100/recording_summary.json`:
  - `frames_recorded=100`
  - `episode_reward=57.38041788397685`
  - `done_reason=rollout_end`

## Planner vs Baseline Comparison
From:
- `output/world_model/stage14_live_comparison.json`
- `output/world_model/stage14_live_comparison.md`

Planner (`stage14_live_planner_100`):
- `policy=planner`
- `progress_total_m=0.8645300944512573`
- `distance_traveled_m=9.34575947454663`
- `avg_speed_kmh=3.6102921489260784`
- `collision_sum_intensity=0.0`
- `stuck_frames=23`
- `stuck_ratio=0.23`

Random (`stage14_live_random_100`):
- `policy=random`
- `progress_total_m=0.35131990439916777`
- `distance_traveled_m=1.2216739047832557`
- `avg_speed_kmh=0.4671231564697165`
- `collision_sum_intensity=0.0`
- `stuck_frames=23`
- `stuck_ratio=0.23`

Autopilot-noise (`stage14_live_autopilot_noise_100`):
- `policy=autopilot_noise`
- `progress_total_m=1.7520021182040322`
- `distance_traveled_m=54.93522047440136`
- `avg_speed_kmh=20.102702161354372`
- `collision_sum_intensity=0.0`
- `stuck_frames=0`
- `stuck_ratio=0.0`

Aggregate decisions:
- `beats_random_by_reward=true`
- `beats_random_by_progress=true`
- `beats_random_by_safety=true`
- `beats_autopilot_noise_by_reward=false`
- `beats_autopilot_noise_by_progress=false`
- `planner_stuck=true`
- Decision summary: planner beats random by reward but still gets stuck.

## Iterative Smoke Cycle
- Artifact:
  - `output/world_model/stage14_iterative_smoke/iterative_summary.json`
- Final model artifacts:
  - `output/world_model/stage14_iterative_smoke/cycle_000/vae/best.pth`
  - `output/world_model/stage14_iterative_smoke/cycle_000/mdrnn/best.pth`
- Planning eval:
  - `output/world_model/stage14_iterative_smoke/cycle_000/planning_eval/planner_metrics.json`
  - `episodes=20`
  - `mean_planner_pred_reward=-0.02807546965777874`
  - `mean_random_pred_reward=-0.028447104210499674`
  - `planner_beats_random=true`

## Visual QA Follow-up
The original planner hero demo (`output/demo/stage14_simulator_demo.mp4`) is rejected as a final visual demo:
- it uses `Mine_01`, not a normal road/lane scene,
- the planner rollout ends with `done_reason=stuck`,
- its raw semantic frames contain only tag `14`,
- it has no vehicle/walker actor metadata in the planner rollout,
- the first renderer treated raw CARLA semantic tags as BGR images, producing a near-black semantic panel.

Quality-gate command:
```bash
python3 scripts/make_stage14_demo.py \
  --rollout-dir data/raw/stage14_live_planner_100 \
  --output-video output/demo/stage14_simulator_demo_rejected.mp4 \
  --fps 10 \
  --min-semantic-tags 3 \
  --require-road \
  --min-vehicle-frames 1 \
  --min-walker-frames 1 \
  --reject-stuck \
  --quality-report-json output/demo/stage14_simulator_demo_rejected_quality.json \
  --overwrite
```
Result:
- failed as intended with:
  - `semantic tag diversity too low: 1 < 3`
  - `road/roadline semantic evidence missing`
  - `vehicle metadata evidence frames too low: 0 < 1`
  - `walker metadata evidence frames too low: 0 < 1`
  - `rollout marked stuck in 23 frame(s)`

Renderer fixes:
- `scripts/make_stage14_demo.py` now colorizes CARLA raw semantic tag images from the red channel.
- The status panel now reports vehicle/walker/cyclist metadata counts and nearest actors.
- The LiDAR BEV panel now overlays metadata actors: green vehicles, red pedestrians, yellow cyclists.
- Optional visual quality gates can fail bad demos before rendering or publishing.

### Replacement Visual Artifacts
No single artifact in the current workspace satisfies road/lane + vehicles + pedestrians in one Town10 live rollout. The replacement outputs are therefore split by evidence type:

Road/lane visual preview:
- `output/demo/stage14_town10_road_lane_preview.mp4`
- `output/demo/stage14_town10_road_lane_preview.gif`
- `output/demo/stage14_town10_road_lane_preview_sheet.jpg`
- `output/demo/stage14_town10_road_lane_preview_quality.json`
- Quality: `semantic_tags=[1,2,3,5,6,7,8,9,11,14,15,18,19,20,21,22,23,24,25]`, `road_frames=18`, `vehicle_evidence_frames=0`, `walker_evidence_frames=0`, `stuck_frames=0`

Actor-rich visual/BEV preview:
- `output/demo/stage14_actor_rich_visual_demo.mp4`
- `output/demo/stage14_actor_rich_visual_demo.gif`
- `output/demo/stage14_actor_rich_visual_demo_sheet.jpg`
- `output/demo/stage14_actor_rich_visual_demo_quality.json`
- Quality: `semantic_tags=[0,11,14,15,25]`, `road_frames=0`, `vehicle_evidence_frames=41`, `walker_evidence_frames=41`, `stuck_frames=0`

Live Stage 14 autopilot-noise replacement preview:
- `output/demo/stage14_live_autopilot_noise_visual.mp4`
- `output/demo/stage14_live_autopilot_noise_visual.gif`
- `output/demo/stage14_live_autopilot_noise_visual_sheet.jpg`
- `output/demo/stage14_live_autopilot_noise_visual_quality.json`
- Quality: `semantic_tags=[0,11,25]`, `road_frames=0`, `vehicle_evidence_frames=60`, `walker_evidence_frames=100`, `stuck_frames=0`

### Town10 Traffic Blocker
The local `carlasim/carla:0.10.0` image only exposes:
- `Mine_01.umap`
- `Town10HD_Opt.umap`

Attempted a lightweight Town10 traffic capture:
```bash
docker run -d --rm --name carla-stage14-fix --net=host --gpus all carlasim/carla:0.10.0 \
  bash CarlaUnreal.sh -RenderOffScreen -nosound -quality-level=Low -ResX=640 -ResY=360

conda run -n carla-client python scripts/carla_recorder.py \
  --config configs/sensor_config_stage14_city_scout.yaml \
  --output-dir data/raw/stage14_town10_traffic_scout_60 \
  --num-frames 60 \
  --width 160 \
  --height 90 \
  --traffic-vehicles 10 \
  --traffic-walkers 8 \
  --lidar-pps 8000 \
  --actor-label-max-distance 80 \
  --seed 1450 \
  --policy autopilot_noise \
  --overwrite
```
Result:
- client aborted with `std::exception`.
- CARLA Docker log:
  - `Out of memory on Vulkan; MemoryTypeIndex=1, AllocSize=13.234MB`
- no usable frames were produced in `data/raw/stage14_town10_traffic_scout_60`.

Conclusion:
- Do not use `output/demo/stage14_simulator_demo.mp4` as final hero evidence.
- Current environment cannot yet produce the desired single high-quality Town10 road/lane + vehicle + pedestrian live demo.
- A stronger final demo requires resolving the CARLA Town10 Vulkan memory/runtime instability or using a different map/runtime profile that supports both lane-rich roads and dynamic actors.

## Demo Video

`scripts/make_stage14_demo.py` renders:
- CARLA RGB camera,
- CARLA semantic camera visualization,
- raw LiDAR BEV projection with metadata actor overlay,
- telemetry/control overlays from metadata.

This demo video is a simulator visualization artifact. It does not by itself validate learned ERFNet or PointPillar model accuracy.

## Metric Quality Audit
References:
- CARLA Leaderboard evaluation metrics: https://leaderboard.carla.org/evaluation_v2_0/
- CARLA benchmark metrics: https://carla.readthedocs.io/en/stable/benchmark_metrics/
- EPLS paper: https://arxiv.org/pdf/2011.11293
- Cityscapes benchmark metrics: https://www.cityscapes-dataset.com/benchmarks/
- nuScenes detection metrics: https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/README.md

### 1) Are current Stage 14 live metrics enough for driving quality?
No. Current Stage 14 metrics are smoke-level diagnostics only. They include reward/progress/speed/collision proxies, but they do not implement the full route-completion and infraction-penalty framework used by CARLA Leaderboard-style driving evaluation.

### 2) Are they enough for EPLS paper-faithful evaluation?
No. EPLS-style claims need broader live evaluation over many episodes and seeds, planning hyperparameter sweeps, and iterative improvement tracking against baselines. Current Stage 14 runs are short smoke runs (100 frames each) and one iterative cycle.

### 3) Are they enough for perception accuracy?
No. The Stage 14 live metrics here are control-rollout metrics, not perception accuracy metrics. Perception requires benchmark metrics such as Cityscapes IoU/iIoU style segmentation measures and detection metrics such as AP/mAP with additional error terms (for example nuScenes-style TP error metrics and NDS-style aggregation).

### 4) What metric suite should be used next?
For driving quality:
- route completion,
- infraction penalty and per-infraction rates,
- collision/offroad/lane/stuck/blocked metrics,
- multi-seed and multi-route averages with variance.

For EPLS evaluation:
- live reward over many episodes/seeds,
- planner vs random and autopilot baselines,
- horizon/generation sweeps,
- iterative collect/train/evaluate trend across cycles,
- held-out world-model prediction losses (latent/reward/terminal).

For perception:
- segmentation: mIoU-class/mIoU-category and iIoU where applicable,
- detection: AP/mAP plus 3D error decomposition metrics (translation/scale/orientation/velocity/attribute style, depending on benchmark protocol).

Conclusion:
- current Stage 14 metrics support smoke validation only.
- demo video and GIF are not accuracy metrics.

## Verification
Post-fix test gates:
```bash
python3 -m py_compile scripts/make_stage14_demo.py
python3 -m pytest -q tests/test_stage14_demo_rendering.py
python3 -m pytest -q tests/test_stage13b_workflow.py
python3 -m pytest -q tests/test_world_model_reward.py tests/test_vae.py tests/test_mdrnn.py tests/test_planners.py tests/test_stage13b_workflow.py tests/test_stage14_demo_rendering.py
```
Results:
- `tests/test_stage14_demo_rendering.py`: `3 passed`
- `tests/test_stage13b_workflow.py`: `6 passed`
- regression gate including Stage 14 renderer tests: `19 passed`

Output checks:
```bash
test -f output/world_model/stage14_live_comparison.json
test -f output/world_model/stage14_live_comparison.md
test -f docs/stage_reports/STAGE_14_SIMULATOR_DEMO_REPORT.md
```
Additional visual artifact gate:
- verified all replacement MP4/GIF/contact-sheet/quality JSON outputs exist and are non-empty.
- verified replacement quality JSON files have `quality_passed=true`.
- verified rejected planner demo quality JSON has `quality_passed=false`.
- command output: `stage14 visual artifact gate passed`

Claim check:
- Executed the verification-gate claim-check command and confirmed output:
  - `report claim check passed`

## Limitations
- live results are still smoke-scale.
- no ffmpeg/ffprobe binaries are available in this shell image.
- report claims remain intentionally bounded to simulator smoke/demo evidence.
- no current single demo video satisfies all desired visual qualities; the split artifacts are QA evidence, not a paper-ready hero demo.
