# STAGE 13 — Evolutionary Planning in Latent Space (Stage 13A)

> Date: 2026-05-19  
> Status: IMPLEMENTED (offline world-model path)

## Scope Mode
- `stage13.mode`: `offline` (default)
- `live_planning` and `iterative` are gated by `--allow-stage13-control` and are not run in this report.

## Implemented Components
- Rollout recording extension with per-frame `control`, `telemetry`, and `world_model` (`reward`, `done`, `done_reason`).
- Rollout collection orchestrator with retry/resume and rollout manifests.
- ConvVAE module, dataset, trainer, and latent encoder CLI.
- MDRNN module, sequence dataset, MDN loss, trainer, and dream rollout CLI.
- RMHC + RHEA planners with offline planner-vs-random evaluation.
- Stage 13 dashboard extension (`world_model` mode) and latent-space PCA visualization.

## Artifacts
- Config: `configs/world_model.yaml`
- Core package: `src/world_model/*`
- CLIs:
  - `scripts/collect_rollouts.py`
  - `scripts/train_vae.py`
  - `scripts/encode_rollouts.py`
  - `scripts/train_mdrnn.py`
  - `scripts/dream_rollout.py`
  - `scripts/run_planner.py`
  - `scripts/visualize_latent_space.py`
- Tests:
  - `tests/test_world_model_reward.py`
  - `tests/test_vae.py`
  - `tests/test_mdrnn.py`
  - `tests/test_planners.py`

## Executed Smoke Run (2026-05-19)
- CARLA recorder smoke on `Mine_01` completed (`20/20` frames):
  - `data/raw/stage13_smoke_contract_20`
- Stage 13 rollout collection completed (`5/5` rollouts, first-attempt success):
  - `data/raw/stage13_rollouts_smoke/rollout_00000` ... `rollout_00004`
  - per rollout: `150` frames
  - total aligned frames: `750`
  - missing aligned frames: `0`

## Metrics (Observed)

### Dataset / latent cache
- Latent manifest: `output/world_model/latents_smoke_stage13/latent_manifest.json`
- `num_rollouts`: `5`
- `total_frames`: `750`

### VAE
- Metrics: `output/world_model/vae_smoke_stage13/training_metrics.json`
- Checkpoint: `output/world_model/vae_smoke_stage13/best.pth`
- Epochs: `3` (smoke)
- `best_val_total`: `0.0506786832`
- Final val reconstruction MSE: `0.0505984343`
- Final val KL: `0.0000802494`

### MDRNN
- Metrics: `output/world_model/mdrnn_smoke_stage13/training_metrics.json`
- Checkpoint: `output/world_model/mdrnn_smoke_stage13/best.pth`
- Epochs: `3` (smoke)
- `best_val_total`: `-227.6749688721`
- Final val GMM-NLL: `-227.6874481201`
- Final val reward MSE: `0.0110728270`
- Final val done BCE: `0.0014058757`

### Dream rollout
- Summary: `output/world_model/dream_smoke_stage13/dream_summary.json`
- Video: `output/world_model/dream_smoke_stage13/dream_rollout.mp4`
- Steps: `30`
- Avg predicted reward: `0.8531245401`
- Avg predicted done probability: `0.0022952196`

### Planner (offline latent-space)
- RMHC metrics: `output/world_model/planning_smoke_stage13_rmhc/planner_metrics.json`
  - `episodes`: `20`
  - `mean_planner_pred_reward`: `16.8634038374`
  - `mean_random_pred_reward`: `16.8606167983`
  - `planner_beats_random`: `true`
- RHEA metrics: `output/world_model/planning_smoke_stage13_rhea/planner_metrics.json`
  - `episodes`: `20`
  - `mean_planner_pred_reward`: `17.0078028679`
  - `mean_random_pred_reward`: `16.8578677736`
  - `planner_beats_random`: `true`

### Visualization
- Latent PCA image: `output/world_model/latent_smoke_stage13.png`
- Latent PCA stats: `output/world_model/latent_smoke_stage13.json`

### Hardware note
- Snapshot VRAM after run: `609 MiB / 8188 MiB` (`nvidia-smi` query)

## Limitations
- Stage 13A is offline-only; no simulator closed-loop control claims.
- CARLA reward/terminal are engineered signals and remain task-specific.
- Planner quality is bounded by MDRNN model fidelity.
- Running CARLA simulator concurrently with world-model dream/planner GPU inference can trigger cuDNN failures on this host. Stage 13 smoke run keeps CARLA and model inference separated.
