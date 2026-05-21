from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_sensor_config(path: Path) -> None:
    path.write_text(
        """
carla:
  host: localhost
  port: 2000
  map: Mine_01
  timeout: 2.0
  tm_port: 10000
sensors:
  rgb_camera:
    blueprint: sensor.camera.rgb
    transform: {x: 1.5, y: 0.0, z: 2.4}
    attributes: {image_size_x: 320, image_size_y: 180, fov: 90}
  semantic_camera:
    blueprint: sensor.camera.semantic_segmentation
    transform: {x: 1.5, y: 0.0, z: 2.4}
    attributes: {image_size_x: 320, image_size_y: 180, fov: 90}
  lidar:
    blueprint: sensor.lidar.ray_cast
    transform: {x: 0.0, y: 0.0, z: 2.5}
    attributes: {points_per_second: 32000, rotation_frequency: 10, channels: 32, range: 80}
recording:
  output_dir: data/raw/test
  num_frames: 20
  use_synchronous_mode: true
  fixed_delta_seconds: 0.1
  sensor_timeout: 2.0
  warmup_ticks: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )


def write_world_model_config(path: Path, mode: str) -> None:
    path.write_text(
        f"""
stage13:
  mode: {mode}
  allow_control_default: false
rollout:
  action_order: [steer, throttle, brake]
  reward_version: carla_progress_v1
  terminal_version: carla_terminal_v1
reward:
  progress_weight: 1.0
terminal:
  collision_intensity_threshold: 0.75
vae:
  image_size: 64
  channels: 3
  latent_size: 64
mdrnn:
  latent_size: 64
  action_size: 3
  hidden_size_smoke: 256
  hidden_size_final: 512
  num_gaussians: 5
planner:
  action_order: [steer, throttle, brake]
  horizon: 5
  generations: 3
  mutation_std:
    steer: 0.15
    throttle: 0.10
    brake: 0.10
  action_bounds:
    steer: [-1.0, 1.0]
    throttle: [0.0, 1.0]
    brake: [0.0, 1.0]
""".strip()
        + "\n",
        encoding="utf-8",
    )


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=False)


def test_run_planning_agent_dry_run(tmp_path: Path) -> None:
    sensor_cfg = tmp_path / "sensor.yaml"
    wm_cfg = tmp_path / "world_model.yaml"
    write_sensor_config(sensor_cfg)
    write_world_model_config(wm_cfg, mode="live_planning")

    vae = tmp_path / "vae.pth"
    mdrnn = tmp_path / "mdrnn.pth"
    vae.write_text("stub", encoding="utf-8")
    mdrnn.write_text("stub", encoding="utf-8")

    out_dir = tmp_path / "live_rollout"
    proc = run(
        [
            sys.executable,
            "scripts/run_planning_agent.py",
            "--config",
            str(sensor_cfg),
            "--world-model-config",
            str(wm_cfg),
            "--vae-checkpoint",
            str(vae),
            "--mdrnn-checkpoint",
            str(mdrnn),
            "--output-dir",
            str(out_dir),
            "--allow-stage13-control",
            "--dry-run",
        ]
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout

    summary_path = out_dir / "recording_summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["dry_run"] is True
    assert summary["stage13_mode"] == "live_planning"


def test_iterative_train_dry_run(tmp_path: Path) -> None:
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
            "--allow-stage13-control",
            "--dry-run",
        ]
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout

    summary_path = output_root / "iterative_summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["mode"] == "iterative"
    assert summary["dry_run"] is True
    assert "carla_python" in summary
    assert "train_python" in summary
    assert len(summary["cycle_summaries"]) == 1
    cycle = summary["cycle_summaries"][0]
    assert "carla_python" in cycle
    assert "train_python" in cycle


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


def test_iterative_train_overwrite_restarts_rollout_ids(tmp_path: Path) -> None:
    sensor_cfg = tmp_path / "sensor.yaml"
    wm_cfg = tmp_path / "world_model.yaml"
    write_sensor_config(sensor_cfg)
    write_world_model_config(wm_cfg, mode="iterative")

    vae = tmp_path / "vae_init.pth"
    mdrnn = tmp_path / "mdrnn_init.pth"
    vae.write_text("stub", encoding="utf-8")
    mdrnn.write_text("stub", encoding="utf-8")

    test_root = ROOT / ".tmp_stage13b_tests" / tmp_path.name
    data_root = test_root / "data"
    output_root = test_root / "output"
    data_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    (data_root / "rollout_00005").mkdir(parents=True, exist_ok=True)
    (output_root / "cycle_123").mkdir(parents=True, exist_ok=True)
    (output_root / "iterative_summary.json").write_text("{}", encoding="utf-8")
    try:
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
                "--allow-stage13-control",
                "--dry-run",
                "--overwrite",
            ]
        )
        assert proc.returncode == 0, proc.stderr or proc.stdout
        assert str(data_root / "rollout_00000") in proc.stdout
        assert not (data_root / "rollout_00005").exists()
        assert not (output_root / "cycle_123").exists()
    finally:
        if test_root.exists():
            shutil.rmtree(test_root)


def test_iterative_train_without_overwrite_keeps_append_behavior(tmp_path: Path) -> None:
    sensor_cfg = tmp_path / "sensor.yaml"
    wm_cfg = tmp_path / "world_model.yaml"
    write_sensor_config(sensor_cfg)
    write_world_model_config(wm_cfg, mode="iterative")

    vae = tmp_path / "vae_init.pth"
    mdrnn = tmp_path / "mdrnn_init.pth"
    vae.write_text("stub", encoding="utf-8")
    mdrnn.write_text("stub", encoding="utf-8")

    test_root = ROOT / ".tmp_stage13b_tests" / tmp_path.name
    data_root = test_root / "data"
    output_root = test_root / "output"
    data_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    (data_root / "rollout_00005").mkdir(parents=True, exist_ok=True)
    try:
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
                "--allow-stage13-control",
                "--dry-run",
            ]
        )
        assert proc.returncode == 0, proc.stderr or proc.stdout
        assert str(data_root / "rollout_00006") in proc.stdout
        assert (data_root / "rollout_00005").exists()
    finally:
        if test_root.exists():
            shutil.rmtree(test_root)


def test_iterative_train_overwrite_refuses_paths_outside_repo(tmp_path: Path) -> None:
    sensor_cfg = tmp_path / "sensor.yaml"
    wm_cfg = tmp_path / "world_model.yaml"
    write_sensor_config(sensor_cfg)
    write_world_model_config(wm_cfg, mode="iterative")

    vae = tmp_path / "vae_init.pth"
    mdrnn = tmp_path / "mdrnn_init.pth"
    vae.write_text("stub", encoding="utf-8")
    mdrnn.write_text("stub", encoding="utf-8")

    data_root = tmp_path / "outside_data"
    output_root = tmp_path / "outside_output"
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
            "--allow-stage13-control",
            "--dry-run",
            "--overwrite",
        ]
    )
    assert proc.returncode != 0
    combined = f"{proc.stdout}\n{proc.stderr}".lower()
    assert "is outside" in combined
