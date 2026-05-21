from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_stage14_demo.py"

spec = importlib.util.spec_from_file_location("make_stage14_demo", SCRIPT)
assert spec is not None and spec.loader is not None
make_stage14_demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_stage14_demo)


def test_render_semantic_colorizes_carla_raw_tag_images() -> None:
    raw = np.zeros((2, 3, 3), dtype=np.uint8)
    raw[:, :1, 2] = 7   # road tag in CARLA raw semantic image red channel
    raw[:, 1:2, 2] = 10  # vehicle tag
    raw[:, 2:, 2] = 12  # pedestrian tag in this repo metadata

    rendered = make_stage14_demo.render_semantic(raw, (3, 2))

    assert rendered.shape == (2, 3, 3)
    assert int(rendered.max()) > 64
    assert not np.array_equal(rendered[0, 0], rendered[0, 1])
    assert not np.array_equal(rendered[0, 1], rendered[0, 2])


def test_visual_quality_gate_rejects_single_tag_stuck_rollout() -> None:
    semantic_tags = [np.full((4, 4), 14, dtype=np.uint8) for _ in range(3)]
    metadata = [
        {
            "actors": [],
            "world_model": {"done_reason": "stuck"},
            "telemetry": {"stuck": True, "progress_m": 0.0},
        }
        for _ in range(3)
    ]

    report = make_stage14_demo.evaluate_visual_quality(semantic_tags, metadata)
    failures = make_stage14_demo.check_visual_quality(
        report,
        min_semantic_tags=3,
        require_road=True,
        min_vehicle_frames=1,
        min_walker_frames=1,
        reject_stuck=True,
    )

    assert failures
    assert any("semantic" in item for item in failures)
    assert any("road" in item for item in failures)
    assert any("vehicle" in item for item in failures)
    assert any("walker" in item for item in failures)
    assert any("stuck" in item for item in failures)


def test_lidar_bev_draws_actor_metadata_overlay() -> None:
    points = np.empty((0, 4), dtype=np.float32)
    meta = {
        "ego_vehicle": {
            "location": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"yaw": 0.0},
        },
        "actors": [
            {
                "class_name": "Vehicle",
                "type_id": "vehicle.tesla.model3",
                "distance_to_ego_m": 10.0,
                "location": {"x": 10.0, "y": 0.0, "z": 0.0},
            },
            {
                "class_name": "Pedestrian",
                "type_id": "walker.pedestrian.0001",
                "distance_to_ego_m": 12.0,
                "location": {"x": 12.0, "y": 5.0, "z": 0.0},
            },
        ],
    }

    bev = make_stage14_demo.render_lidar_bev(points, (220, 140), meta=meta)

    vehicle_color = np.array(make_stage14_demo.ACTOR_COLORS_BGR["vehicle"], dtype=np.uint8)
    walker_color = np.array(make_stage14_demo.ACTOR_COLORS_BGR["walker"], dtype=np.uint8)
    assert np.any(np.all(bev == vehicle_color, axis=2))
    assert np.any(np.all(bev == walker_color, axis=2))
