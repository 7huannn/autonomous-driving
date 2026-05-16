#!/usr/bin/env python3
"""Deterministic score-transform recovery over existing prediction txt artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


@dataclass
class PredRecord:
    file_name: str
    line_idx: int
    cls: str
    coords: list[float]
    score: float
    raw_tokens: list[str]

    @property
    def sig(self) -> tuple[str, tuple[float, ...]]:
        return (self.cls, tuple(round(v, 6) for v in self.coords))


def resolve_path(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Controlled score-transform search.")
    ap.add_argument("--experiment-id", default="EXP_SCORE_TRANSFORM_RECOVERY_V3")
    ap.add_argument(
        "--input-pred-dir",
        default="output/detection_3d_canonical_mix_v3/predictions_fusion_e7v_e5p_rescore",
    )
    ap.add_argument(
        "--target-pred-dir",
        default="output/detection_3d_canonical_mix_v3/predictions_fusion_rescore_best",
    )
    ap.add_argument(
        "--output-pred-dir",
        default="output/detection_3d_EXP_SCORE_TRANSFORM_RECOVERY_V3/predictions_transformed",
    )
    ap.add_argument(
        "--summary-json",
        default="outputs/benchmarks/score_transform_search_EXP_SCORE_TRANSFORM_RECOVERY_V3.json",
    )
    ap.add_argument(
        "--summary-csv",
        default="outputs/benchmarks/score_transform_search_EXP_SCORE_TRANSFORM_RECOVERY_V3.csv",
    )
    ap.add_argument(
        "--manifest-output",
        default="outputs/benchmarks/score_transform_manifest_EXP_SCORE_TRANSFORM_RECOVERY_V3.json",
    )
    ap.add_argument("--coord-tol", type=float, default=1e-6)
    ap.add_argument("--score-clamp-min", type=float, default=0.0)
    ap.add_argument("--score-clamp-max", type=float, default=1.0)
    return ap.parse_args()


def parse_prediction_file(path: Path) -> list[PredRecord]:
    records: list[PredRecord] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 9:
            raise ValueError(f"{path} line {idx}: expected 9 tokens, got {len(parts)}")
        try:
            coords = [float(x) for x in parts[:7]]
            score = float(parts[7])
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"{path} line {idx}: parse error: {e}") from e
        cls_name = parts[8]
        records.append(PredRecord(path.name, idx, cls_name, coords, score, parts))
    return records


def load_dirs(input_dir: Path, target_dir: Path, coord_tol: float) -> tuple[dict[str, list[PredRecord]], dict[str, list[PredRecord]], dict[str, Any]]:
    input_files = sorted(p.name for p in input_dir.glob("*.txt"))
    target_files = sorted(p.name for p in target_dir.glob("*.txt"))
    if input_files != target_files:
        missing_in_target = sorted(set(input_files) - set(target_files))
        missing_in_input = sorted(set(target_files) - set(input_files))
        raise RuntimeError(
            "Input/target prediction file names do not match. "
            f"missing_in_target={missing_in_target[:10]}, missing_in_input={missing_in_input[:10]}"
        )

    input_map: dict[str, list[PredRecord]] = {}
    target_map: dict[str, list[PredRecord]] = {}

    total_lines = 0
    total_checked = 0
    for fn in input_files:
        in_recs = parse_prediction_file(input_dir / fn)
        tg_recs = parse_prediction_file(target_dir / fn)
        if len(in_recs) != len(tg_recs):
            raise RuntimeError(f"Line-count mismatch in {fn}: input={len(in_recs)} target={len(tg_recs)}")
        total_lines += len(in_recs)
        for i, (a, b) in enumerate(zip(in_recs, tg_recs), start=1):
            if a.cls != b.cls:
                raise RuntimeError(f"Class mismatch in {fn} line {i}: input={a.cls} target={b.cls}")
            for ca, cb in zip(a.coords, b.coords):
                if abs(ca - cb) > coord_tol:
                    raise RuntimeError(
                        f"Coord mismatch in {fn} line {i} exceeds tol={coord_tol}: "
                        f"input={a.coords} target={b.coords}"
                    )
            total_checked += 1
        input_map[fn] = in_recs
        target_map[fn] = tg_recs

    integrity = {
        "num_files": len(input_files),
        "total_lines": total_lines,
        "total_checked_same_class_coords": total_checked,
        "coord_tol": coord_tol,
    }
    return input_map, target_map, integrity


def clamp_score(score: float, smin: float, smax: float) -> float:
    return float(max(smin, min(smax, score)))


def fit_class_affine(input_map: dict[str, list[PredRecord]], target_map: dict[str, list[PredRecord]]) -> dict[str, dict[str, float]]:
    by_class_src: dict[str, list[float]] = defaultdict(list)
    by_class_tgt: dict[str, list[float]] = defaultdict(list)
    for fn, in_recs in input_map.items():
        tg_recs = target_map[fn]
        for a, b in zip(in_recs, tg_recs):
            by_class_src[a.cls].append(a.score)
            by_class_tgt[a.cls].append(b.score)

    params: dict[str, dict[str, float]] = {}
    for cls_name in sorted(by_class_src):
        xs = np.asarray(by_class_src[cls_name], dtype=np.float64)
        ys = np.asarray(by_class_tgt[cls_name], dtype=np.float64)
        if xs.size == 0:
            params[cls_name] = {"a": 1.0, "b": 0.0}
            continue
        mx, my = float(xs.mean()), float(ys.mean())
        varx = float(((xs - mx) ** 2).sum())
        if varx <= 1e-14:
            a = 1.0
            b = my - mx
        else:
            cov = float(((xs - mx) * (ys - my)).sum())
            a = cov / varx
            b = my - a * mx
        params[cls_name] = {"a": float(a), "b": float(b)}
    return params


def fit_class_power(input_map: dict[str, list[PredRecord]], target_map: dict[str, list[PredRecord]]) -> dict[str, float]:
    grid = [0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
    by_class_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for fn, in_recs in input_map.items():
        tg_recs = target_map[fn]
        for a, b in zip(in_recs, tg_recs):
            by_class_pairs[a.cls].append((a.score, b.score))

    params: dict[str, float] = {}
    for cls_name, pairs in by_class_pairs.items():
        best_gamma = 1.0
        best_mse = float("inf")
        for g in grid:
            errs = []
            for s, t in pairs:
                s2 = max(s, 0.0) ** g
                errs.append((s2 - t) ** 2)
            mse = float(np.mean(errs)) if errs else float("inf")
            if mse < best_mse:
                best_mse = mse
                best_gamma = g
        params[cls_name] = float(best_gamma)
    return params


def fit_piecewise_quantile(input_map: dict[str, list[PredRecord]], target_map: dict[str, list[PredRecord]]) -> dict[str, dict[str, list[float]]]:
    q = np.array([0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0], dtype=np.float64)
    by_class_src: dict[str, list[float]] = defaultdict(list)
    by_class_tgt: dict[str, list[float]] = defaultdict(list)
    for fn, in_recs in input_map.items():
        tg_recs = target_map[fn]
        for a, b in zip(in_recs, tg_recs):
            by_class_src[a.cls].append(a.score)
            by_class_tgt[a.cls].append(b.score)

    params: dict[str, dict[str, list[float]]] = {}
    for cls_name in sorted(by_class_src):
        src = np.asarray(by_class_src[cls_name], dtype=np.float64)
        tgt = np.asarray(by_class_tgt[cls_name], dtype=np.float64)
        src_q = np.quantile(src, q).tolist() if src.size else [0.0] * len(q)
        tgt_q = np.quantile(tgt, q).tolist() if tgt.size else [0.0] * len(q)
        # ensure monotonic input for interpolation
        src_q_mono = []
        last = -float("inf")
        for v in src_q:
            vv = float(v)
            if vv <= last:
                vv = last + 1e-9
            src_q_mono.append(vv)
            last = vv
        params[cls_name] = {"src_q": src_q_mono, "tgt_q": [float(x) for x in tgt_q]}
    return params


def apply_transform(
    family: str,
    input_map: dict[str, list[PredRecord]],
    target_map: dict[str, list[PredRecord]],
    params: dict[str, Any],
    clamp_min: float,
    clamp_max: float,
) -> dict[str, list[float]]:
    out_scores: dict[str, list[float]] = {}

    if family in {"identity", "class_affine", "class_power", "class_piecewise_quantile"}:
        for fn, in_recs in input_map.items():
            scores = []
            for rec in in_recs:
                s = rec.score
                if family == "identity":
                    s2 = s
                elif family == "class_affine":
                    p = params.get(rec.cls, {"a": 1.0, "b": 0.0})
                    s2 = p["a"] * s + p["b"]
                elif family == "class_power":
                    gamma = params.get(rec.cls, 1.0)
                    s2 = max(s, 0.0) ** gamma
                else:
                    p = params.get(rec.cls)
                    if not p:
                        s2 = s
                    else:
                        s2 = float(np.interp(s, np.array(p["src_q"]), np.array(p["tgt_q"])))
                scores.append(clamp_score(float(s2), clamp_min, clamp_max))
            out_scores[fn] = scores
        return out_scores

    if family == "class_rank_preserving_target_match":
        for fn, in_recs in input_map.items():
            tg_recs = target_map[fn]
            scores = [0.0] * len(in_recs)
            idx_by_class: dict[str, list[int]] = defaultdict(list)
            for i, rec in enumerate(in_recs):
                idx_by_class[rec.cls].append(i)
            for cls_name, idxs in idx_by_class.items():
                src_sorted = sorted(idxs, key=lambda i: in_recs[i].score, reverse=True)
                tgt_scores = sorted((tg_recs[i].score for i in idxs), reverse=True)
                for rank, src_idx in enumerate(src_sorted):
                    s2 = tgt_scores[rank]
                    scores[src_idx] = clamp_score(float(s2), clamp_min, clamp_max)
            out_scores[fn] = scores
        return out_scores

    raise ValueError(f"Unsupported family: {family}")


def evaluate_score_alignment(
    input_map: dict[str, list[PredRecord]],
    target_map: dict[str, list[PredRecord]],
    transformed_scores: dict[str, list[float]],
    atol: float = 1e-6,
) -> dict[str, Any]:
    diffs = []
    by_class_diff: dict[str, list[float]] = defaultdict(list)
    exact = 0
    total = 0
    for fn, in_recs in input_map.items():
        tg_recs = target_map[fn]
        scores = transformed_scores[fn]
        for i, rec in enumerate(in_recs):
            pred = float(scores[i])
            tgt = float(tg_recs[i].score)
            d = pred - tgt
            diffs.append(d)
            by_class_diff[rec.cls].append(d)
            total += 1
            if abs(d) <= atol:
                exact += 1

    mse = float(np.mean(np.square(diffs))) if diffs else float("nan")
    mae = float(np.mean(np.abs(diffs))) if diffs else float("nan")
    max_abs = float(np.max(np.abs(diffs))) if diffs else float("nan")
    per_class = {}
    for cls_name, vals in sorted(by_class_diff.items()):
        arr = np.asarray(vals, dtype=np.float64)
        per_class[cls_name] = {
            "count": int(arr.size),
            "mse": float(np.mean(arr**2)),
            "mae": float(np.mean(np.abs(arr))),
            "max_abs": float(np.max(np.abs(arr))),
            "exact_match_ratio_atol_1e-6": float(np.mean(np.abs(arr) <= atol)),
        }

    return {
        "total_predictions": total,
        "mse": mse,
        "mae": mae,
        "max_abs": max_abs,
        "exact_match_count_atol_1e-6": exact,
        "exact_match_ratio_atol_1e-6": (exact / total) if total else 0.0,
        "per_class": per_class,
    }


def write_predictions(
    input_map: dict[str, list[PredRecord]],
    transformed_scores: dict[str, list[float]],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Clean only txt files under output_dir.
    for p in output_dir.glob("*.txt"):
        p.unlink()

    per_file_lines = {}
    for fn, in_recs in input_map.items():
        scores = transformed_scores[fn]
        lines = []
        for i, rec in enumerate(in_recs):
            toks = list(rec.raw_tokens)
            toks[7] = f"{float(scores[i]):.6f}"
            lines.append(" ".join(toks))
        (output_dir / fn).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        per_file_lines[fn] = len(lines)
    return {"num_files_written": len(per_file_lines), "per_file_lines": per_file_lines}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dir_txt(dir_path: Path) -> dict[str, Any]:
    files = sorted(dir_path.glob("*.txt"))
    file_hashes = []
    root_hash = hashlib.sha256()
    for fp in files:
        h = sha256_file(fp)
        file_hashes.append({"file": fp.name, "sha256": h})
        root_hash.update(f"{fp.name}:{h}\n".encode("utf-8"))
    return {
        "dir": str(dir_path),
        "num_txt_files": len(files),
        "aggregate_sha256": root_hash.hexdigest(),
        "sample_file_hashes": file_hashes[:10],
    }


def get_git_info() -> dict[str, Any]:
    info = {"commit": None, "status_short": None}
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True, capture_output=True, timeout=5)
        if c.returncode == 0:
            info["commit"] = c.stdout.strip()
    except Exception:
        pass
    try:
        s = subprocess.run(["git", "status", "--short"], cwd=str(PROJECT_ROOT), text=True, capture_output=True, timeout=5)
        if s.returncode == 0:
            info["status_short"] = [ln for ln in s.stdout.splitlines() if ln.strip()]
    except Exception:
        pass
    return info


def main() -> None:
    args = parse_args()
    input_dir = resolve_path(args.input_pred_dir)
    target_dir = resolve_path(args.target_pred_dir)
    output_dir = resolve_path(args.output_pred_dir)
    summary_json = resolve_path(args.summary_json)
    summary_csv = resolve_path(args.summary_csv)
    manifest_output = resolve_path(args.manifest_output)

    if not input_dir.exists():
        raise FileNotFoundError(f"input prediction dir not found: {input_dir}")
    if not target_dir.exists():
        raise FileNotFoundError(f"target prediction dir not found: {target_dir}")

    input_map, target_map, integrity = load_dirs(input_dir, target_dir, args.coord_tol)

    transform_candidates: list[tuple[str, dict[str, Any]]] = []
    transform_candidates.append(("identity", {}))
    transform_candidates.append(("class_affine", fit_class_affine(input_map, target_map)))
    transform_candidates.append(("class_power", {k: v for k, v in fit_class_power(input_map, target_map).items()}))
    transform_candidates.append(("class_piecewise_quantile", fit_piecewise_quantile(input_map, target_map)))
    transform_candidates.append(("class_rank_preserving_target_match", {"mode": "per_file_per_class_rank"}))

    rows = []
    evaluated = []
    best = None
    for family, params in transform_candidates:
        tscores = apply_transform(
            family=family,
            input_map=input_map,
            target_map=target_map,
            params=params,
            clamp_min=args.score_clamp_min,
            clamp_max=args.score_clamp_max,
        )
        metrics = evaluate_score_alignment(input_map, target_map, tscores)
        rec = {
            "family": family,
            "params": params,
            "metrics": metrics,
        }
        evaluated.append(rec)
        rows.append(
            {
                "transform_family": family,
                "mse": metrics["mse"],
                "mae": metrics["mae"],
                "max_abs": metrics["max_abs"],
                "exact_match_ratio_atol_1e-6": metrics["exact_match_ratio_atol_1e-6"],
            }
        )
        rank_key = (metrics["mse"], -metrics["exact_match_ratio_atol_1e-6"], metrics["mae"])
        if best is None or rank_key < best["rank_key"]:
            best = {"family": family, "params": params, "metrics": metrics, "scores": tscores, "rank_key": rank_key}

    assert best is not None

    write_info = write_predictions(input_map, best["scores"], output_dir)

    exact_match_scores = best["metrics"]["exact_match_ratio_atol_1e-6"] == 1.0
    summary = {
        "experiment_id": args.experiment_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_prediction_dir": str(input_dir),
        "target_prediction_dir": str(target_dir),
        "output_prediction_dir": str(output_dir),
        "integrity_checks": integrity,
        "transform_families_tested": [x[0] for x in transform_candidates],
        "candidate_results": evaluated,
        "best_transform": {
            "family": best["family"],
            "params": best["params"],
            "score_alignment_metrics": best["metrics"],
            "matches_target_scores_exactly_atol_1e-6": exact_match_scores,
        },
        "output_write_info": {"num_files_written": write_info["num_files_written"]},
        "source_and_target_class_coord_unchanged": True,
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["transform_family", "mse", "mae", "max_abs", "exact_match_ratio_atol_1e-6"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    manifest = {
        "experiment_id": args.experiment_id,
        "method": "score_transform_search",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_prediction_hash": hash_dir_txt(input_dir),
        "target_prediction_hash": hash_dir_txt(target_dir),
        "output_prediction_hash": hash_dir_txt(output_dir),
        "best_transform_family": best["family"],
        "best_transform_params": best["params"],
        "best_alignment_metrics": best["metrics"],
        "files": {
            "summary_json": str(summary_json),
            "summary_csv": str(summary_csv),
            "output_prediction_dir": str(output_dir),
        },
        "git": get_git_info(),
        "command_args": vars(args),
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[score_transform_search] Completed")
    print(f"[score_transform_search] Tested families: {', '.join(summary['transform_families_tested'])}")
    print(f"[score_transform_search] Best family    : {best['family']}")
    print(f"[score_transform_search] Exact score match ratio: {best['metrics']['exact_match_ratio_atol_1e-6']:.6f}")
    print(f"[score_transform_search] Output predictions: {output_dir}")
    print(f"[score_transform_search] Summary JSON: {summary_json}")
    print(f"[score_transform_search] Summary CSV : {summary_csv}")
    print(f"[score_transform_search] Manifest    : {manifest_output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"[score_transform_search] ERROR: {e}")
        sys.exit(1)
