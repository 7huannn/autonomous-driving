# Baseline Manifest Policy

## Purpose
This policy defines when a detection result is eligible to be treated as baseline-grade and potentially replace the current comparison anchor.

## Baseline-Grade Result
A baseline-grade result is a result that is:
- reproducible from recorded inputs and command provenance,
- verifiable by hash checks for result, config, split, and source files,
- validated by harness checks (environment, dataset quality, compare, gate), and
- accompanied by a manifest that passes `evals/check_manifest.py --mode baseline-grade`.

## Historical Baseline Status (mAP=0.4591)
Current anchor result:
- `output/detection_3d_canonical_mix_v3/eval_results_fusion_rescore_best.json`
- mAP `0.4591`

This result is frozen as **historical-only** until a manifest-backed reproduction exists, because the original run metadata is incomplete (exact rescore parameters, original command invocation, and timestamped manifest are missing).

## Why Raw Result JSON Is Not Enough
Result JSON files with only metric fields (e.g., `mAP`, `class_ap`) do not prove reproducibility. They do not include sufficient provenance for:
- exact source lineage,
- command arguments,
- hash-verified inputs,
- runtime context and git state.

## Required Manifest Fields
A valid manifest must include (see `evals/manifest_schema.json`):
- result metadata (`manifest_version`, `experiment_id`, `result_file`, `result_sha256`, `result_type`)
- metric metadata (`metric_name`, `mAP`, `class_ap.Vehicle`, `class_ap.Pedestrian`)
- dataset provenance (path, split file, split hash, frame/GT counts)
- config provenance (path + hash)
- source file lineage (path, hash, role)
- command provenance (executable, args, cwd, conda env)
- git/runtime metadata
- rescore parameters for fusion/postprocess results
- validation artifact references
- eligibility decision block

## Eligibility States
### Historical baseline
- Existing anchor used for comparison continuity.
- May fail baseline-grade manifest checks.
- Cannot be claimed as fully reproducible until reconstructed.

### Provisional result
- Candidate experiment output for immediate comparison.
- Must pass harness checks relevant to experiment stage.
- Manifest check in `provisional` mode is recommended.

### Baseline-grade result
- Candidate allowed to replace anchor only if:
  1. Harness gate passes (provisional/target as required), and
  2. Manifest check passes in `baseline-grade` mode.

## Replacement Rule
No future result may replace the current baseline anchor unless both are true:
1. harness gate pass, and
2. manifest check pass (`--mode baseline-grade`).

Until then, `0.4591` remains the official comparison anchor.

## Review Checklist
- [ ] Result file hash matches manifest.
- [ ] Config hash matches manifest.
- [ ] Split file hash matches manifest.
- [ ] Source file hashes and roles are complete.
- [ ] Command args and environment are recorded.
- [ ] Git commit and status are recorded.
- [ ] Rescore params are explicit (if fusion/postprocess).
- [ ] Manifest check passes for intended mode.
- [ ] Harness gate decision is recorded.
