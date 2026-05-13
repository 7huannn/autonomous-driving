# Dataset Quarantine List (Stage 03-08)

Date: 2026-05-13

Machine-readable source of truth:
- `output/stage_validation/quarantine_datasets.json`

## Policy

A dataset is quarantined when one or more conditions hold:
- `empty_dataset`
- `unaligned_modalities`
- `missing_completion_marker`
- `no_actor_labels_for_detection`
- `town10_chunk_collection_failed`

Quarantined datasets are kept for debugging only and must not be used as canonical inputs for stage progression.

## Notable Quarantined Cases

- `data/raw/stage03_diag_600_light` (unaligned modalities)
- `data/raw/stage03_longrun_1000_after_fix` (unaligned modalities)
- `data/raw/stage03_probe_300_town10_light` (empty)
- `data/raw/stage08_label_probe_mix2` (empty)
- `data/raw/stage03_town10_chunks` (`town10_chunk_collection_failed`)
- `data/raw/stage03_longrun_1000_snapshotmeta` (no actor labels for detection)
- `data/raw/stage03_mine_main_1000` (no actor labels for detection)

## Runtime Note

Town10 collection remains unstable on this host/runtime (native `std::exception`/abort behavior). Until this runtime issue is resolved, Town10 attempts are quarantined and excluded from readiness gates.
