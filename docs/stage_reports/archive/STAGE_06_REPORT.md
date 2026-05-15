# STAGE 06 — Dataset Interface Report

> Date: 2026-05-13  
> Status: COMPLETE (WITH LABEL FALLBACK)

## Scope

Executed `final_plan/STAGE_06_DATASET_INTERFACE.md` only:
- implement CARLA -> PAD conversion script
- implement CARLA -> OpenPCDet conversion script
- generate `data/processed/pad_format` and `data/processed/pcdet_format`
- verify converted outputs and split files
- provide stage report with assumptions/fallbacks

## Implemented Artifacts

- `scripts/convert_to_pad.py`
  - supports `--help`
  - converts `rgb/` + `semantic/` to:
    - `images/*.png`
    - `masks/*.png` (Cityscapes IDs)
    - `splits/train.txt`, `splits/val.txt`
  - writes summary JSON
- `scripts/convert_to_pcdet.py`
  - supports `--help`
  - converts `lidar/*.npy` to `points/*.npy` float32 `(N,4)`
  - generates `labels/*.txt` in OpenPCDet custom format
  - generates `ImageSets/train.txt`, `ImageSets/val.txt`
  - optional `--generate-custom-infos` to create:
    - `custom_infos_train.pkl`
    - `custom_infos_val.pkl`
    - `custom_dbinfos_train.pkl`
  - writes summary JSON
- `configs/carla_lidar.yaml`
  - project-level dataset/conversion reference for CARLA LiDAR -> OpenPCDet
- `docs/stage_reports/archive/STAGE_06_REPORT.md`

## Conversion Runs

Input used:
- `data/raw/stage03_longrun_1000_snapshotmeta`

### PAD conversion

Command:
```bash
conda run -n carla-client python scripts/convert_to_pad.py \
  --input-dir data/raw/stage03_longrun_1000_snapshotmeta \
  --output-dir data/processed/pad_format \
  --num-frames 1000 --overwrite --shuffle-split --seed 42
```

Result:
- converted frames: `1000`
- split: `train=800`, `val=200`
- mapped classes seen: `[3, 255]`
- sample shape: image `(180,320,3)`, mask `(180,320)`

### OpenPCDet conversion

Command (final):
```bash
conda run -n pcdet python scripts/convert_to_pcdet.py \
  --input-dir data/raw/stage03_longrun_1000_snapshotmeta \
  --output-dir data/processed/pcdet_format \
  --num-frames 1000 --overwrite --shuffle-split --seed 42 \
  --generate-custom-infos
```

Result:
- converted frames: `1000`
- split: `train=800`, `val=200`
- points dtype/shape check: `float32`, `(N,4)`
- non-finite filtering frames: `0`
- labels written: `0` (all frames empty-label fallback)
- generated infos/dbinfos:
  - `custom_infos_train.pkl`
  - `custom_infos_val.pkl`
  - `custom_dbinfos_train.pkl`

## Validation Performed

1. CLI + syntax checks
- `conda run -n carla-client python -m py_compile scripts/convert_to_pad.py scripts/convert_to_pcdet.py`
- `conda run -n carla-client python scripts/convert_to_pad.py --help`
- `conda run -n carla-client python scripts/convert_to_pcdet.py --help`

2. Output integrity checks
- PAD:
  - image/mask shape match
  - mask classes subset of `{0..18,255}`
  - split files exist and disjoint
- OpenPCDet:
  - `points/*.npy` shape `(N,4)` and finite values
  - `labels/*.txt` format valid (empty allowed)
  - split files exist and disjoint

3. PAD loader compatibility smoke
- `conda run -n pad python tools/vis/seg_img_dir.py ...` (1000-frame directory run)
- PASS (no crash)

4. OpenPCDet custom dataset load smoke (with same runtime stubs used in Stage 05)
- instantiated `CustomDataset` against `data/processed/pcdet_format`
- PASS (`len=200` on test split, sample retrieval succeeds)

## Exit Criteria Status

- [x] Both conversion scripts run without errors on 1000 frames
- [x] PAD format loadable by PAD visualizer path
- [x] Train/val split generation done for PAD and PCDet
- [x] Point clouds are float32 `(N,4)` with no NaN/Inf
- [ ] OpenPCDet upstream `create_custom_infos` command runs cleanly on this dataset

## Known Issues / Fallbacks

1. **Missing actor annotations in Stage 03 metadata**
- Current metadata contains only ego state (`ego_vehicle`) + sensor frame IDs.
- No actor bounding boxes are available to produce true 3D labels.
- Fallback applied: write empty label files and still generate dataset structure + infos.

2. **OpenPCDet upstream `create_custom_infos` bug with empty label files**
- `pcdet/datasets/custom/custom_dataset.py` assumes non-empty label arrays and crashes (`IndexError`) when labels are empty.
- Fallback applied in project script:
  - `scripts/convert_to_pcdet.py --generate-custom-infos` generates compatible info/dbinfo files directly in `carla-perception-lab`.

3. **Cross-env pickle compatibility**
- Infos generated in `carla-client` env (NumPy 2.x) are not loadable in `pcdet` env (NumPy 1.24) due pickle module path mismatch.
- Fallback applied: run `convert_to_pcdet.py` with `pcdet` env for final artifacts.

## Post-Stage Fix Update (2026-05-13)

After Stage 06, root-cause fix was implemented before Stage 07:
- `scripts/carla_recorder.py` now writes `actors` with actor class + world bbox metadata per frame.
- Probe recording (`data/raw/stage06_label_probe`) verified non-empty actor lists.
- Re-conversion to probe PCDet dataset produced non-empty labels:
  - `frames_with_empty_labels`: `13/80`
  - `total_labels_written`: `67`

This resolves the all-empty-label issue for **new recordings**.  
Historical dataset `data/raw/stage03_longrun_1000_snapshotmeta` remains unchanged and still has empty-label fallback behavior.

Additional conversion fix applied later:
- `scripts/convert_to_pcdet.py` now flips LiDAR point `y` axis by default (`--flip-point-y` enabled).
- This keeps point cloud convention consistent with generated PCDet labels (`y_pcdet = -y_carla`) and avoids point/label frame mismatch.

## Assumptions

1. Stage 06 allows empty-label fallback when source data has no actor bbox annotations.
2. Stage 06 success can be based on reproducible conversion interfaces + format correctness, while documenting OpenPCDet upstream limitations explicitly.

## Next Recommended Action

Proceed to `STAGE_07_SEGMENTATION_INTEGRATION.md`.

## Revalidation Update (2026-05-13, Hardened Plan)

After implementing hardened Stage 03-08 changes:

### Converter fixes validated

1. `scripts/convert_to_pad.py`
- updated default semantic mapping to CARLA 0.10 profile (`carla_010`)
- supports multi-input merge (`--input-dirs`)
- strict output policy: existing output dir now hard-fails without `--overwrite`
- emits `conversion_manifest.json` + richer histogram/unknown-ID summary

2. `scripts/convert_to_pcdet.py`
- supports multi-input merge (`--input-dirs`)
- strict output policy: existing output dir now hard-fails without `--overwrite`
- emits `conversion_manifest.json` + richer label/empty-ratio summary

### Regression checks

- Re-running converters into an existing output dir **without** `--overwrite` now fails as intended.
- CARLA semantic ID `25` is no longer dropped to ignore by default mapping.

### Canonical mixed conversion run

Commands:

```bash
conda run -n carla-client python scripts/convert_to_pad.py \
  --input-dirs data/raw/stage03_mine_main_1000 data/raw/stage03_mine_aux_200_traffic_far \
  --output-dir data/processed/pad_format_stage08_canonical_mix \
  --num-frames 0 --overwrite --shuffle-split --seed 42 \
  --summary-json output/stage_validation/stage06_pad_summary_canonical_mix.json \
  --manifest-json output/stage_validation/stage06_pad_manifest_canonical_mix.json

conda run -n pcdet python scripts/convert_to_pcdet.py \
  --input-dirs data/raw/stage03_mine_main_1000 data/raw/stage03_mine_aux_200_traffic_far \
  --output-dir data/processed/pcdet_format_stage08_canonical_mix \
  --num-frames 0 --overwrite --shuffle-split --seed 42 \
  --generate-custom-infos \
  --summary-json output/stage_validation/stage06_pcdet_summary_canonical_mix.json \
  --manifest-json output/stage_validation/stage06_pcdet_manifest_canonical_mix.json
```

Results:
- PAD converted frames: `1200`
- PAD mapped classes seen: `[9, 10, 13, 14, 255]`
- PCDet converted frames: `1200`
- PCDet labels: `Vehicle=187`, `Pedestrian=30`, `Cyclist=0`
- PCDet empty-label ratio: `0.8425`

This run satisfies the hardened class gate requirement (Vehicle + at least one secondary class).
