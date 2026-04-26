# thesis

Unified training pipeline for combining three datasets (breast, lungs, ISIC) into one **CNN + ViT hybrid** classifier with:

- automatic `.zip` extraction into separate folders
- unified metadata generation (`image_path`, `label`, `source_dataset`)
- label normalization to a shared binary target
- stratified 80/10/10 split (train/val/test)
- train + validation with augmentation, class weighting, early stopping
- final test metrics overall and per dataset

## Expected data layout

Use absolute path root:

`C:\Users\fatem\Desktop\thesis_project\data`

Expected files inside data root:

- `breast_images.zip`
- `lungs_images.zip`
- `ISIC_2024_Training_Input.zip`
- `ISIC_2024_Training_GroundTruth.csv`

## Run

```bash
python train_hybrid.py \
  --data-root /absolute/path/to/thesis_project/data \
  --output-dir /absolute/path/to/thesis_project/output \
  --epochs 20 \
  --batch-size 16 \
  --patience 5
```

## Output files

- `metadata_unified.csv`
- `train.csv`
- `val.csv`
- `test.csv`
- `hybrid_cnn_vit.pt`
- `metrics_test.json`
- extracted datasets under `output/extracted/`

## Notes

- The script enforces absolute `--data-root`.
- If dataset labels differ, they are normalized to binary labels using keyword and numeric mapping.
- Stratification uses `label + source_dataset` when possible, then falls back to label-only if required by class counts.
