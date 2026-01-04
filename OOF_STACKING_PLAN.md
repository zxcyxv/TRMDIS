# K-Fold OOF Stacking Plan (FiLM+Spatial MoE + XGBoost)

This document captures the final, agreed execution plan and routing policy.
All paths are under `/workspace/TRMDIS/Test3` unless noted.

## Goals

- Remove data leakage by using 5-fold OOF CLS features for training.
- Keep router features clean for train (OOF) and stable for test (single best fold).
- Feed clean OOF features into XGBoost to improve generalization.

## Key Decisions

1) CLS features
- Train: OOF per fold (each sample uses its own fold model).
- Test: 5-fold ensemble average.

2) Router features
- Train: OOF per fold (each sample uses its own fold model).
- Test: use fold 3 checkpoint only (best fold).

3) Episode-level splitting
- Always split by `game_episode` to avoid leakage.

## Artifacts

Train (OOF):
- `Test3/open_track1/film_smoe_cond_features_oof.npz`
- `Test3/open_track1/xgb_hybrid_features_oof.csv` (router OOF, to be generated)
- `Test3/open_track1/cls_pca_train_oof.csv`

Test:
- `Test3/open_track1/film_smoe_cond_features_test_oof.npz`
- `Test3/open_track1/xgb_hybrid_features_test_fold3.csv` (router test, fold 3)
- `Test3/open_track1/cls_pca_test_oof.csv`

Checkpoints:
- `Test3/open_track1/film_smoe_fold0.pt` ... `film_smoe_fold4.pt`

## Execution Steps

1) K-fold FiLM+MoE training (already done)
```
python Test3/train_film_smoe_kfold.py
```

2) CLS features (train/test)
```
python Test3/scripts/extract_cls_features_test_ensemble.py
python Test3/scripts/build_cls_pca_features_oof.py
```

3) Router features (train OOF)
- Use the same episode-level K-fold split (n_splits=5, random_state=42).
- For each fold:
  - load `film_smoe_fold{fold}.pt`
  - scale inputs with fold scaler
  - extract router gate + zone logits for that fold’s validation indices
  - write into the global OOF arrays
- Output: `Test3/open_track1/xgb_hybrid_features_oof.csv`

4) Router features (test, fold 3 only)
- Load `film_smoe_fold3.pt` and extract router gate + zone logits on full test set.
- Output: `Test3/open_track1/xgb_hybrid_features_test_fold3.csv`

5) XGBoost training (OOF)
```
python Test3/scripts/train_xgb_regressor.py
```

## Validation Checklist

- No episode overlap between train/val folds.
- OOF CLS correlation vs in-sample < 0.98 (`validate_oof_features.py`).
- Router diagnostics for fold 3 look sane (gate variance, zone separation).
- OOF XGB score slightly higher than in-sample (expected).

## Notes

- Fold 3 is selected for router test features because it has the best fold distance.
- If router OOF extraction script does not exist yet, implement it by adapting
  `Test3/scripts/extract_router_features.py` to iterate fold checkpoints and
  write OOF outputs by validation indices.
