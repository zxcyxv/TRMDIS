# Hybrid Router + XGBoost Pipeline (Clean Workspace)

This folder contains only the minimum code needed to reproduce the
Router + XGBoost regression pipeline that achieved ~13.1m OOF distance.

## Contents

- `train_film_smoe2.py`
  - FiLM + Spatial MoE Transformer used to produce router outputs
  - Trains and saves `open_track1/film_smoe_transformer.pt`
- `feature_engineering_k5.py`
  - Generates K=8 wide features used by XGBoost
  - Outputs `open_track1/train_features_k8.csv` and `open_track1/test_features_k8.csv`
- `scripts/extract_router_features.py`
  - Uses the trained checkpoint to extract:
    - `router_gate`
    - `router_zone_logit_0..3`
  - Writes `open_track1/xgb_hybrid_features.csv`
- `scripts/train_xgb_regressor.py`
  - Merges K=8 features with router features
  - 5-fold OOF training and final fit
- `XGB_FEATURES.md`
  - Exact description of the XGBoost input features

## End-to-end steps

1) Generate K=8 wide features

```
uv run feature_engineering_k5.py
```

2) Train the router model

```
uv run train_film_smoe2.py
```

3) Extract router features (gate + 4-class logits)

```
uv run scripts/extract_router_features.py
```

4) Train XGBoost with OOF evaluation

```
uv run scripts/train_xgb_regressor.py
```

5) Extract router features for test set

```
uv run scripts/extract_router_features_test.py
```

6) Generate submission from saved XGBoost models

```
uv run scripts/predict_xgb_submission.py
```

## Notes

- All data paths are relative to this folder: `hybrid_router_xgb/open_track1`.
- `train_features_k8.csv` and `xgb_hybrid_features.csv` are merged on `game_episode`.
- If you already have the transformer checkpoint, you can skip step (2).
