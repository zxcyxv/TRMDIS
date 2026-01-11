"""
Compare XGBoost performance with PCA 16-dim vs Full 128-dim CLS features.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import KFold


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "open_track1"
MODEL_DIR = ROOT_DIR / "models"
K8_PATH = DATA_DIR / "train_features_k8.csv"
GATE_PATH = DATA_DIR / "xgb_hybrid_features_oof.csv"
PCA_PATH = DATA_DIR / "cls_pca_train_oof.csv"
FULL_CLS_PATH = DATA_DIR / "cls_full_train_oof.csv"


def euclidean_distance(y_true_x, y_true_y, y_pred_x, y_pred_y):
    """Calculate mean Euclidean distance between predictions and ground truth."""
    return np.mean(np.sqrt((y_true_x - y_pred_x) ** 2 + (y_true_y - y_pred_y) ** 2))


def get_feature_columns(df):
    """Extract all feature columns (excluding episode ID and targets)."""
    exclude_cols = ["game_episode", "target_end_x", "target_end_y", "last_result_name"]
    return [c for c in df.columns if c not in exclude_cols]


def train_xgb_oof(X, y_x, y_y, n_splits=5, seed=42):
    """Train XGBoost with K-fold OOF validation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_pred_x = np.zeros(len(X))
    oof_pred_y = np.zeros(len(X))

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train_x, y_val_x = y_x.iloc[train_idx], y_x.iloc[val_idx]
        y_train_y, y_val_y = y_y.iloc[train_idx], y_y.iloc[val_idx]

        # Train X model
        model_x = xgb.XGBRegressor(
            objective='reg:absoluteerror',
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            early_stopping_rounds=50,
            n_jobs=-1,
        )
        model_x.fit(
            X_train, y_train_x,
            eval_set=[(X_val, y_val_x)],
            verbose=False
        )

        # Train Y model
        model_y = xgb.XGBRegressor(
            objective='reg:absoluteerror',
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            early_stopping_rounds=50,
            n_jobs=-1,
        )
        model_y.fit(
            X_train, y_train_y,
            eval_set=[(X_val, y_val_y)],
            verbose=False
        )

        # OOF predictions
        oof_pred_x[val_idx] = model_x.predict(X_val)
        oof_pred_y[val_idx] = model_y.predict(X_val)

        fold_dist = euclidean_distance(y_val_x, y_val_y,
                                       oof_pred_x[val_idx], oof_pred_y[val_idx])
        print(f"  Fold {fold_idx}: {fold_dist:.4f}m")

    overall_dist = euclidean_distance(y_x, y_y, oof_pred_x, oof_pred_y)
    return overall_dist


def main():
    print("="*70)
    print("XGBoost CLS Feature Comparison")
    print("="*70)

    # Load base features
    print("\nLoading features...")
    df_k8 = pd.read_csv(K8_PATH)
    df_gate = pd.read_csv(GATE_PATH)

    if "game_episode" not in df_gate.columns:
        raise KeyError("xgb_hybrid_features_oof.csv must include game_episode.")

    gate_cols = ["game_episode", "router_gate"]
    gate_cols += [c for c in df_gate.columns if c.startswith("router_zone_logit_")]
    df_base = df_k8.merge(df_gate[gate_cols], on="game_episode", how="left")

    if df_base["router_gate"].isna().any():
        raise ValueError("router_gate has missing values after merge.")

    print(f"  Base features: {len(get_feature_columns(df_base))} columns")

    # Prepare targets
    y_x = df_base["target_end_x"]
    y_y = df_base["target_end_y"]

    # ========================================================================
    # Test 1: PCA 16-dim
    # ========================================================================
    print("\n" + "="*70)
    print("Test 1: Base + Router + PCA 16-dim")
    print("="*70)

    df_pca = pd.read_csv(PCA_PATH)
    df_with_pca = df_base.merge(df_pca, on="game_episode", how="left")

    if df_pca.drop(columns=["game_episode"]).isna().any().any():
        raise ValueError("PCA features contain NaNs.")

    feature_cols_pca = get_feature_columns(df_with_pca)
    X_pca = df_with_pca[feature_cols_pca]

    print(f"  Total features: {len(feature_cols_pca)}")
    print(f"  Training with 5-fold OOF...")

    dist_pca = train_xgb_oof(X_pca, y_x, y_y)
    print(f"\n  [PCA 16-dim] OOF Distance: {dist_pca:.4f}m")

    # ========================================================================
    # Test 2: Full 128-dim
    # ========================================================================
    print("\n" + "="*70)
    print("Test 2: Base + Router + Full 128-dim CLS")
    print("="*70)

    df_full = pd.read_csv(FULL_CLS_PATH)
    df_with_full = df_base.merge(df_full, on="game_episode", how="left")

    if df_full.drop(columns=["game_episode"]).isna().any().any():
        raise ValueError("Full CLS features contain NaNs.")

    feature_cols_full = get_feature_columns(df_with_full)
    X_full = df_with_full[feature_cols_full]

    print(f"  Total features: {len(feature_cols_full)}")
    print(f"  Training with 5-fold OOF...")

    dist_full = train_xgb_oof(X_full, y_x, y_y)
    print(f"\n  [Full 128-dim] OOF Distance: {dist_full:.4f}m")

    # ========================================================================
    # Comparison
    # ========================================================================
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"  PCA 16-dim:    {dist_pca:.4f}m  ({len(feature_cols_pca)} features)")
    print(f"  Full 128-dim:  {dist_full:.4f}m  ({len(feature_cols_full)} features)")
    print(f"  Difference:    {dist_full - dist_pca:+.4f}m")

    if dist_pca < dist_full:
        improvement = ((dist_full - dist_pca) / dist_full) * 100
        print(f"\n  ✓ PCA 16-dim is BETTER by {improvement:.2f}%")
    else:
        improvement = ((dist_pca - dist_full) / dist_pca) * 100
        print(f"\n  ✓ Full 128-dim is BETTER by {improvement:.2f}%")

    print("="*70)


if __name__ == "__main__":
    main()
