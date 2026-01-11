"""
Train XGBoost with 80/20 split (same as TabPFN v2) for fair comparison.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "open_track1"
MODEL_DIR = ROOT_DIR / "models"
K8_PATH = DATA_DIR / "train_features_k8.csv"
GATE_PATH = DATA_DIR / "xgb_hybrid_features_oof.csv"
PCA_PATH = DATA_DIR / "cls_pca_train_oof.csv"


def euclidean_distance(y_true_x, y_true_y, y_pred_x, y_pred_y):
    return np.mean(np.sqrt((y_true_x - y_pred_x) ** 2 + (y_true_y - y_pred_y) ** 2))


def get_feature_columns(df):
    exclude_cols = ["game_episode", "target_end_x", "target_end_y", "last_result_name"]
    return [c for c in df.columns if c not in exclude_cols]


def main() -> None:
    print("="*70)
    print("XGBoost Training with 80/20 split")
    print("="*70)

    # Load features
    print("\nLoading features...")
    df_k8 = pd.read_csv(K8_PATH)
    df_gate = pd.read_csv(GATE_PATH)

    if "game_episode" not in df_gate.columns:
        raise KeyError("Gate features must include game_episode.")

    gate_cols = ["game_episode", "router_gate"]
    gate_cols += [c for c in df_gate.columns if c.startswith("router_zone_logit_")]
    df = df_k8.merge(df_gate[gate_cols], on="game_episode", how="left")

    if df["router_gate"].isna().any():
        raise ValueError("router_gate has missing values after merge.")

    if PCA_PATH.exists():
        df_pca = pd.read_csv(PCA_PATH)
        df = df.merge(df_pca, on="game_episode", how="left")
        if df_pca.drop(columns=["game_episode"]).isna().any().any():
            raise ValueError("PCA features contain NaNs.")
        print(f"  Added {len(df_pca.columns) - 1} PCA features")

    feature_cols = get_feature_columns(df)
    if "router_gate" not in feature_cols:
        feature_cols.append("router_gate")

    print(f"  Total features: {len(feature_cols)}")

    X = df[feature_cols]
    y_x = df["target_end_x"]
    y_y = df["target_end_y"]

    # Split 80/20 with SAME random_state as TabPFN
    X_train, X_val, y_train_x, y_val_x, y_train_y, y_val_y = train_test_split(
        X, y_x, y_y, test_size=0.2, random_state=42
    )

    print(f"\n  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")

    # Train XGBoost for end_x
    print("\n  Training XGBoost for end_x...")
    model_x = xgb.XGBRegressor(
        objective='reg:absoluteerror',
        tree_method='hist',
        device='cuda',
        max_depth=7,
        learning_rate=0.05,
        n_estimators=1000,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
        verbosity=0,
    )
    model_x.fit(
        X_train, y_train_x,
        eval_set=[(X_val, y_val_x)],
        verbose=False
    )

    # Train XGBoost for end_y
    print("  Training XGBoost for end_y...")
    model_y = xgb.XGBRegressor(
        objective='reg:absoluteerror',
        tree_method='hist',
        device='cuda',
        max_depth=7,
        learning_rate=0.05,
        n_estimators=1000,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
        verbosity=0,
    )
    model_y.fit(
        X_train, y_train_y,
        eval_set=[(X_val, y_val_y)],
        verbose=False
    )

    # Validation predictions
    print("\n  Predicting on validation set...")
    val_pred_x = model_x.predict(X_val)
    val_pred_y = model_y.predict(X_val)

    val_dist = euclidean_distance(y_val_x, y_val_y, val_pred_x, val_pred_y)
    print(f"\n[Validation Result] Distance: {val_dist:.4f}m")

    # Save models
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_x.get_booster().save_model(str(MODEL_DIR / "xgb_8020_x.json"))
    model_y.get_booster().save_model(str(MODEL_DIR / "xgb_8020_y.json"))

    print(f"\nSaved models:")
    print(f"  {MODEL_DIR / 'xgb_8020_x.json'}")
    print(f"  {MODEL_DIR / 'xgb_8020_y.json'}")

    # Save validation predictions for correlation analysis
    val_predictions = pd.DataFrame({
        'game_episode': df.iloc[X_val.index]['game_episode'].values,
        'xgb_pred_x': val_pred_x,
        'xgb_pred_y': val_pred_y,
        'true_x': y_val_x.values,
        'true_y': y_val_y.values,
    })
    val_pred_path = DATA_DIR / "xgb_8020_val_predictions.csv"
    val_predictions.to_csv(val_pred_path, index=False)
    print(f"\nSaved validation predictions: {val_pred_path}")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
