import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = str(ROOT_DIR / "open_track1")
MODEL_DIR = ROOT_DIR / "models"
K8_PATH = f"{DATA_DIR}/train_features_k8.csv"
GATE_PATH = f"{DATA_DIR}/xgb_hybrid_features.csv"


def euclidean_distance(y_true_x, y_true_y, y_pred_x, y_pred_y):
    return np.mean(np.sqrt((y_true_x - y_pred_x) ** 2 + (y_true_y - y_pred_y) ** 2))


def get_feature_columns(df):
    exclude_cols = ["game_episode", "target_end_x", "target_end_y", "last_result_name"]
    return [c for c in df.columns if c not in exclude_cols]


def main() -> None:
    df_k8 = pd.read_csv(K8_PATH)
    df_gate = pd.read_csv(GATE_PATH)
    pca_path = Path(DATA_DIR) / "cls_pca_train.csv"

    if "game_episode" not in df_gate.columns:
        raise KeyError("xgb_hybrid_features.csv must include game_episode for merge.")

    gate_cols = ["game_episode", "router_gate"]
    gate_cols += [c for c in df_gate.columns if c.startswith("router_zone_logit_")]
    df = df_k8.merge(df_gate[gate_cols], on="game_episode", how="left")
    if df["router_gate"].isna().any():
        raise ValueError("router_gate has missing values after merge.")
    if pca_path.exists():
        df_pca = pd.read_csv(pca_path)
        df = df.merge(df_pca, on="game_episode", how="left")
        if df_pca.drop(columns=["game_episode"]).isna().any().any():
            raise ValueError("PCA features contain NaNs.")
        print(f"Loaded {len(df_pca.columns) - 1} PCA features")

    feature_cols = get_feature_columns(df)
    if "router_gate" not in feature_cols:
        feature_cols.append("router_gate")

    print(f"Total features: {len(feature_cols)}")

    X = df[feature_cols]
    y_x = df["target_end_x"].values
    y_y = df["target_end_y"].values

    # 80/20 split
    print("\nTraining XGBoost with 80/20 split...")
    X_train, X_val, yx_train, yx_val, yy_train, yy_val = train_test_split(
        X, y_x, y_y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train)} samples, Val: {len(X_val)} samples")

    params = {
        "objective": "reg:absoluteerror",
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 7,
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 50,
        "verbosity": 0,
    }

    print("\n  Training XGBoost for end_x...")
    model_x = xgb.XGBRegressor(**params)
    model_x.fit(X_train, yx_train, eval_set=[(X_val, yx_val)], verbose=False)

    print("  Training XGBoost for end_y...")
    model_y = xgb.XGBRegressor(**params)
    model_y.fit(X_train, yy_train, eval_set=[(X_val, yy_val)], verbose=False)

    print("  Predicting on validation set...")
    val_pred_x = model_x.predict(X_val)
    val_pred_y = model_y.predict(X_val)

    val_dist = euclidean_distance(yx_val, yy_val, val_pred_x, val_pred_y)
    print(f"\n[Validation Result] Distance: {val_dist:.4f}m")

    print("\nTraining full models on all data...")
    params_no_es = dict(params)
    params_no_es.pop("early_stopping_rounds", None)
    full_x = xgb.XGBRegressor(**params_no_es)
    full_y = xgb.XGBRegressor(**params_no_es)
    full_x.fit(X, y_x, verbose=False)
    full_y.fit(X, y_y, verbose=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    full_x.get_booster().save_model(str(MODEL_DIR / "xgb_full_x_8020.json"))
    full_y.get_booster().save_model(str(MODEL_DIR / "xgb_full_y_8020.json"))

    importances = full_x.feature_importances_
    feature_names = X.columns
    indices = np.argsort(importances)[-20:]

    print("\nTop 20 Feature Importances (X-coordinate):")
    for idx in indices[::-1]:
        print(f"  {feature_names[idx]}: {importances[idx]:.6f}")

    print("\n=== Training Complete ===")
    print(f"Validation Distance: {val_dist:.4f}m")


if __name__ == "__main__":
    main()
