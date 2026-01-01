import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold


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

    if "game_episode" not in df_gate.columns:
        raise KeyError("xgb_hybrid_features.csv must include game_episode for merge.")

    gate_cols = ["game_episode", "router_gate"]
    gate_cols += [c for c in df_gate.columns if c.startswith("router_zone_logit_")]
    df = df_k8.merge(df_gate[gate_cols], on="game_episode", how="left")
    if df["router_gate"].isna().any():
        raise ValueError("router_gate has missing values after merge.")

    feature_cols = get_feature_columns(df)
    if "router_gate" not in feature_cols:
        feature_cols.append("router_gate")
    X = df[feature_cols]
    y_x = df["target_end_x"].values
    y_y = df["target_end_y"].values

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

    print("OOF training with 5 folds...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred_x = np.zeros(len(X))
    oof_pred_y = np.zeros(len(X))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        yx_tr, yx_val = y_x[tr_idx], y_x[val_idx]
        yy_tr, yy_val = y_y[tr_idx], y_y[val_idx]

        model_x = xgb.XGBRegressor(**params)
        model_y = xgb.XGBRegressor(**params)

        model_x.fit(X_tr, yx_tr, eval_set=[(X_val, yx_val)], verbose=False)
        model_y.fit(X_tr, yy_tr, eval_set=[(X_val, yy_val)], verbose=False)

        oof_pred_x[val_idx] = model_x.predict(X_val)
        oof_pred_y[val_idx] = model_y.predict(X_val)

        fold_dist = euclidean_distance(yx_val, yy_val, oof_pred_x[val_idx], oof_pred_y[val_idx])
        print(f"  Fold {fold}: dist={fold_dist:.4f}m")

    oof_dist = euclidean_distance(y_x, y_y, oof_pred_x, oof_pred_y)
    print(f"\n[OOF Result] Distance: {oof_dist:.4f}m")

    print("\nTraining full models on all data...")
    params_no_es = dict(params)
    params_no_es.pop("early_stopping_rounds", None)
    full_x = xgb.XGBRegressor(**params_no_es)
    full_y = xgb.XGBRegressor(**params_no_es)
    full_x.fit(X, y_x, verbose=False)
    full_y.fit(X, y_y, verbose=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    full_x.save_model(str(MODEL_DIR / "xgb_full_x.json"))
    full_y.save_model(str(MODEL_DIR / "xgb_full_y.json"))
    (MODEL_DIR / "feature_columns.txt").write_text("\n".join(feature_cols))

    importances = full_x.feature_importances_
    feature_names = X.columns
    indices = np.argsort(importances)[-20:]

    print("\nTop 20 Feature Importances (X-coordinate):")
    for idx in indices[::-1]:
        print(f"  {feature_names[idx]}: {importances[idx]:.6f}")


if __name__ == "__main__":
    main()
