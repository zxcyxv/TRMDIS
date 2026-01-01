import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "open_track1"
MODEL_DIR = ROOT_DIR / "models"

FIELD_X = 105.0
FIELD_Y = 68.0


def get_feature_columns(df):
    exclude_cols = ["game_episode", "target_end_x", "target_end_y", "last_result_name"]
    return [c for c in df.columns if c not in exclude_cols]


def main() -> None:
    test_k8 = pd.read_csv(DATA_DIR / "test_features_k8.csv")
    gate_df = pd.read_csv(DATA_DIR / "xgb_hybrid_features_test.csv")

    df = test_k8.merge(gate_df, on="game_episode", how="left")
    if df["router_gate"].isna().any():
        raise ValueError("router_gate has missing values after merge.")

    feature_cols_path = MODEL_DIR / "feature_columns.txt"
    if feature_cols_path.exists():
        feature_cols = [line.strip() for line in feature_cols_path.read_text().splitlines() if line.strip()]
    else:
        feature_cols = get_feature_columns(df)
        if "router_gate" not in feature_cols:
            feature_cols.append("router_gate")

    X = df[feature_cols]

    model_x = xgb.XGBRegressor()
    model_y = xgb.XGBRegressor()
    model_x.load_model(str(MODEL_DIR / "xgb_full_x.json"))
    model_y.load_model(str(MODEL_DIR / "xgb_full_y.json"))

    pred_x = np.clip(model_x.predict(X), 0, FIELD_X)
    pred_y = np.clip(model_y.predict(X), 0, FIELD_Y)

    submission = pd.DataFrame({
        "game_episode": df["game_episode"],
        "end_x": pred_x,
        "end_y": pred_y,
    })
    out_path = DATA_DIR / "submission_xgb_hybrid.csv"
    submission.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
