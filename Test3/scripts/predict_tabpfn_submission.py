"""
Generate submission file using TabPFN v2 models.
"""
import numpy as np
import pandas as pd
import pickle
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
    # Load test features
    test_k8 = pd.read_csv(DATA_DIR / "test_features_k8.csv")
    gate_df = pd.read_csv(DATA_DIR / "xgb_hybrid_features_test.csv")
    pca_path = DATA_DIR / "cls_pca_test_oof.csv"
    sample_path = DATA_DIR / "sample_submission.csv"

    # Merge features
    df = test_k8.merge(gate_df, on="game_episode", how="left")
    if df["router_gate"].isna().any():
        raise ValueError("router_gate has missing values after merge.")

    if pca_path.exists():
        df_pca = pd.read_csv(pca_path)
        df = df.merge(df_pca, on="game_episode", how="left")
        if df_pca.drop(columns=["game_episode"]).isna().any().any():
            raise ValueError("PCA features contain NaNs.")
        print(f"Added {len(df_pca.columns) - 1} PCA features")

    # Load feature columns
    feature_cols_path = MODEL_DIR / "tabpfn_feature_columns.txt"
    if feature_cols_path.exists():
        feature_cols = [line.strip() for line in feature_cols_path.read_text().splitlines() if line.strip()]
        print(f"Loaded {len(feature_cols)} feature columns from {feature_cols_path}")
    else:
        feature_cols = get_feature_columns(df)
        if "router_gate" not in feature_cols:
            feature_cols.append("router_gate")
        print(f"Using {len(feature_cols)} feature columns")

    X = df[feature_cols]
    print(f"Feature matrix shape: {X.shape}")

    # Load TabPFN models
    print("\nLoading TabPFN v2 models...")
    with open(MODEL_DIR / "tabpfn_full_x.pkl", "rb") as f:
        model_x = pickle.load(f)
    with open(MODEL_DIR / "tabpfn_full_y.pkl", "rb") as f:
        model_y = pickle.load(f)
    print("Models loaded successfully")

    # Predict
    print("\nPredicting...")
    pred_x = np.clip(model_x.predict(X), 0, FIELD_X)
    pred_y = np.clip(model_y.predict(X), 0, FIELD_Y)
    print(f"Predictions completed: X range [{pred_x.min():.2f}, {pred_x.max():.2f}], Y range [{pred_y.min():.2f}, {pred_y.max():.2f}]")

    # Create submission
    submission = pd.DataFrame({
        "game_episode": df["game_episode"],
        "end_x": pred_x,
        "end_y": pred_y,
    })

    # Reindex to match sample submission order if available
    if sample_path.exists():
        sample_ids = pd.read_csv(sample_path)["game_episode"]
        submission = submission.set_index("game_episode").reindex(sample_ids).reset_index()
        print(f"Reindexed to match sample submission order")

    # Save submission
    out_path = DATA_DIR / "submission_tabpfn_v2_hybrid.csv"
    submission.to_csv(out_path, index=False)
    print(f"\n✓ Saved submission: {out_path}")
    print(f"  Rows: {len(submission)}")
    print(f"  Columns: {list(submission.columns)}")


if __name__ == "__main__":
    main()
