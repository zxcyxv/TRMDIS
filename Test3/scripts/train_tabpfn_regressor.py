"""
Train TabPFN v2 regressor directly from HuggingFace for end_x and end_y prediction.

TabPFN v2 is a transformer-based foundation model for tabular data.

References:
- HuggingFace: Prior-Labs/TabPFN-v2-reg
- Paper: TabPFN (Nature, January 2025)
- GitHub: https://github.com/PriorLabs/TabPFN
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle

try:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
except ImportError:
    raise ImportError(
        "TabPFN not installed. Run: pip install tabpfn"
    )


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "open_track1"
MODEL_DIR = ROOT_DIR / "models"
K8_PATH = DATA_DIR / "train_features_k8.csv"
GATE_PATH = DATA_DIR / "xgb_hybrid_features.csv"
PCA_PATH = DATA_DIR / "cls_pca_train.csv"


def euclidean_distance(y_true_x, y_true_y, y_pred_x, y_pred_y):
    """Calculate mean Euclidean distance between predictions and ground truth."""
    return np.mean(np.sqrt((y_true_x - y_pred_x) ** 2 + (y_true_y - y_pred_y) ** 2))


def get_feature_columns(df):
    """Extract all feature columns (excluding episode ID and targets)."""
    exclude_cols = ["game_episode", "target_end_x", "target_end_y", "last_result_name"]
    return [c for c in df.columns if c not in exclude_cols]


def prepare_data():
    """Load and merge all feature sources."""
    print("Loading feature data...")
    df_k8 = pd.read_csv(K8_PATH)
    df_gate = pd.read_csv(GATE_PATH)

    if "game_episode" not in df_gate.columns:
        raise KeyError("xgb_hybrid_features.csv must include game_episode for merge.")

    # Merge router features
    gate_cols = ["game_episode", "router_gate"]
    gate_cols += [c for c in df_gate.columns if c.startswith("router_zone_logit_")]
    df = df_k8.merge(df_gate[gate_cols], on="game_episode", how="left")

    if df["router_gate"].isna().any():
        raise ValueError("router_gate has missing values after merge.")

    # Merge PCA features if available
    if PCA_PATH.exists():
        print(f"Loading PCA features from {PCA_PATH}")
        df_pca = pd.read_csv(PCA_PATH)
        df = df.merge(df_pca, on="game_episode", how="left")
        if df_pca.drop(columns=["game_episode"]).isna().any().any():
            raise ValueError("PCA features contain NaNs.")
        print(f"  Added {len(df_pca.columns) - 1} PCA features")
    else:
        print("  No PCA features found, skipping.")

    feature_cols = get_feature_columns(df)
    # Ensure router_gate is included (same as XGBoost code)
    if "router_gate" not in feature_cols:
        feature_cols.append("router_gate")
    print(f"Total features: {len(feature_cols)}")

    return df, feature_cols


def train_tabpfn_validation(df, feature_cols):
    """Train TabPFN v2 with 80/20 train/val split."""
    print(f"\nTraining TabPFN v2 with 80/20 split...")

    # Split data 80/20
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"  Train: {len(train_df)} samples, Val: {len(val_df)} samples")

    X_train = train_df[feature_cols].values
    y_train_x = train_df["target_end_x"].values
    y_train_y = train_df["target_end_y"].values

    X_val = val_df[feature_cols].values
    y_val_x = val_df["target_end_x"].values
    y_val_y = val_df["target_end_y"].values

    # Train TabPFN v2 for end_x
    print("\n  Training TabPFN v2 for end_x...")
    model_x = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2,
        ignore_pretraining_limits=True
    )
    model_x.fit(X_train, y_train_x)

    # Train TabPFN v2 for end_y
    print("  Training TabPFN v2 for end_y...")
    model_y = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2,
        ignore_pretraining_limits=True
    )
    model_y.fit(X_train, y_train_y)

    # Predict on validation set
    print("  Predicting on validation set...")
    val_pred_x = model_x.predict(X_val)
    val_pred_y = model_y.predict(X_val)

    # Calculate validation distance
    val_dist = euclidean_distance(y_val_x, y_val_y, val_pred_x, val_pred_y)
    print(f"\n[Validation Result] Distance: {val_dist:.4f}m")

    # Save validation models
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    with open(MODEL_DIR / "tabpfn_val_x.pkl", "wb") as f:
        pickle.dump(model_x, f)
    with open(MODEL_DIR / "tabpfn_val_y.pkl", "wb") as f:
        pickle.dump(model_y, f)

    print(f"Saved validation models to {MODEL_DIR}")

    return model_x, model_y, val_dist


def train_tabpfn_full(df, feature_cols):
    """Train full TabPFN v2 models on all data for final submission."""
    print("\n" + "="*50)
    print("Training final TabPFN v2 models on all data...")
    print("="*50)
    print(f"  Using all {len(df)} samples")

    X_train = df[feature_cols].values
    y_train_x = df["target_end_x"].values
    y_train_y = df["target_end_y"].values

    # Train full model for end_x
    print("\n  Training full TabPFN v2 for end_x...")
    model_x = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2,
        ignore_pretraining_limits=True
    )
    model_x.fit(X_train, y_train_x)

    # Train full model for end_y
    print("  Training full TabPFN v2 for end_y...")
    model_y = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2,
        ignore_pretraining_limits=True
    )
    model_y.fit(X_train, y_train_y)

    # Save full models
    with open(MODEL_DIR / "tabpfn_full_x.pkl", "wb") as f:
        pickle.dump(model_x, f)
    with open(MODEL_DIR / "tabpfn_full_y.pkl", "wb") as f:
        pickle.dump(model_y, f)

    # Save feature columns for inference
    (MODEL_DIR / "tabpfn_feature_columns.txt").write_text("\n".join(feature_cols))
    print(f"\nSaved feature columns to {MODEL_DIR / 'tabpfn_feature_columns.txt'}")

    return model_x, model_y


def main():
    """Main training pipeline."""
    # Prepare data
    df, feature_cols = prepare_data()

    # Train with 80/20 split for validation
    model_x, model_y, val_dist = train_tabpfn_validation(df, feature_cols)

    # Train full models on all data
    final_model_x, final_model_y = train_tabpfn_full(df, feature_cols)

    print("\n=== Training Complete ===")
    print(f"Validation Distance: {val_dist:.4f}m")
    print(f"Validation models: {MODEL_DIR}/tabpfn_val_x.pkl and {MODEL_DIR}/tabpfn_val_y.pkl")
    print(f"Final models: {MODEL_DIR}/tabpfn_full_x.pkl and {MODEL_DIR}/tabpfn_full_y.pkl")


if __name__ == "__main__":
    main()
