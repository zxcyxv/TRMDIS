"""
Analyze correlation between XGBoost and TabPFN v2 predictions.
Helps determine if ensemble will be beneficial.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "open_track1"
MODEL_DIR = ROOT_DIR / "models"
K8_PATH = DATA_DIR / "train_features_k8.csv"
GATE_PATH = DATA_DIR / "xgb_hybrid_features_oof.csv"
PCA_PATH = DATA_DIR / "cls_pca_train_oof.csv"


def euclidean_distance(y_true_x, y_true_y, y_pred_x, y_pred_y):
    """Calculate mean Euclidean distance."""
    return np.mean(np.sqrt((y_true_x - y_pred_x) ** 2 + (y_true_y - y_pred_y) ** 2))


def get_feature_columns(df):
    """Extract all feature columns."""
    exclude_cols = ["game_episode", "target_end_x", "target_end_y", "last_result_name"]
    return [c for c in df.columns if c not in exclude_cols]


def main():
    print("="*70)
    print("Model Correlation Analysis: XGBoost vs TabPFN v2")
    print("="*70)

    # Load and merge features
    print("\nLoading features...")
    df_k8 = pd.read_csv(K8_PATH)
    df_gate = pd.read_csv(GATE_PATH)

    gate_cols = ["game_episode", "router_gate"]
    gate_cols += [c for c in df_gate.columns if c.startswith("router_zone_logit_")]
    df = df_k8.merge(df_gate[gate_cols], on="game_episode", how="left")

    if df["router_gate"].isna().any():
        raise ValueError("router_gate has missing values after merge.")

    # Merge PCA features
    df_pca = pd.read_csv(PCA_PATH)
    df = df.merge(df_pca, on="game_episode", how="left")

    if df_pca.drop(columns=["game_episode"]).isna().any().any():
        raise ValueError("PCA features contain NaNs.")

    print(f"  Total samples: {len(df)}")

    # Load feature columns
    feature_cols_path = MODEL_DIR / "tabpfn_feature_columns.txt"
    if feature_cols_path.exists():
        feature_cols = [line.strip() for line in feature_cols_path.read_text().splitlines() if line.strip()]
    else:
        feature_cols = get_feature_columns(df)

    print(f"  Total features: {len(feature_cols)}")

    X = df[feature_cols]
    y_x = df["target_end_x"]
    y_y = df["target_end_y"]

    # Split into train/val (80/20) - same as TabPFN validation
    X_train, X_val, y_train_x, y_val_x, y_train_y, y_val_y = train_test_split(
        X, y_x, y_y, test_size=0.2, random_state=42
    )

    print(f"  Validation set: {len(X_val)} samples")

    # ========================================================================
    # Load and predict with XGBoost (80/20 split model)
    # ========================================================================
    print("\n" + "="*70)
    print("XGBoost Predictions (80/20 model)")
    print("="*70)

    xgb_model_x = xgb.XGBRegressor()
    xgb_model_y = xgb.XGBRegressor()
    xgb_model_x.load_model(str(MODEL_DIR / "xgb_8020_x.json"))
    xgb_model_y.load_model(str(MODEL_DIR / "xgb_8020_y.json"))

    xgb_pred_x = xgb_model_x.predict(X_val)
    xgb_pred_y = xgb_model_y.predict(X_val)
    xgb_dist = euclidean_distance(y_val_x, y_val_y, xgb_pred_x, xgb_pred_y)

    print(f"  XGBoost Distance: {xgb_dist:.4f}m")

    # ========================================================================
    # Load and predict with TabPFN v2 (80/20 split model)
    # ========================================================================
    print("\n" + "="*70)
    print("TabPFN v2 Predictions (80/20 model)")
    print("="*70)

    with open(MODEL_DIR / "tabpfn_val_x.pkl", "rb") as f:
        tabpfn_model_x = pickle.load(f)
    with open(MODEL_DIR / "tabpfn_val_y.pkl", "rb") as f:
        tabpfn_model_y = pickle.load(f)

    tabpfn_pred_x = tabpfn_model_x.predict(X_val)
    tabpfn_pred_y = tabpfn_model_y.predict(X_val)
    tabpfn_dist = euclidean_distance(y_val_x, y_val_y, tabpfn_pred_x, tabpfn_pred_y)

    print(f"  TabPFN v2 Distance: {tabpfn_dist:.4f}m")

    # ========================================================================
    # Calculate correlations
    # ========================================================================
    print("\n" + "="*70)
    print("Correlation Analysis")
    print("="*70)

    # Pearson correlation for end_x predictions
    corr_x = np.corrcoef(xgb_pred_x, tabpfn_pred_x)[0, 1]
    print(f"\n  end_x predictions correlation: {corr_x:.4f}")

    # Pearson correlation for end_y predictions
    corr_y = np.corrcoef(xgb_pred_y, tabpfn_pred_y)[0, 1]
    print(f"  end_y predictions correlation: {corr_y:.4f}")

    # Average correlation
    avg_corr = (corr_x + corr_y) / 2
    print(f"  Average correlation: {avg_corr:.4f}")

    # ========================================================================
    # Test ensemble performance
    # ========================================================================
    print("\n" + "="*70)
    print("Ensemble Analysis")
    print("="*70)

    # Simple average ensemble
    ens_pred_x = (xgb_pred_x + tabpfn_pred_x) / 2
    ens_pred_y = (xgb_pred_y + tabpfn_pred_y) / 2
    ens_dist = euclidean_distance(y_val_x, y_val_y, ens_pred_x, ens_pred_y)

    print(f"\n  Individual models:")
    print(f"    XGBoost:    {xgb_dist:.4f}m")
    print(f"    TabPFN v2:  {tabpfn_dist:.4f}m")
    print(f"\n  Ensemble (50/50 average):")
    print(f"    Distance:   {ens_dist:.4f}m")

    # Compare with best individual model
    best_individual = min(xgb_dist, tabpfn_dist)
    improvement = best_individual - ens_dist

    print(f"\n  Best individual:  {best_individual:.4f}m")
    print(f"  Improvement:      {improvement:+.4f}m ({(improvement/best_individual)*100:.2f}%)")

    # ========================================================================
    # Interpretation
    # ========================================================================
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    print(f"\n  Average correlation: {avg_corr:.4f}")

    if avg_corr >= 0.95:
        print("  ⚠️  Very high correlation (≥0.95)")
        print("      → Models make very similar predictions")
        print("      → Ensemble benefit will be minimal")
        print("      → Consider using different architectures or features")
    elif avg_corr >= 0.85:
        print("  ⚠️  High correlation (0.85-0.95)")
        print("      → Models are quite similar")
        print("      → Ensemble may provide small benefit")
        print("      → Consider optimizing weights instead of 50/50")
    elif avg_corr >= 0.7:
        print("  ✓  Moderate correlation (0.70-0.85)")
        print("      → Models have some complementary behavior")
        print("      → Ensemble should provide modest improvement")
        print("      → Good candidate for weighted ensemble")
    else:
        print("  ✓✓ Low correlation (<0.70)")
        print("      → Models make quite different predictions")
        print("      → Ensemble should provide significant improvement")
        print("      → Excellent candidate for ensemble!")

    if improvement > 0:
        print(f"\n  ✅ ENSEMBLE IS BENEFICIAL!")
        print(f"     Achieved {improvement:.4f}m improvement")
    else:
        print(f"\n  ❌ ENSEMBLE NOT HELPFUL")
        print(f"     Worse by {-improvement:.4f}m")

    # ========================================================================
    # Optimal weight search
    # ========================================================================
    print("\n" + "="*70)
    print("OPTIMAL WEIGHT SEARCH")
    print("="*70)

    best_dist = float('inf')
    best_weight = 0.5

    print("\n  Testing weights (XGBoost weight):")
    for w in np.arange(0.0, 1.01, 0.1):
        ens_x = w * xgb_pred_x + (1 - w) * tabpfn_pred_x
        ens_y = w * xgb_pred_y + (1 - w) * tabpfn_pred_y
        dist = euclidean_distance(y_val_x, y_val_y, ens_x, ens_y)

        marker = " ← BEST" if dist < best_dist else ""
        print(f"    w={w:.1f}: {dist:.4f}m{marker}")

        if dist < best_dist:
            best_dist = dist
            best_weight = w

    print(f"\n  Optimal weight: {best_weight:.1f} (XGBoost) / {1-best_weight:.1f} (TabPFN)")
    print(f"  Best distance:  {best_dist:.4f}m")
    print(f"  Improvement:    {best_individual - best_dist:+.4f}m")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
