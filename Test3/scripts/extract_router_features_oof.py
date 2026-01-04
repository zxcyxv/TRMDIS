import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from train_film_smoe2 import FiLMSpatialMoETransformer, prepare_sequence_data
from train_film_smoe_kfold import create_episode_kfold_splits


DATA_DIR = ROOT_DIR / "open_track1"
BATCH_SIZE = 256
TEMPERATURE = 0.05


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract OOF router features using K-fold checkpoints."
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds (default: 5)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="K-fold random state (default: 42)",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=DATA_DIR / "xgb_hybrid_features_oof.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    df = pd.read_csv(DATA_DIR / "train_features_v2.csv")
    X, _ = prepare_sequence_data(df)
    n_samples = len(df)

    fold_splits = create_episode_kfold_splits(
        df, n_splits=args.n_splits, random_state=args.random_state
    )

    gate_oof = np.zeros(n_samples, dtype=np.float32)
    zone_logits_oof = np.zeros((n_samples, 4), dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold_idx, (_, val_idx) in enumerate(fold_splits):
        ckpt_path = DATA_DIR / f"film_smoe_fold{fold_idx}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        scaler_mean = ckpt["scaler_mean"]
        scaler_scale = ckpt["scaler_scale"]

        X_val = X[val_idx]
        X_val_scaled = (X_val - scaler_mean) / scaler_scale

        model = FiLMSpatialMoETransformer(input_size=X.shape[2]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        X_t = torch.FloatTensor(X_val_scaled).to(device)

        gates = []
        zone_logits = []
        with torch.no_grad():
            for i in range(0, len(X_t), BATCH_SIZE):
                batch = X_t[i : i + BATCH_SIZE]
                outputs = model(batch, temperature=TEMPERATURE, return_all=True)
                gates.append(outputs["gate"].cpu().numpy())
                zone_logits.append(outputs["aux_zone"].cpu().numpy())

        gate_vals = np.concatenate(gates, axis=0).reshape(-1)
        zone_vals = np.concatenate(zone_logits, axis=0)

        gate_oof[val_idx] = gate_vals
        zone_logits_oof[val_idx] = zone_vals

    base_features = [
        "start_x", "start_y", "dt", "ep_idx_norm", "x_zone", "lane",
        "dist_to_goal", "angle_to_goal", "type_id", "res_id", "is_home",
        "pressure_x_weight", "is_zone14", "angle_visible",
    ]
    masked_features = [
        "end_x", "end_y", "dx", "dy", "dist", "speed",
        "action_angle", "action_progress", "action_dist", "action_lateral",
    ]

    base_cols = [f"{feat}_7" for feat in base_features if f"{feat}_7" in df.columns]
    masked_cols = [f"{feat}_6" for feat in masked_features if f"{feat}_6" in df.columns]

    if not base_cols:
        raise KeyError("No base feature columns found for t=7.")

    out_df = df[base_cols + masked_cols].copy()
    if "game_episode" in df.columns:
        out_df.insert(0, "game_episode", df["game_episode"].values)
    out_df["router_gate"] = gate_oof
    for i in range(zone_logits_oof.shape[1]):
        out_df[f"router_zone_logit_{i}"] = zone_logits_oof[:, i]
    out_df["target_end_x"] = df["target_end_x"].values
    out_df["target_end_y"] = df["target_end_y"].values

    out_df.to_csv(args.out_path, index=False)
    print(f"Saved: {args.out_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
