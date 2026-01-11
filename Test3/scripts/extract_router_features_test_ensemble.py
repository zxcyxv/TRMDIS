"""
Extract router features for test set using 5-fold ensemble.
Averages predictions from all 5 fold checkpoints.
"""
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from train_film_smoe2 import FiLMSpatialMoETransformer, prepare_sequence_data


DATA_DIR = ROOT_DIR / "open_track1"
BATCH_SIZE = 256
TEMPERATURE = 0.05
N_FOLDS = 5


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract router features for test set using 5-fold ensemble."
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=N_FOLDS,
        help="Number of folds (default: 5)",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=DATA_DIR / "xgb_hybrid_features_test.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    df = pd.read_csv(DATA_DIR / "test_features_v2.csv")
    X, _ = prepare_sequence_data(df)
    n_samples = len(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Accumulate predictions from all folds
    gate_sum = np.zeros(n_samples, dtype=np.float32)
    zone_logits_sum = np.zeros((n_samples, 4), dtype=np.float32)

    for fold_idx in range(args.n_folds):
        ckpt_path = DATA_DIR / f"film_smoe_fold{fold_idx}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"Processing fold {fold_idx}...")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        scaler_mean = checkpoint["scaler_mean"]
        scaler_scale = checkpoint["scaler_scale"]
        X_scaled = (X - scaler_mean) / scaler_scale

        model = FiLMSpatialMoETransformer(input_size=X.shape[2]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        X_t = torch.FloatTensor(X_scaled).to(device)

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

        gate_sum += gate_vals
        zone_logits_sum += zone_vals

    # Average across folds
    gate_avg = gate_sum / args.n_folds
    zone_logits_avg = zone_logits_sum / args.n_folds

    # Build output dataframe
    out_df = pd.DataFrame()
    if "game_episode" in df.columns:
        out_df.insert(0, "game_episode", df["game_episode"].values)
    else:
        raise KeyError("test_features_v2.csv must include game_episode.")

    out_df["router_gate"] = gate_avg
    for i in range(zone_logits_avg.shape[1]):
        out_df[f"router_zone_logit_{i}"] = zone_logits_avg[:, i]

    out_df.to_csv(args.out_path, index=False)
    print(f"Saved: {args.out_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
