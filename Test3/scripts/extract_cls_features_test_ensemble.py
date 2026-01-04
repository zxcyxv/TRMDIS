"""
Extract CLS features from K-Fold trained FiLM+Spatial MoE models for test set.
Averages predictions from all 5 fold models to create ensemble test features.
Outputs a per-episode CLS embedding map for TRMDIS usage at test-time.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_film_smoe2 import (  # noqa: E402
    FiLMSpatialMoETransformer,
    prepare_sequence_data,
    K,
)


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "open_track1"
    out_path = data_dir / "film_smoe_cond_features_test_oof.npz"
    n_folds = 5

    # Load test data
    df = pd.read_csv(data_dir / "test_features_v2.csv")
    episode_ids = df["game_episode"].values
    X_seq, _ = prepare_sequence_data(df, K)
    n_samples, n_steps, n_features = X_seq.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize ensemble accumulators
    cls_out_ensemble = np.zeros((n_samples, 128), dtype=np.float32)
    fourier_seq_ensemble = np.zeros((n_samples, n_steps, 64), dtype=np.float32)
    router_logits_ensemble = np.zeros((n_samples, 4), dtype=np.float32)

    print(f"Ensembling {n_folds} fold models for {n_samples} test samples...")

    # Loop through all fold models
    for fold_idx in range(n_folds):
        ckpt_path = data_dir / f"film_smoe_fold{fold_idx}.pt"
        print(f"\nFold {fold_idx}: Loading {ckpt_path.name}")

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        scaler_mean = ckpt["scaler_mean"]
        scaler_scale = ckpt["scaler_scale"]

        # Normalize with fold-specific scaler
        X_flat = X_seq.reshape(-1, n_features)
        X_norm = (X_flat - scaler_mean) / scaler_scale
        X_norm = X_norm.reshape(n_samples, n_steps, n_features)

        # Initialize model
        model = FiLMSpatialMoETransformer(
            input_size=n_features,
            d_model=128,
            nhead=4,
            num_layers=2,
            dropout=0.2,
            fourier_mapping_size=32,
            fourier_scale=5.0,
            film_hidden_dim=64,
            router_hidden_dim=32,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Extract features for this fold
        cls_out_list = []
        fourier_seq_list = []
        router_logits_list = []

        with torch.no_grad():
            for i in range(0, n_samples, 512):
                batch = torch.tensor(
                    X_norm[i : i + 512], dtype=torch.float32, device=device
                )
                outputs = model(batch, return_all=True)
                cls_out_list.append(outputs["cls_out"].cpu().numpy())

                coord_features = batch[:, :, model.coord_indices]
                fourier_features = model.fourier_layer(coord_features)
                fourier_seq_list.append(fourier_features.cpu().numpy())

                router_logits_list.append(outputs["aux_zone"].cpu().numpy())

        # Accumulate features (will average later)
        cls_out_ensemble += np.concatenate(cls_out_list, axis=0)
        fourier_seq_ensemble += np.concatenate(fourier_seq_list, axis=0)
        router_logits_ensemble += np.concatenate(router_logits_list, axis=0)

        print(f"  Extracted features: cls_out {cls_out_ensemble.shape}")

    # Average across folds
    cls_out_ensemble /= n_folds
    fourier_seq_ensemble /= n_folds
    router_logits_ensemble /= n_folds

    # Save ensemble features
    np.savez(
        out_path,
        episode_ids=episode_ids,
        cls_out=cls_out_ensemble,
        fourier_seq=fourier_seq_ensemble,
        router_logits=router_logits_ensemble,
    )

    print(f"\n✓ Saved ensemble test features: {out_path}")
    print(f"  Shape: cls_out={cls_out_ensemble.shape}, "
          f"fourier_seq={fourier_seq_ensemble.shape}, "
          f"router_logits={router_logits_ensemble.shape}")


if __name__ == "__main__":
    main()
