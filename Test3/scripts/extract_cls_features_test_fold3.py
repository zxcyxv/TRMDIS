"""
Extract CLS features from a single fold checkpoint for the test set (fold3).
Outputs per-episode CLS embeddings + fourier_seq + router_logits.
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


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "open_track1"
    ckpt_path = data_dir / "film_smoe_fold3.pt"
    out_path = data_dir / "film_smoe_cond_features_test_fold3.npz"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    # Load test data
    df = pd.read_csv(data_dir / "test_features_v2.csv")
    episode_ids = df["game_episode"].values
    X_seq, _ = prepare_sequence_data(df, K)
    n_samples, n_steps, n_features = X_seq.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    cls_out = np.concatenate(cls_out_list, axis=0)
    fourier_seq = np.concatenate(fourier_seq_list, axis=0)
    router_logits = np.concatenate(router_logits_list, axis=0)

    np.savez(
        out_path,
        episode_ids=episode_ids,
        cls_out=cls_out,
        fourier_seq=fourier_seq,
        router_logits=router_logits,
    )

    print(f"\n✓ Saved fold3 test features: {out_path}")
    print(f"  Shape: cls_out={cls_out.shape}, "
          f"fourier_seq={fourier_seq.shape}, "
          f"router_logits={router_logits.shape}")


if __name__ == "__main__":
    main()
