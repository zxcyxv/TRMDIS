"""
Validate a TRM+DIS checkpoint on the validation split.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ModelConfig
from src.data.collate import collate_episodes
from src.data.dataset import EpisodeDataset, create_train_val_split
from src.data.preprocessing import EpisodeEncoder
from src.models.trm_dis import TRMDIS
from src.training.trainer import Trainer


def _load_cond_features(path: Path) -> dict[str, dict[str, np.ndarray]]:
    data = np.load(path, allow_pickle=True)
    ids = data["episode_ids"]
    cls_out = data["cls_out"]
    fourier_seq = data["fourier_seq"]
    router_logits = data["router_logits"]
    return {
        str(eid): {
            "cls_out": cls_out[i],
            "fourier_seq": fourier_seq[i],
            "router_logits": router_logits[i],
        }
        for i, eid in enumerate(ids)
    }


def build_model(config: ModelConfig) -> TRMDIS:
    return TRMDIS(
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        n_action_types=config.n_action_types,
        n_result_types=config.n_result_types,
        n_sup=config.n_sup,
        n=config.n,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
        cls_dim=config.cls_dim,
        use_cls_only=config.use_cls_only,
        use_cond_features=config.use_cond_features,
        cond_dim=config.cond_dim,
        use_spectral_norm=config.use_spectral_norm,
        use_dynamic_stopping=config.use_dynamic_stopping,
        stopping_threshold=config.stopping_threshold,
        stopping_kappa=config.stopping_kappa,
        use_denoising_training=config.use_denoising_training,
        denoising_sigma_init=config.denoising_sigma_init,
        denoising_sigma_final=config.denoising_sigma_final,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate TRM+DIS checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/workspace/TRMDIS/trm_dis/outputs/checkpoints/epoch_24.pt",
        help="Path to checkpoint .pt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda or cpu (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size override (default: from config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for train/val split",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Use EMA weights if present in checkpoint",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", ModelConfig())
    state_dict = checkpoint["model_state_dict"]

    has_spectral_norm = any(k.endswith("weight_orig") for k in state_dict.keys())
    has_stop_gate = any(k.startswith("stop_gate_proj") for k in state_dict.keys())
    config.use_spectral_norm = has_spectral_norm
    config.use_dynamic_stopping = has_stop_gate

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or config.batch_size

    vocab_path = Path(config.checkpoint_dir) / "vocab.json"
    encoder = EpisodeEncoder()
    encoder.load_vocab(str(vocab_path))

    df = pd.read_csv(config.train_csv_path)
    train_df, val_df = create_train_val_split(
        df,
        val_episodes=config.val_episodes,
        seed=args.seed,
    )

    cls_features = None
    cond_features = None
    if config.use_cond_features and config.cond_feature_path:
        cond_features = _load_cond_features(Path(config.cond_feature_path))

    train_dataset = EpisodeDataset(
        train_df,
        encoder,
        include_target=True,
        cls_features=cls_features,
        cond_features=cond_features,
    )
    val_dataset = EpisodeDataset(
        val_df,
        encoder,
        include_target=True,
        cls_features=cls_features,
        cond_features=cond_features,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_episodes,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_episodes,
        pin_memory=(device == "cuda"),
    )

    model = build_model(config).to(device)
    model.load_state_dict(state_dict)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )
    if args.use_ema and "ema_state_dict" in checkpoint and trainer.ema is not None:
        trainer.ema.load_state_dict(checkpoint["ema_state_dict"])

    metrics = trainer.validate(use_ema=args.use_ema)
    print("\nValidation metrics:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k2, v2 in value.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
