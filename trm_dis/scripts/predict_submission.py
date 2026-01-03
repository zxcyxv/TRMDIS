"""
Generate submission.csv for open_track1 test set using a TRM+DIS checkpoint.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import ModelConfig
from src.data.collate import collate_episodes
from src.data.dataset import TestEpisodeDataset
from src.data.preprocessing import EpisodeEncoder
from src.models.ema import EMA
from src.models.trm_dis import TRMDIS


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
    parser = argparse.ArgumentParser(description="TRM+DIS test submission generator")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/workspace/TRMDIS/trm_dis/outputs/checkpoints/epoch_24.pt",
        help="Path to checkpoint .pt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/workspace/TRMDIS/open_track1/submission.csv",
        help="Output submission CSV path",
    )
    parser.add_argument(
        "--cond-features-path",
        type=str,
        default=None,
        help="Optional .npz path for test cond features (overrides config)",
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

    # Encoder + vocab
    vocab_path = Path(config.checkpoint_dir) / "vocab.json"
    encoder = EpisodeEncoder()
    encoder.load_vocab(str(vocab_path))

    # Test index
    test_csv_path = Path(config.test_csv_path)
    test_df = pd.read_csv(test_csv_path)
    base_dir = test_csv_path.parent
    test_df["path"] = test_df["path"].apply(lambda p: str((base_dir / p).resolve()))

    # Conditional features (optional)
    cond_features = None
    if config.use_cond_features:
        cond_path = None
        if args.cond_features_path:
            cond_path = Path(args.cond_features_path)
        else:
            default_test = Path("/workspace/TRMDIS/open_track1/film_smoe_cond_features_test.npz")
            cond_path = default_test if default_test.exists() else Path(config.cond_feature_path)
        if cond_path and cond_path.exists():
            cond_features = _load_cond_features(cond_path)

    test_dataset = TestEpisodeDataset(
        test_df,
        encoder,
        test_dir=config.test_dir_path,
        cond_features=cond_features,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_episodes,
        pin_memory=(device == "cuda"),
    )

    model = build_model(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    ema = None
    if args.use_ema and "ema_state_dict" in checkpoint:
        ema = EMA(model, decay=config.ema_decay)
        ema.load_state_dict(checkpoint["ema_state_dict"])

    preds = []
    with torch.no_grad():
        ctx = ema.average_parameters() if ema is not None else nullcontext()
        with ctx:
            for batch in test_loader:
                continuous = batch["continuous"].to(device)
                categorical = batch["categorical"].to(device)
                mask = batch["mask"].to(device)
                cls_out = batch.get("cls_out")
                cond_vec = batch.get("cond_vec")
                if cls_out is not None:
                    cls_out = cls_out.to(device)
                if cond_vec is not None:
                    cond_vec = cond_vec.to(device)

                outputs = model(
                    continuous,
                    categorical,
                    mask,
                    cls_out=cls_out,
                    cond_vec=cond_vec,
                )
                coords = outputs["final_pred"].cpu().numpy()
                preds.append(coords)

    preds = np.concatenate(preds, axis=0)
    end_x = preds[:, 0] * encoder.field_x_max
    end_y = preds[:, 1] * encoder.field_y_max

    submission = pd.DataFrame(
        {
            "game_episode": test_df["game_episode"].values,
            "end_x": end_x,
            "end_y": end_y,
        }
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Saved submission to: {output_path}")


if __name__ == "__main__":
    main()
