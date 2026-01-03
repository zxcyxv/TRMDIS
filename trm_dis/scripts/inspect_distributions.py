"""
Inspect layer input/output distributions for TRM+DIS.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ModelConfig
from src.data.collate import collate_episodes
from src.data.dataset import EpisodeDataset, TestEpisodeDataset, create_train_val_split
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


def _tensor_stats(x: torch.Tensor) -> dict[str, float]:
    x = x.detach()
    return {
        "shape": tuple(x.shape),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "nan": int(torch.isnan(x).sum().item()),
        "inf": int(torch.isinf(x).sum().item()),
    }


def _print_stats(label: str, stats: dict[str, float]) -> None:
    shape = stats["shape"]
    print(
        f"{label:<28} shape={shape} min={stats['min']:.4f} max={stats['max']:.4f} "
        f"mean={stats['mean']:.4f} std={stats['std']:.4f} nan={stats['nan']} inf={stats['inf']}"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect TRM+DIS layer distributions")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/workspace/TRMDIS/trm_dis/outputs/checkpoints/epoch_24.pt",
        help="Path to checkpoint .pt",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="val",
        help="Which split to inspect",
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
        "--num-batches",
        type=int,
        default=1,
        help="Number of batches to inspect",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Use EMA weights if present in checkpoint",
    )
    parser.add_argument(
        "--cond-features-path",
        type=str,
        default=None,
        help="Optional .npz path for cond features (overrides config)",
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

    cond_features = None
    if config.use_cond_features:
        cond_path = None
        if args.cond_features_path:
            cond_path = Path(args.cond_features_path)
        else:
            if args.split in {"train", "val"}:
                cond_path = Path(config.cond_feature_path)
            else:
                default_test = Path("/workspace/TRMDIS/open_track1/film_smoe_cond_features_test.npz")
                cond_path = default_test if default_test.exists() else Path(config.cond_feature_path)
        if cond_path and cond_path.exists():
            cond_features = _load_cond_features(cond_path)

    if args.split in {"train", "val"}:
        df = pd.read_csv(config.train_csv_path)
        train_df, val_df = create_train_val_split(
            df,
            val_episodes=config.val_episodes,
            seed=42,
        )
        data_df = train_df if args.split == "train" else val_df
        dataset = EpisodeDataset(
            data_df,
            encoder,
            include_target=True,
            cond_features=cond_features,
        )
    else:
        test_df = pd.read_csv(config.test_csv_path)
        base_dir = Path(config.test_csv_path).parent
        test_df["path"] = test_df["path"].apply(lambda p: str((base_dir / p).resolve()))
        dataset = TestEpisodeDataset(
            test_df,
            encoder,
            test_dir=config.test_dir_path,
            cond_features=cond_features,
        )

    loader = DataLoader(
        dataset,
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

    # Hooks to capture internal activations
    captures: list[tuple[str, torch.Tensor]] = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            captures.append((name, output))
        return hook

    hooks = [
        model.embedding.register_forward_hook(make_hook("embedding_out")),
        model.f_theta.register_forward_hook(make_hook("f_theta_out")),
        model.output_head.register_forward_hook(make_hook("output_head_out")),
    ]

    batches_done = 0
    with torch.no_grad():
        ctx = ema.average_parameters() if ema is not None else nullcontext()
        with ctx:
            for batch in loader:
                batches_done += 1
                captures.clear()
                continuous = batch["continuous"].to(device)
                categorical = batch["categorical"].to(device)
                mask = batch["mask"].to(device)
                cls_out = batch.get("cls_out")
                cond_seq = batch.get("cond_seq")
                if cls_out is not None:
                    cls_out = cls_out.to(device)
                if cond_seq is not None:
                    cond_seq = cond_seq.to(device)

                print(f"\nBatch {batches_done}")
                _print_stats("continuous", _tensor_stats(continuous))
                _print_stats("categorical", _tensor_stats(categorical.float()))
                _print_stats("mask", _tensor_stats(mask))
                if cls_out is not None:
                    _print_stats("cls_out", _tensor_stats(cls_out))
                if cond_seq is not None:
                    _print_stats("cond_seq", _tensor_stats(cond_seq))
                else:
                    print("cond_seq                  MISSING")

                outputs = model(
                    continuous,
                    categorical,
                    mask,
                    cls_out=cls_out,
                    cond_seq=cond_seq,
                )
                preds = outputs["predictions"]
                final_pred = outputs["final_pred"]
                _print_stats("predictions", _tensor_stats(preds))
                _print_stats("final_pred", _tensor_stats(final_pred))

                for name, tensor in captures:
                    _print_stats(name, _tensor_stats(tensor))

                if batches_done >= args.num_batches:
                    break

    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
