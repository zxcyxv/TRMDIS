"""
Evaluate TRM+DIS + XGB hybrid stacking on the validation split.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ModelConfig
from src.data.collate import collate_episodes
from src.data.dataset import EpisodeDataset, create_train_val_split
from src.data.preprocessing import EpisodeEncoder
from src.models.ema import EMA
from src.models.trm_dis import TRMDIS


FIELD_X = 105.0
FIELD_Y = 68.0


def create_boundary_labels(end_x: np.ndarray, end_y: np.ndarray) -> np.ndarray:
    """Zone labels: 0=In-field, 1=Top-out, 2=Bottom-out, 3=Goal-line."""
    labels = np.zeros(len(end_x), dtype=int)
    labels[(end_x > FIELD_X - 5) | (end_x < 5)] = 3
    labels[(labels == 0) & (end_y > FIELD_Y - 5)] = 1
    labels[(labels == 0) & (end_y < 5)] = 2
    return labels


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude_cols = ["game_episode", "target_end_x", "target_end_y", "last_result_name"]
    return [c for c in df.columns if c not in exclude_cols]


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
        diffusion_type=getattr(config, "diffusion_type", "linear"),
        hybrid_noise_scale=getattr(config, "hybrid_noise_scale", 0.05),
        gaussian_sigma_init=getattr(config, "gaussian_sigma_init", 0.3),
        gaussian_sigma_final=getattr(config, "gaussian_sigma_final", 0.0),
    )


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


def mae_euclid(y_true_x, y_true_y, y_pred_x, y_pred_y) -> tuple[float, float]:
    mae_x = np.mean(np.abs(y_true_x - y_pred_x))
    mae_y = np.mean(np.abs(y_true_y - y_pred_y))
    euclid = np.mean(np.sqrt((y_true_x - y_pred_x) ** 2 + (y_true_y - y_pred_y) ** 2))
    return float((mae_x + mae_y) / 2.0), float(euclid)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Stacking meta-learner evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/workspace/TRMDIS/trm_dis/outputs/checkpoints/epoch_24.pt",
        help="TRM+DIS checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda or cpu (default: auto)",
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
    parser.add_argument(
        "--mlp",
        action="store_true",
        help="Also evaluate a shallow MLP meta-learner",
    )
    parser.add_argument(
        "--zone-source",
        type=str,
        choices=["none", "trm", "xgb", "mean"],
        default="none",
        help="Use predicted zone_id from specified source",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", ModelConfig())
    state_dict = checkpoint["model_state_dict"]

    has_spectral_norm = any(k.endswith("weight_orig") for k in state_dict.keys())
    has_stop_gate = any(k.startswith("stop_gate_proj") for k in state_dict.keys())
    config.use_spectral_norm = has_spectral_norm
    config.use_dynamic_stopping = has_stop_gate
    if "diffusion.noise_weights" not in state_dict:
        config.diffusion_type = "linear"

    df = pd.read_csv(config.train_csv_path)
    train_df, val_df = create_train_val_split(
        df,
        val_episodes=config.val_episodes,
        seed=args.seed,
    )
    val_episode_ids = set(val_df["game_episode"].unique())

    # Ground truth from last action
    idx = val_df.groupby("game_episode")["action_id"].idxmax()
    last_actions = val_df.loc[idx].copy()
    y_true_x = last_actions["end_x"].values
    y_true_y = last_actions["end_y"].values

    # Last result success flag (1=Successful, else 0)
    last_result = last_actions["result_name"].fillna("")
    last_success = (last_result == "Successful").astype(int).values

    # Router logits from XGB hybrid features
    gate_path = Path("/workspace/TRMDIS/open_track1/xgb_hybrid_features.csv")
    df_gate = pd.read_csv(gate_path)
    gate_cols = ["game_episode"] + [c for c in df_gate.columns if c.startswith("router_zone_logit_")]
    df_gate = df_gate[gate_cols].copy()
    df_gate = df_gate[df_gate["game_episode"].isin(val_episode_ids)]
    gate_map = {
        ge: df_gate.loc[df_gate["game_episode"] == ge].iloc[0].values[1:].astype(float)
        for ge in df_gate["game_episode"].values
    }
    router_logits = np.stack([gate_map[ge] for ge in last_actions["game_episode"].values], axis=0)

    # TRM predictions
    vocab_path = Path(config.checkpoint_dir) / "vocab.json"
    encoder = EpisodeEncoder()
    encoder.load_vocab(str(vocab_path))

    cond_features = None
    if config.use_cond_features and config.cond_feature_path:
        cond_features = _load_cond_features(Path(config.cond_feature_path))

    val_dataset = EpisodeDataset(
        val_df,
        encoder,
        include_target=True,
        cond_features=cond_features,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
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

    trm_preds = []
    trm_episode_ids = []
    with torch.no_grad():
        ctx = ema.average_parameters() if ema is not None else nullcontext()
        with ctx:
            for batch in val_loader:
                continuous = batch["continuous"].to(device)
                categorical = batch["categorical"].to(device)
                mask = batch["mask"].to(device)
                cls_out = batch.get("cls_out")
                cond_seq = batch.get("cond_seq")
                if cls_out is not None:
                    cls_out = cls_out.to(device)
                if cond_seq is not None:
                    cond_seq = cond_seq.to(device)
                outputs = model(
                    continuous,
                    categorical,
                    mask,
                    cls_out=cls_out,
                    cond_seq=cond_seq,
                )
                coords = outputs["final_pred"].cpu().numpy()
                trm_preds.append(coords)
                trm_episode_ids.extend(batch["episode_ids"])

    trm_preds = np.concatenate(trm_preds, axis=0)
    trm_x = trm_preds[:, 0] * config.field_x_max
    trm_y = trm_preds[:, 1] * config.field_y_max

    # XGB predictions on same val episodes
    k8_path = Path("/workspace/TRMDIS/open_track1/train_features_k8.csv")
    df_k8 = pd.read_csv(k8_path)
    df_gate_full = pd.read_csv("/workspace/TRMDIS/open_track1/xgb_hybrid_features.csv")
    gate_cols_full = ["game_episode", "router_gate"]
    gate_cols_full += [c for c in df_gate_full.columns if c.startswith("router_zone_logit_")]
    df_all = df_k8.merge(df_gate_full[gate_cols_full], on="game_episode", how="left")

    feature_cols = get_feature_columns(df_all)
    if "router_gate" not in feature_cols:
        feature_cols.append("router_gate")

    df_val = df_all[df_all["game_episode"].isin(val_episode_ids)].copy()
    df_train = df_all[~df_all["game_episode"].isin(val_episode_ids)].copy()

    X_train = df_train[feature_cols]
    X_val = df_val[feature_cols]
    yx_train = df_train["target_end_x"].values
    yy_train = df_train["target_end_y"].values

    params = {
        "objective": "reg:absoluteerror",
        "tree_method": "hist",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_depth": 7,
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 50,
        "verbosity": 0,
    }

    model_x = xgb.XGBRegressor(**params)
    model_y = xgb.XGBRegressor(**params)
    model_x.fit(X_train, yx_train, eval_set=[(X_val, df_val["target_end_x"].values)], verbose=False)
    model_y.fit(X_train, yy_train, eval_set=[(X_val, df_val["target_end_y"].values)], verbose=False)

    xgb_x = model_x.predict(X_val)
    xgb_y = model_y.predict(X_val)

    # Align ordering to val_df (game_episode order from last_actions)
    val_order = last_actions["game_episode"].values
    pred_map = {
        ge: (x, y) for ge, x, y in zip(df_val["game_episode"].values, xgb_x, xgb_y)
    }
    xgb_x_ordered = np.array([pred_map[ge][0] for ge in val_order])
    xgb_y_ordered = np.array([pred_map[ge][1] for ge in val_order])

    # Meta-features
    features = [
        trm_x,
        trm_y,
        xgb_x_ordered,
        xgb_y_ordered,
        router_logits[:, 0],
        router_logits[:, 1],
        router_logits[:, 2],
        router_logits[:, 3],
        last_success.astype(float),
    ]

    if args.zone_source != "none":
        if args.zone_source == "trm":
            z_src_x, z_src_y = trm_x, trm_y
        elif args.zone_source == "xgb":
            z_src_x, z_src_y = xgb_x_ordered, xgb_y_ordered
        else:
            z_src_x = (trm_x + xgb_x_ordered) / 2.0
            z_src_y = (trm_y + xgb_y_ordered) / 2.0
        zone_id = create_boundary_labels(z_src_x, z_src_y)
        zone_onehot = np.eye(4)[zone_id]
        for i in range(4):
            features.append(zone_onehot[:, i])

    X_meta = np.stack(features, axis=1)

    # Ridge meta-learner
    ridge_x = Ridge(alpha=1.0)
    ridge_y = Ridge(alpha=1.0)
    ridge_x.fit(X_meta, y_true_x)
    ridge_y.fit(X_meta, y_true_y)
    meta_x = ridge_x.predict(X_meta)
    meta_y = ridge_y.predict(X_meta)

    mae_avg, euclid = mae_euclid(y_true_x, y_true_y, meta_x, meta_y)
    print(f"\nMeta (Ridge) MAE(avg): {mae_avg:.3f}m, Euclid: {euclid:.3f}m")
    if args.mlp:
        mlp_x = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=500,
            random_state=args.seed,
        )
        mlp_y = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=500,
            random_state=args.seed,
        )
        mlp_x.fit(X_meta, y_true_x)
        mlp_y.fit(X_meta, y_true_y)
        mlp_pred_x = mlp_x.predict(X_meta)
        mlp_pred_y = mlp_y.predict(X_meta)
        mlp_mae, mlp_euclid = mae_euclid(y_true_x, y_true_y, mlp_pred_x, mlp_pred_y)
        print(f"Meta (MLP) MAE(avg): {mlp_mae:.3f}m, Euclid: {mlp_euclid:.3f}m")

    # Baselines
    trm_mae, trm_euclid = mae_euclid(y_true_x, y_true_y, trm_x, trm_y)
    xgb_mae, xgb_euclid = mae_euclid(y_true_x, y_true_y, xgb_x_ordered, xgb_y_ordered)
    mean_mae, mean_euclid = mae_euclid(
        y_true_x, y_true_y,
        (trm_x + xgb_x_ordered) / 2.0,
        (trm_y + xgb_y_ordered) / 2.0,
    )
    print(f"TRM+DIS MAE(avg): {trm_mae:.3f}m, Euclid: {trm_euclid:.3f}m")
    print(f"XGB Hybrid MAE(avg): {xgb_mae:.3f}m, Euclid: {xgb_euclid:.3f}m")
    print(f"Mean Ensemble MAE(avg): {mean_mae:.3f}m, Euclid: {mean_euclid:.3f}m")


if __name__ == "__main__":
    from contextlib import nullcontext

    main()
