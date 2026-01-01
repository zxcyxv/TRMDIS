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
MODEL_PATH = DATA_DIR / "film_smoe_transformer.pt"
BATCH_SIZE = 256
TEMPERATURE = 0.05


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    df = pd.read_csv(DATA_DIR / "test_features_v2.csv")
    X, _ = prepare_sequence_data(df)

    scaler_mean = checkpoint["scaler_mean"]
    scaler_scale = checkpoint["scaler_scale"]
    X_scaled = (X - scaler_mean) / scaler_scale

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    gate_feat = np.concatenate(gates, axis=0).reshape(-1)
    zone_logits_feat = np.concatenate(zone_logits, axis=0)

    out_df = pd.DataFrame()
    if "game_episode" in df.columns:
        out_df.insert(0, "game_episode", df["game_episode"].values)
    else:
        raise KeyError("test_features_v2.csv must include game_episode.")

    out_df["router_gate"] = gate_feat
    for i in range(zone_logits_feat.shape[1]):
        out_df[f"router_zone_logit_{i}"] = zone_logits_feat[:, i]

    out_path = DATA_DIR / "xgb_hybrid_features_test.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
