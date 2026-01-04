"""
Validate fold3 router logits consistency between CSV and NPZ outputs.
Compares xgb_hybrid_features_test_fold3.csv vs film_smoe_cond_features_test_fold3.npz.
"""
from pathlib import Path
import csv
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "open_track1"

CSV_PATH = DATA_DIR / "xgb_hybrid_features_test_fold3.csv"
NPZ_PATH = DATA_DIR / "film_smoe_cond_features_test_fold3.npz"


def load_csv_logits(path: Path):
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        idx_map = {name: i for i, name in enumerate(header)}
        cols = [f"router_zone_logit_{i}" for i in range(4)]
        missing = [c for c in cols if c not in idx_map]
        if missing:
            raise KeyError(f"Missing columns in CSV: {missing}")
        ids = []
        logits = []
        for row in reader:
            ids.append(row[idx_map["game_episode"]])
            logits.append([float(row[idx_map[c]]) for c in cols])
    return ids, np.asarray(logits, dtype=np.float32)


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV: {CSV_PATH}")
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"Missing NPZ: {NPZ_PATH}")

    csv_ids, csv_logits = load_csv_logits(CSV_PATH)
    npz = np.load(NPZ_PATH, allow_pickle=True)
    npz_ids = [str(x) for x in npz["episode_ids"]]
    npz_logits = np.asarray(npz["router_logits"], dtype=np.float32)

    if len(csv_ids) != len(npz_ids):
        raise ValueError(f"Row count mismatch: CSV={len(csv_ids)} NPZ={len(npz_ids)}")

    # Align by episode_id to avoid ordering differences
    npz_map = {eid: i for i, eid in enumerate(npz_ids)}
    aligned = np.zeros_like(csv_logits)
    missing = 0
    for i, eid in enumerate(csv_ids):
        idx = npz_map.get(eid)
        if idx is None:
            missing += 1
            continue
        aligned[i] = npz_logits[idx]

    if missing:
        raise ValueError(f"Missing {missing} episode_ids in NPZ.")

    diff = csv_logits - aligned
    max_abs = np.max(np.abs(diff))
    mean_abs = np.mean(np.abs(diff))

    print("Router logits comparison (CSV vs NPZ, fold3)")
    print(f"  max_abs_diff: {max_abs:.8f}")
    print(f"  mean_abs_diff: {mean_abs:.8f}")

    if max_abs > 1e-6:
        print("  ⚠ Differences detected. Check scaling/model consistency.")
    else:
        print("  ✓ Logits match within tolerance.")


if __name__ == "__main__":
    main()
