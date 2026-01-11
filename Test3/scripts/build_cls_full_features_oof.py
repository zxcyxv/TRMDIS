"""
Build full 128-dim CLS features (no PCA) for comparison.
"""
from pathlib import Path
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "open_track1"
TRAIN_NPZ = DATA_DIR / "film_smoe_cond_features_oof.npz"
TEST_NPZ = DATA_DIR / "film_smoe_cond_features_test_oof.npz"
TRAIN_OUT = DATA_DIR / "cls_full_train_oof.csv"
TEST_OUT = DATA_DIR / "cls_full_test_oof.csv"


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    episode_ids = data["episode_ids"].astype(str)
    cls_out = data["cls_out"].astype(np.float32)
    return episode_ids, cls_out


def main() -> None:
    if not TRAIN_NPZ.exists():
        raise FileNotFoundError(f"Missing train CLS npz: {TRAIN_NPZ}")
    if not TEST_NPZ.exists():
        raise FileNotFoundError(f"Missing test CLS npz: {TEST_NPZ}")

    train_ids, train_cls = load_npz(TRAIN_NPZ)
    test_ids, test_cls = load_npz(TEST_NPZ)

    print(f"Train CLS shape: {train_cls.shape}")
    print(f"Test CLS shape: {test_cls.shape}")

    # Create column names: cls_0, cls_1, ..., cls_127
    cols = [f"cls_{i}" for i in range(train_cls.shape[1])]

    train_df = pd.DataFrame(train_cls, columns=cols)
    train_df.insert(0, "game_episode", train_ids)
    train_df.to_csv(TRAIN_OUT, index=False)

    test_df = pd.DataFrame(test_cls, columns=cols)
    test_df.insert(0, "game_episode", test_ids)
    test_df.to_csv(TEST_OUT, index=False)

    print(f"✓ Saved full 128-dim CLS features:")
    print(f"  Train: {TRAIN_OUT} ({len(train_df)} rows)")
    print(f"  Test:  {TEST_OUT} ({len(test_df)} rows)")


if __name__ == "__main__":
    main()
