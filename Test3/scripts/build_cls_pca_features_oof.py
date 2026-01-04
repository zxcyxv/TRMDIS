"""
Build PCA features from FiLM+Spatial MoE CLS embeddings (OOF version).
Uses OOF (Out-of-Fold) features to prevent data leakage.
Outputs train/test CSVs with 16 PCA components.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "open_track1"
TRAIN_NPZ = DATA_DIR / "film_smoe_cond_features_oof.npz"
TEST_NPZ = DATA_DIR / "film_smoe_cond_features_test_oof.npz"
TRAIN_OUT = DATA_DIR / "cls_pca_train_oof.csv"
TEST_OUT = DATA_DIR / "cls_pca_test_oof.csv"


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

    pca = PCA(n_components=16, random_state=42)
    train_pca = pca.fit_transform(train_cls)
    test_pca = pca.transform(test_cls)

    cols = [f"cls_pca_{i}" for i in range(train_pca.shape[1])]
    train_df = pd.DataFrame(train_pca, columns=cols)
    train_df.insert(0, "game_episode", train_ids)
    train_df.to_csv(TRAIN_OUT, index=False)

    test_df = pd.DataFrame(test_pca, columns=cols)
    test_df.insert(0, "game_episode", test_ids)
    test_df.to_csv(TEST_OUT, index=False)

    print(f"✓ Saved OOF PCA features:")
    print(f"  Train: {TRAIN_OUT} ({len(train_df)} rows)")
    print(f"  Test:  {TEST_OUT} ({len(test_df)} rows)")


if __name__ == "__main__":
    main()
