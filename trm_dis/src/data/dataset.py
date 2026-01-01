"""
PyTorch Dataset for Football Episode Data
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from .preprocessing import EpisodeEncoder


class EpisodeDataset(Dataset):
    """
    Dataset for football episode sequences

    Each sample is one episode with:
    - Variable-length action sequence (padded/truncated to max_seq_len)
    - Target: end coordinates of last action
    """

    def __init__(
        self,
        df: pd.DataFrame,
        encoder: EpisodeEncoder,
        episode_col: str = 'game_episode',
        include_target: bool = True,
        cls_features: Optional[Dict[str, 'np.ndarray']] = None,
        cond_features: Optional[Dict[str, 'np.ndarray']] = None
    ):
        """
        Args:
            df: DataFrame containing all actions
            encoder: Fitted EpisodeEncoder
            episode_col: Column name for episode ID
            include_target: Whether to include target coordinates
        """
        self.encoder = encoder
        self.episode_col = episode_col
        self.include_target = include_target
        self.cls_features = cls_features
        self.cond_features = cond_features

        # Group by episode
        self.episodes = []
        for episode_id, group in df.groupby(episode_col):
            self.episodes.append({
                'episode_id': episode_id,
                'data': group.copy()
            })

        print(f"Loaded {len(self.episodes)} episodes")

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get one episode

        Returns:
            Dictionary with torch tensors:
                - continuous: [max_seq_len, 4]
                - categorical: [max_seq_len, 3]
                - mask: [max_seq_len]
                - target: [2] (if include_target)
                - episode_id: str
        """
        episode = self.episodes[idx]
        encoded = self.encoder.encode_episode(
            episode['data'],
            include_target=self.include_target
        )

        result = {
            'continuous': torch.from_numpy(encoded['continuous']),
            'categorical': torch.from_numpy(encoded['categorical']),
            'mask': torch.from_numpy(encoded['mask']),
            'episode_id': episode['episode_id']
        }

        if self.include_target:
            result['target'] = torch.from_numpy(encoded['target'])

        if self.cls_features is not None:
            cls_vec = self.cls_features.get(str(episode['episode_id']))
            if cls_vec is not None:
                result['cls_out'] = torch.from_numpy(cls_vec)

        if self.cond_features is not None:
            cond_pack = self.cond_features.get(str(episode['episode_id']))
            if cond_pack is not None:
                fourier_seq = cond_pack['fourier_seq']
                cls_out = cond_pack['cls_out']
                router_logits = cond_pack['router_logits']
                cond_seq = self._build_cond_seq(fourier_seq, cls_out, router_logits)
                result['cond_seq'] = torch.from_numpy(cond_seq)

        return result

    def _build_cond_seq(
        self,
        fourier_seq: 'np.ndarray',  # [K, 64]
        cls_out: 'np.ndarray',      # [128]
        router_logits: 'np.ndarray' # [4]
    ) -> 'np.ndarray':
        """Build per-step conditioning sequence aligned to max_seq_len."""
        k = fourier_seq.shape[0]
        cond_dim = fourier_seq.shape[1] + cls_out.shape[0] + router_logits.shape[0]
        cond_seq = np.zeros((self.encoder.max_seq_len, cond_dim), dtype=np.float32)
        pad_len = max(self.encoder.max_seq_len - k, 0)
        fourier_trim = fourier_seq[-self.encoder.max_seq_len:] if k > self.encoder.max_seq_len else fourier_seq
        cls_rep = np.repeat(cls_out[None, :], fourier_trim.shape[0], axis=0)
        router_rep = np.repeat(router_logits[None, :], fourier_trim.shape[0], axis=0)
        merged = np.concatenate([cls_rep, fourier_trim, router_rep], axis=1).astype(np.float32)
        cond_seq[pad_len:pad_len + merged.shape[0]] = merged
        return cond_seq


class TestEpisodeDataset(Dataset):
    """
    Dataset for test episodes (loaded from individual CSV files)
    """

    def __init__(
        self,
        test_df: pd.DataFrame,
        encoder: EpisodeEncoder,
        test_dir: str = "/workspace/open_track1/test"
    ):
        """
        Args:
            test_df: Test index DataFrame with columns [game_id, game_episode, path]
            encoder: Fitted EpisodeEncoder
            test_dir: Directory containing test episode CSVs
        """
        self.test_df = test_df.reset_index(drop=True)
        self.encoder = encoder
        self.test_dir = test_dir

        print(f"Loaded {len(self.test_df)} test episodes")

    def __len__(self) -> int:
        return len(self.test_df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get one test episode

        Returns:
            Dictionary with torch tensors:
                - continuous: [max_seq_len, 4]
                - categorical: [max_seq_len, 3]
                - mask: [max_seq_len]
                - episode_id: str
        """
        row = self.test_df.iloc[idx]
        episode_id = row['game_episode']
        path = row['path']

        # Load episode data
        episode_df = pd.read_csv(path)

        # Encode (no target for test data)
        encoded = self.encoder.encode_episode(episode_df, include_target=False)

        return {
            'continuous': torch.from_numpy(encoded['continuous']),
            'categorical': torch.from_numpy(encoded['categorical']),
            'mask': torch.from_numpy(encoded['mask']),
            'episode_id': episode_id
        }


def create_train_val_split(
    df: pd.DataFrame,
    val_episodes: int = 1000,
    episode_col: str = 'game_episode',
    seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and validation sets by episodes

    Args:
        df: Full training DataFrame
        val_episodes: Number of episodes for validation
        episode_col: Column name for episode ID
        seed: Random seed

    Returns:
        (train_df, val_df)
    """
    unique_episodes = df[episode_col].unique()

    # Shuffle and split
    import numpy as np
    np.random.seed(seed)
    shuffled = np.random.permutation(unique_episodes)

    val_episode_ids = set(shuffled[:val_episodes])
    train_episode_ids = set(shuffled[val_episodes:])

    train_df = df[df[episode_col].isin(train_episode_ids)].copy()
    val_df = df[df[episode_col].isin(val_episode_ids)].copy()

    print(f"Train: {len(train_episode_ids)} episodes, {len(train_df)} actions")
    print(f"Val: {len(val_episode_ids)} episodes, {len(val_df)} actions")

    return train_df, val_df
