"""
Data preprocessing for TRM+DIS Football Coordinate Prediction
Handles normalization, encoding, and sequence preparation
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class EpisodeEncoder:
    """
    Encodes football episode data into normalized sequences

    Critical design:
    - Preserves sequence dimension (NO mean pooling)
    - Normalizes coordinates to [0, 1]
    - Handles variable-length sequences with padding/truncation
    """

    def __init__(
        self,
        max_seq_len: int = 16,
        field_x_max: float = 105.0,
        field_y_max: float = 68.0
    ):
        self.max_seq_len = max_seq_len
        self.field_x_max = field_x_max
        self.field_y_max = field_y_max

        # Build encoding dictionaries
        self.type_to_id: Dict[str, int] = {}
        self.result_to_id: Dict[str, int] = {}

    def fit(self, df: pd.DataFrame) -> 'EpisodeEncoder':
        """
        Build encoding dictionaries from training data

        Args:
            df: Training DataFrame

        Returns:
            self
        """
        # Action type encoding (26 types)
        unique_types = sorted(df['type_name'].unique())
        self.type_to_id = {t: i for i, t in enumerate(unique_types)}

        # Result encoding (8 types + NaN)
        unique_results = sorted(df['result_name'].dropna().unique())
        self.result_to_id = {r: i for i, r in enumerate(unique_results)}
        self.result_to_id['__NAN__'] = len(self.result_to_id)  # NaN as separate category

        return self

    def encode_episode(
        self,
        episode_df: pd.DataFrame,
        include_target: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Encode a single episode into fixed-length sequences

        Args:
            episode_df: Episode DataFrame (sorted by action_id)
            include_target: Whether to include target coordinates

        Returns:
            Dictionary containing:
                - continuous: [max_seq_len, 4] - (start_x, start_y, time_norm, padding_flag)
                - categorical: [max_seq_len, 3] - (type_id, result_id, is_home)
                - mask: [max_seq_len] - attention mask (1=valid, 0=padding)
                - target: [2] - (end_x, end_y) normalized [0, 1] (if include_target)
        """
        # Sort by action_id to ensure temporal order
        episode_df = episode_df.sort_values('action_id').reset_index(drop=True)

        seq_len = len(episode_df)

        # Extract features
        start_x = episode_df['start_x'].values / self.field_x_max  # Normalize
        start_y = episode_df['start_y'].values / self.field_y_max

        # Normalize time relative to episode duration
        time_seconds = episode_df['time_seconds'].values
        time_min = time_seconds[0] if seq_len > 0 else 0
        time_max = time_seconds[-1] if seq_len > 0 else 1
        time_norm = (time_seconds - time_min) / (time_max - time_min + 1e-8)

        # Categorical features
        type_ids = episode_df['type_name'].map(self.type_to_id).values
        result_ids = episode_df['result_name'].map(
            lambda x: self.result_to_id.get(x, self.result_to_id['__NAN__'])
            if pd.notna(x) else self.result_to_id['__NAN__']
        ).values
        is_home = episode_df['is_home'].astype(np.float32).values

        # Truncate or pad sequence
        if seq_len > self.max_seq_len:
            # Take last max_seq_len actions
            start_idx = seq_len - self.max_seq_len
            continuous = np.stack([
                start_x[start_idx:],
                start_y[start_idx:],
                time_norm[start_idx:],
                np.ones(self.max_seq_len)  # All valid (no padding)
            ], axis=1).astype(np.float32)

            categorical = np.stack([
                type_ids[start_idx:],
                result_ids[start_idx:],
                is_home[start_idx:]
            ], axis=1).astype(np.int64)

            mask = np.ones(self.max_seq_len, dtype=np.float32)

        else:
            # Pad at the beginning (prepend zeros)
            pad_len = self.max_seq_len - seq_len

            continuous = np.zeros((self.max_seq_len, 4), dtype=np.float32)
            continuous[pad_len:, 0] = start_x
            continuous[pad_len:, 1] = start_y
            continuous[pad_len:, 2] = time_norm
            continuous[pad_len:, 3] = 1.0  # Valid flag

            categorical = np.zeros((self.max_seq_len, 3), dtype=np.int64)
            categorical[pad_len:, 0] = type_ids
            categorical[pad_len:, 1] = result_ids
            categorical[pad_len:, 2] = is_home

            mask = np.zeros(self.max_seq_len, dtype=np.float32)
            mask[pad_len:] = 1.0

        result = {
            'continuous': continuous,
            'categorical': categorical,
            'mask': mask
        }

        # Extract target (last action's end coordinates)
        if include_target:
            last_action = episode_df.iloc[-1]
            target = np.array([
                last_action['end_x'] / self.field_x_max,
                last_action['end_y'] / self.field_y_max
            ], dtype=np.float32)
            result['target'] = target

        return result

    def save_vocab(self, path: str):
        """Save encoding vocabularies"""
        import json
        vocab = {
            'type_to_id': self.type_to_id,
            'result_to_id': self.result_to_id,
            'field_x_max': self.field_x_max,
            'field_y_max': self.field_y_max,
            'max_seq_len': self.max_seq_len
        }
        with open(path, 'w') as f:
            json.dump(vocab, f, indent=2)

    def load_vocab(self, path: str):
        """Load encoding vocabularies"""
        import json
        with open(path, 'r') as f:
            vocab = json.load(f)
        self.type_to_id = vocab['type_to_id']
        self.result_to_id = vocab['result_to_id']
        self.field_x_max = vocab['field_x_max']
        self.field_y_max = vocab['field_y_max']
        self.max_seq_len = vocab['max_seq_len']
        return self
