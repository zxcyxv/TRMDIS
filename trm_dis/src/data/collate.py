"""
Custom collate function for batching episodes
"""
import torch
from typing import List, Dict


def collate_episodes(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for episode batches

    Since all episodes are already padded to max_seq_len,
    we can simply stack them.

    Args:
        batch: List of episode dictionaries

    Returns:
        Batched dictionary:
            - continuous: [B, max_seq_len, 4]
            - categorical: [B, max_seq_len, 3]
            - mask: [B, max_seq_len]
            - target: [B, 2] (if present)
            - episode_ids: List[str]
    """
    # Stack tensors
    continuous = torch.stack([item['continuous'] for item in batch])
    categorical = torch.stack([item['categorical'] for item in batch])
    mask = torch.stack([item['mask'] for item in batch])
    episode_ids = [item['episode_id'] for item in batch]

    result = {
        'continuous': continuous,
        'categorical': categorical,
        'mask': mask,
        'episode_ids': episode_ids
    }

    # Include targets if present
    if 'target' in batch[0]:
        target = torch.stack([item['target'] for item in batch])
        result['target'] = target

    if 'cls_out' in batch[0]:
        cls_out = torch.stack([item['cls_out'] for item in batch])
        result['cls_out'] = cls_out

    if 'cond_seq' in batch[0]:
        cond_seq = torch.stack([item['cond_seq'] for item in batch])
        result['cond_seq'] = cond_seq

    return result
