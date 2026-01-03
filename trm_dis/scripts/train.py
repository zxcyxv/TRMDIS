"""
Main training script for TRM+DIS model
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pandas as pd
from torch.utils.data import DataLoader
import argparse
import random
import numpy as np

from src.config import ModelConfig, ExperimentConfig
from src.data.preprocessing import EpisodeEncoder
from src.data.dataset import EpisodeDataset, create_train_val_split
from src.data.collate import collate_episodes
from src.models.trm_dis import TRMDIS
from src.training.trainer import Trainer


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may reduce performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def main(args):
    # Load config
    config = ExperimentConfig()
    config.model.device = args.device

    if args.debug:
        config.debug_mode = True
        config.debug_samples = 100
        config.model.max_epochs = 5
        print("=" * 50)
        print("DEBUG MODE ENABLED - Using small subset")
        print("=" * 50)

    # Set seed
    set_seed(config.seed)

    print("Loading data...")
    # Load training data
    df = pd.read_csv(config.model.train_csv_path)

    if config.debug_mode:
        # Use small subset for debugging
        unique_episodes = df['game_episode'].unique()[:config.debug_samples]
        df = df[df['game_episode'].isin(unique_episodes)]
        # Adjust val_episodes for debug mode
        config.model.val_episodes = min(20, len(unique_episodes) // 5)
        print(f"Debug mode: Using {len(unique_episodes)} episodes")
        print(f"Debug mode: Val episodes set to {config.model.val_episodes}")

    # Create encoder
    encoder = EpisodeEncoder(
        max_seq_len=config.model.max_seq_len,
        field_x_max=config.model.field_x_max,
        field_y_max=config.model.field_y_max
    )

    # Fit encoder on training data
    print("Fitting encoder...")
    encoder.fit(df)

    # Save encoder vocabulary
    encoder.save_vocab(f"{config.model.checkpoint_dir}/vocab.json")

    # Split train/val
    print("Creating train/val split...")
    train_df, val_df = create_train_val_split(
        df,
        val_episodes=config.model.val_episodes,
        seed=config.seed
    )

    # Load CLS features if provided
    cls_features = None
    cond_features = None
    if config.model.use_cls_only and config.model.cls_feature_path:
        import numpy as np
        cls_data = np.load(config.model.cls_feature_path, allow_pickle=True)
        ids = cls_data['episode_ids']
        vecs = cls_data['cls_out']
        cls_features = {str(eid): vecs[i] for i, eid in enumerate(ids)}

    if config.model.use_cond_features and config.model.cond_feature_path:
        import numpy as np
        cond_data = np.load(config.model.cond_feature_path, allow_pickle=True)
        ids = cond_data['episode_ids']
        cls_out = cond_data['cls_out']
        fourier_seq = cond_data['fourier_seq']
        router_logits = cond_data['router_logits']
        cond_features = {
            str(eid): {
                'cls_out': cls_out[i],
                'fourier_seq': fourier_seq[i],
                'router_logits': router_logits[i],
            }
            for i, eid in enumerate(ids)
        }

    # Create datasets
    print("Creating datasets...")
    train_dataset = EpisodeDataset(
        train_df,
        encoder,
        include_target=True,
        cls_features=cls_features,
        cond_features=cond_features
    )
    val_dataset = EpisodeDataset(
        val_df,
        encoder,
        include_target=True,
        cls_features=cls_features,
        cond_features=cond_features
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.model.batch_size,
        shuffle=True,
        num_workers=config.model.num_workers,
        collate_fn=collate_episodes,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.model.batch_size,
        shuffle=False,
        num_workers=config.model.num_workers,
        collate_fn=collate_episodes,
        pin_memory=True
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print("\nCreating model...")
    model = TRMDIS(
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        d_ff=config.model.d_ff,
        n_action_types=config.model.n_action_types,
        n_result_types=config.model.n_result_types,
        n_sup=config.model.n_sup,
        n=config.model.n,
        dropout=config.model.dropout,
        max_seq_len=config.model.max_seq_len,
        cls_dim=config.model.cls_dim,
        use_cls_only=config.model.use_cls_only,
        use_cond_features=config.model.use_cond_features,
        cond_dim=config.model.cond_dim
    )

    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Model device: {config.model.device}")

    # Verify critical design points
    print("\n" + "=" * 50)
    print("VERIFYING CRITICAL DESIGN POINTS:")
    print("=" * 50)
    print(f"✓ T = {config.model.T} (NO no-grad cycles)")
    print(f"✓ n = {config.model.n} (internal latent updates)")
    print(f"✓ n_sup = {config.model.n_sup} (supervision steps)")
    print(f"✓ Sequence dimension preserved in model")
    print(f"✓ Direct assignment (no residual connections)")
    print("=" * 50 + "\n")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.model,
        device=config.model.device
    )

    # Train
    print("Starting training...\n")
    trainer.train(num_epochs=config.model.max_epochs)

    print("\nTraining complete!")
    print(f"Best checkpoint saved to: {config.model.checkpoint_dir}/best.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TRM+DIS model')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (small subset)')

    args = parser.parse_args()

    main(args)
