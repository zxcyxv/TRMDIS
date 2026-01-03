"""
Trainer for TRM+DIS model
Handles training loop, validation, and checkpointing

DIS-Specific Monitoring:
1. Monotonic Improvement: Distance to target decreases with each step s
2. Step-wise Loss Distribution: Loss at each supervision step
3. Cosine Similarity Convergence: Direction alignment improves with s
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import os
from typing import Optional, Dict, List
import json

from ..models.ema import EMA
from .loss import CombinedLoss, CTRILoss
from .optimizers import AdamAtan


class Trainer:
    """
    Training manager for TRM+DIS model
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = AdamAtan(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler (warmup + cosine decay)
        self.scheduler = self._create_scheduler()

        # Loss function selection
        self.use_ctri_loss = getattr(config, 'use_ctri_loss', False)
        if self.use_ctri_loss:
            self.criterion = CTRILoss(
                lambda_mono=config.ctri_lambda_mono,
                lambda_trust=config.ctri_lambda_trust,
                gamma=getattr(config, 'ctri_gamma', 0.95),
                x_weight=config.x_weight,
                y_weight=config.y_weight
            )
            print(f"Using CTRILoss (lambda_mono={config.ctri_lambda_mono}, "
                  f"lambda_trust={config.ctri_lambda_trust}, "
                  f"gamma={getattr(config, 'ctri_gamma', 0.95)})")
        else:
            self.criterion = CombinedLoss(
                huber_delta=config.huber_delta,
                adv_margin=config.adv_margin,
                adv_weight=config.adv_weight,
                work_weight=config.work_weight,
                work_eta=config.work_eta,
                work_eps=config.work_eps,
                work_kappa=config.work_kappa,
                x_weight=config.x_weight,
                y_weight=config.y_weight
            )
            print("Using AL-GPI CombinedLoss")

        # EMA
        self.ema = EMA(model, decay=config.ema_decay) if hasattr(config, 'ema_decay') else None

        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        # DIS metrics history
        self.dis_metrics_history: List[Dict] = []

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                # Warmup
                return step / self.config.warmup_steps
            else:
                # Cosine decay
                progress = (step - self.config.warmup_steps) / \
                          (len(self.train_loader) * self.config.max_epochs - self.config.warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

        return LambdaLR(self.optimizer, lr_lambda)

    @torch.no_grad()
    def compute_dis_metrics(
        self,
        predictions: torch.Tensor,  # [B, n_sup, 2]
        targets: torch.Tensor       # [B, 2]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute DIS-specific monitoring metrics

        Args:
            predictions: [B, n_sup, 2] - Predictions at each supervision step
            targets: [B, 2] - Ground truth coordinates

        Returns:
            Dictionary with DIS metrics:
                - step_distances: [n_sup] - Euclidean distance at each step
                - monotonic_violations: int - Number of steps where distance increased
                - step_losses: [n_sup] - Huber loss at each step
                - step_cosine_sims: [n_sup-1] - Cosine similarity between consecutive movements
                - improvement_ratios: [n_sup-1] - Distance reduction ratio between steps
        """
        B, n_sup, _ = predictions.shape

        # Expand targets for comparison: [B, n_sup, 2]
        targets_expanded = targets.unsqueeze(1).expand(-1, n_sup, -1)

        # ================================================================
        # 1. MONOTONIC IMPROVEMENT (Lyapunov Contraction)
        # Distance to target should decrease with each step s
        # ================================================================
        # Euclidean distance at each step: [B, n_sup]
        distances = torch.norm(predictions - targets_expanded, dim=-1)

        # Mean distance per step: [n_sup]
        step_distances = distances.mean(dim=0)

        # Count monotonic violations: distance[s+1] > distance[s]
        distance_diffs = distances[:, 1:] - distances[:, :-1]  # [B, n_sup-1]
        monotonic_violations = (distance_diffs > 0).sum().item()
        total_transitions = B * (n_sup - 1)
        monotonic_rate = 1.0 - (monotonic_violations / total_transitions)

        # Improvement ratios: how much closer we get at each step
        # ratio = (d[s] - d[s+1]) / d[s]
        improvement_ratios = -distance_diffs / (distances[:, :-1] + 1e-8)  # [B, n_sup-1]
        mean_improvement_ratios = improvement_ratios.mean(dim=0)  # [n_sup-1]

        # ================================================================
        # 2. STEP-WISE LOSS DISTRIBUTION
        # Track loss at each supervision step
        # ================================================================
        step_losses = []
        huber_fn = nn.HuberLoss(delta=self.config.huber_delta, reduction='mean')
        for s in range(n_sup):
            loss_s = huber_fn(predictions[:, s, :], targets)
            step_losses.append(loss_s)
        step_losses = torch.stack(step_losses)  # [n_sup]

        # ================================================================
        # 3. COSINE SIMILARITY CONVERGENCE (Advantage Margin)
        # Movement direction should align with target direction
        # ================================================================
        # Movement vectors: y_{s+1} - y_s
        movements = predictions[:, 1:, :] - predictions[:, :-1, :]  # [B, n_sup-1, 2]

        # Target direction from each step: y* - y_s
        target_dirs = targets_expanded[:, :-1, :] - predictions[:, :-1, :]  # [B, n_sup-1, 2]

        # Normalize and compute cosine similarity
        movements_norm = F.normalize(movements + 1e-8, dim=-1)
        target_dirs_norm = F.normalize(target_dirs + 1e-8, dim=-1)
        cosine_sims = F.cosine_similarity(movements_norm, target_dirs_norm, dim=-1)  # [B, n_sup-1]

        # Mean cosine similarity per step: [n_sup-1]
        step_cosine_sims = cosine_sims.mean(dim=0)

        return {
            'step_distances': step_distances,           # [n_sup] - should decrease
            'monotonic_rate': monotonic_rate,           # scalar - should be close to 1.0
            'monotonic_violations': monotonic_violations,
            'step_losses': step_losses,                 # [n_sup] - distribution check
            'step_cosine_sims': step_cosine_sims,       # [n_sup-1] - should approach 1.0
            'improvement_ratios': mean_improvement_ratios  # [n_sup-1] - should be positive
        }

    def format_dis_metrics(self, metrics: Dict) -> str:
        """Format DIS metrics for display"""
        lines = []
        lines.append("  DIS Metrics:")

        # Step distances
        dists = metrics['step_distances']
        dist_str = " → ".join([f"{d:.3f}" for d in dists.tolist()])
        lines.append(f"    Step Distances: [{dist_str}]")

        # Monotonic rate
        lines.append(f"    Monotonic Rate: {metrics['monotonic_rate']:.1%} "
                    f"({metrics['monotonic_violations']} violations)")

        # Step losses
        losses = metrics['step_losses']
        loss_str = " → ".join([f"{l:.4f}" for l in losses.tolist()])
        lines.append(f"    Step Losses: [{loss_str}]")

        # Cosine similarities
        cosines = metrics['step_cosine_sims']
        cos_str = " → ".join([f"{c:.3f}" for c in cosines.tolist()])
        lines.append(f"    Cosine Sims (s→s+1): [{cos_str}]")

        # Improvement ratios
        ratios = metrics['improvement_ratios']
        ratio_str = " → ".join([f"{r:.1%}" for r in ratios.tolist()])
        lines.append(f"    Improvement Ratios: [{ratio_str}]")

        return "\n".join(lines)

    def train_epoch(self) -> dict:
        """
        Train for one epoch

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_coord_loss = 0.0
        total_dir_loss = 0.0
        total_work_loss = 0.0
        total_mono_loss = 0.0
        total_trust_loss = 0.0
        total_mono_violation_rate = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch in pbar:
            # Move to device
            continuous = batch['continuous'].to(self.device)
            categorical = batch['categorical'].to(self.device)
            mask = batch['mask'].to(self.device)
            targets = batch['target'].to(self.device)
            cls_out = batch.get('cls_out')
            if cls_out is not None:
                cls_out = cls_out.to(self.device)
            cond_vec = batch.get('cond_vec')
            if cond_vec is not None:
                cond_vec = cond_vec.to(self.device)

            # Forward
            outputs = self.model(
                continuous, categorical, mask, targets, cls_out=cls_out, cond_vec=cond_vec
            )

            # Loss
            loss_dict = self.criterion(outputs['predictions'], outputs['diff_targets'])

            # Backward
            self.optimizer.zero_grad()
            loss_dict['total'].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip_norm
            )

            self.optimizer.step()
            self.scheduler.step()

            # Update EMA
            if self.ema is not None:
                self.ema.update()

            # Logging
            total_loss += loss_dict['total'].item()
            total_coord_loss += loss_dict['coord'].item()
            total_dir_loss += loss_dict['direction'].item()
            total_work_loss += loss_dict['work'].item()
            num_batches += 1
            self.global_step += 1

            # CTRI-specific metrics
            if self.use_ctri_loss:
                total_mono_loss += loss_dict.get('mono', torch.tensor(0.0)).item()
                total_trust_loss += loss_dict.get('trust', torch.tensor(0.0)).item()
                total_mono_violation_rate += loss_dict.get('mono_violation_rate', torch.tensor(0.0)).item()
                pbar.set_postfix({
                    'loss': f"{loss_dict['total'].item():.4f}",
                    'coord': f"{loss_dict['coord'].item():.4f}",
                    'mono': f"{loss_dict['mono'].item():.4f}",
                    'trust': f"{loss_dict['trust'].item():.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
            else:
                pbar.set_postfix({
                    'loss': f"{loss_dict['total'].item():.4f}",
                    'coord': f"{loss_dict['coord'].item():.4f}",
                    'dir': f"{loss_dict['direction'].item():.4f}",
                    'work': f"{loss_dict['work'].item():.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                })

        result = {
            'loss': total_loss / num_batches,
            'coord_loss': total_coord_loss / num_batches,
            'dir_loss': total_dir_loss / num_batches,
            'work_loss': total_work_loss / num_batches
        }

        # Add CTRI-specific metrics
        if self.use_ctri_loss:
            result['mono_loss'] = total_mono_loss / num_batches
            result['trust_loss'] = total_trust_loss / num_batches
            result['mono_violation_rate'] = total_mono_violation_rate / num_batches

        return result

    @torch.no_grad()
    def validate(self, use_ema: bool = True) -> dict:
        """
        Validate on validation set with DIS-specific metrics

        Args:
            use_ema: Whether to use EMA parameters

        Returns:
            Dictionary with validation metrics including DIS monitoring
        """
        if use_ema and self.ema is not None:
            ctx = self.ema.average_parameters()
            ctx.__enter__()

        self.model.eval()

        total_loss = 0.0
        total_coord_loss = 0.0
        total_dir_loss = 0.0
        total_work_loss = 0.0
        total_mono_loss = 0.0
        total_trust_loss = 0.0
        total_mono_violation_rate = 0.0
        total_mae_x = 0.0
        total_mae_y = 0.0
        total_euclidean = 0.0
        num_batches = 0

        # DIS metrics accumulators
        n_sup = self.model.n_sup
        all_step_distances = torch.zeros(n_sup, device=self.device)
        all_step_losses = torch.zeros(n_sup, device=self.device)
        all_step_cosines = torch.zeros(n_sup - 1, device=self.device)
        all_improvement_ratios = torch.zeros(n_sup - 1, device=self.device)
        total_monotonic_violations = 0
        total_transitions = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            continuous = batch['continuous'].to(self.device)
            categorical = batch['categorical'].to(self.device)
            mask = batch['mask'].to(self.device)
            targets = batch['target'].to(self.device)
            cls_out = batch.get('cls_out')
            if cls_out is not None:
                cls_out = cls_out.to(self.device)
            cond_vec = batch.get('cond_vec')
            if cond_vec is not None:
                cond_vec = cond_vec.to(self.device)

            # Forward (note: in eval mode, diff_targets won't be generated)
            outputs = self.model(
                continuous, categorical, mask, targets=None, cls_out=cls_out, cond_vec=cond_vec
            )

            # Generate diffusion targets manually for validation loss
            # (model doesn't generate them in eval mode)
            coords_init = continuous[:, -1, :2]  # Match training init (last action start_xy)
            diff_targets = self.model.diffusion.generate_targets(
                coords_init,
                targets
            )

            # Loss
            loss_dict = self.criterion(outputs['predictions'], diff_targets)

            total_loss += loss_dict['total'].item()
            total_coord_loss += loss_dict['coord'].item()
            total_dir_loss += loss_dict['direction'].item()
            total_work_loss += loss_dict['work'].item()

            # CTRI-specific metrics
            if self.use_ctri_loss:
                total_mono_loss += loss_dict.get('mono', torch.tensor(0.0)).item()
                total_trust_loss += loss_dict.get('trust', torch.tensor(0.0)).item()
                total_mono_violation_rate += loss_dict.get('mono_violation_rate', torch.tensor(0.0)).item()

            # Compute DIS-specific metrics
            dis_metrics = self.compute_dis_metrics(outputs['predictions'], targets)
            all_step_distances += dis_metrics['step_distances']
            all_step_losses += dis_metrics['step_losses']
            all_step_cosines += dis_metrics['step_cosine_sims']
            all_improvement_ratios += dis_metrics['improvement_ratios']
            total_monotonic_violations += dis_metrics['monotonic_violations']
            total_transitions += batch['target'].size(0) * (n_sup - 1)

            # Compute metrics on final prediction (denormalized)
            final_pred = outputs['final_pred']  # [B, 2]
            # Denormalize
            final_pred_denorm = final_pred * torch.tensor(
                [self.config.field_x_max, self.config.field_y_max],
                device=self.device
            )
            targets_denorm = targets * torch.tensor(
                [self.config.field_x_max, self.config.field_y_max],
                device=self.device
            )

            # Log step-wise predictions for the first sample in the first batch
            if num_batches == 0:
                idx = torch.randint(0, outputs['predictions'].size(0), (1,)).item()
                preds_norm = outputs['predictions'][idx]  # [n_sup, 2]
                preds_denorm = preds_norm * torch.tensor(
                    [self.config.field_x_max, self.config.field_y_max],
                    device=self.device
                )
                target_norm = targets[idx]
                target_denorm = targets_denorm[idx]
                pred_list = ", ".join(
                    [f"s{i+1}:({p[0]:.3f},{p[1]:.3f})" for i, p in enumerate(preds_norm)]
                )
                pred_list_denorm = ", ".join(
                    [f"s{i+1}:({p[0]:.2f},{p[1]:.2f})" for i, p in enumerate(preds_denorm)]
                )
                print(f"  Sample{idx} Pred (norm):", pred_list)
                print(f"  Sample{idx} Target (norm): ({target_norm[0]:.3f},{target_norm[1]:.3f})")
                print(f"  Sample{idx} Pred (m):", pred_list_denorm)
                print(f"  Sample{idx} Target (m): ({target_denorm[0]:.2f},{target_denorm[1]:.2f})")

            # MAE per dimension
            mae_x = torch.abs(final_pred_denorm[:, 0] - targets_denorm[:, 0]).mean()
            mae_y = torch.abs(final_pred_denorm[:, 1] - targets_denorm[:, 1]).mean()

            # Euclidean distance
            euclidean = torch.norm(final_pred_denorm - targets_denorm, dim=-1).mean()

            total_mae_x += mae_x.item()
            total_mae_y += mae_y.item()
            total_euclidean += euclidean.item()
            num_batches += 1

        if use_ema and self.ema is not None:
            ctx.__exit__(None, None, None)

        # Aggregate DIS metrics
        dis_metrics_avg = {
            'step_distances': all_step_distances / num_batches,
            'step_losses': all_step_losses / num_batches,
            'step_cosine_sims': all_step_cosines / num_batches,
            'improvement_ratios': all_improvement_ratios / num_batches,
            'monotonic_rate': 1.0 - (total_monotonic_violations / total_transitions),
            'monotonic_violations': total_monotonic_violations
        }

        result = {
            'loss': total_loss / num_batches,
            'coord_loss': total_coord_loss / num_batches,
            'dir_loss': total_dir_loss / num_batches,
            'work_loss': total_work_loss / num_batches,
            'mae_x': total_mae_x / num_batches,
            'mae_y': total_mae_y / num_batches,
            'mae_avg': (total_mae_x + total_mae_y) / (2 * num_batches),
            'euclidean': total_euclidean / num_batches,
            'dis_metrics': dis_metrics_avg
        }

        # Add CTRI-specific metrics
        if self.use_ctri_loss:
            result['mono_loss'] = total_mono_loss / num_batches
            result['trust_loss'] = total_trust_loss / num_batches
            result['mono_violation_rate'] = total_mono_violation_rate / num_batches

        return result

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()

        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, path)

        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best.pt')
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        path = os.path.join(self.config.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        if self.ema is not None and 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])

    def train(self, num_epochs: Optional[int] = None):
        """
        Full training loop with DIS-specific monitoring

        Args:
            num_epochs: Number of epochs (uses config if None)
        """
        if num_epochs is None:
            num_epochs = self.config.max_epochs

        print(f"Starting training for {num_epochs} epochs")
        print(f"Model parameters: {self.model.get_num_params():,}")
        print(f"DIS Configuration: n_sup={self.model.n_sup}, n={self.model.n}")
        print("=" * 60)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate(use_ema=True)

            # Log standard metrics
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")

            # Log loss-specific metrics
            if self.use_ctri_loss:
                print(f"  Train Mono Loss: {train_metrics.get('mono_loss', 0):.4f}")
                print(f"  Train Trust Loss: {train_metrics.get('trust_loss', 0):.4f}")
                print(f"  Val Mono Loss: {val_metrics.get('mono_loss', 0):.4f}")
                print(f"  Val Trust Loss: {val_metrics.get('trust_loss', 0):.4f}")
                print(f"  Mono Violation Rate: {val_metrics.get('mono_violation_rate', 0):.1%}")
            else:
                print(f"  Train Work Loss: {train_metrics['work_loss']:.4f}")
                print(f"  Val Work Loss: {val_metrics['work_loss']:.4f}")

            print(f"  Val MAE (avg): {val_metrics['mae_avg']:.2f}m")
            print(f"  Val Euclidean: {val_metrics['euclidean']:.2f}m")

            # Log DIS-specific metrics
            if 'dis_metrics' in val_metrics:
                print(self.format_dis_metrics(val_metrics['dis_metrics']))

                # Store for history
                self.dis_metrics_history.append({
                    'epoch': epoch,
                    'step_distances': val_metrics['dis_metrics']['step_distances'].tolist(),
                    'step_losses': val_metrics['dis_metrics']['step_losses'].tolist(),
                    'step_cosine_sims': val_metrics['dis_metrics']['step_cosine_sims'].tolist(),
                    'monotonic_rate': val_metrics['dis_metrics']['monotonic_rate']
                })

                # Warning if monotonic improvement is failing
                if val_metrics['dis_metrics']['monotonic_rate'] < 0.8:
                    print("  ⚠️  WARNING: Monotonic improvement rate below 80%!")
                    print("      Consider: adjusting learning rate, increasing n_sup, or checking loss weights")

            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                print(f"  ✓ New best val loss: {self.best_val_loss:.4f}")

            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f'epoch_{epoch}.pt', is_best=is_best)

                # Save DIS metrics history
                metrics_path = os.path.join(self.config.log_dir, 'dis_metrics.json')
                with open(metrics_path, 'w') as f:
                    json.dump(self.dis_metrics_history, f, indent=2)

        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Best val loss: {self.best_val_loss:.4f}")

        # Final DIS analysis
        if self.dis_metrics_history:
            final_metrics = self.dis_metrics_history[-1]
            print("\nFinal DIS Analysis:")
            print(f"  Monotonic Rate: {final_metrics['monotonic_rate']:.1%}")
            print(f"  Distance Trend: {final_metrics['step_distances'][0]:.3f} → {final_metrics['step_distances'][-1]:.3f}")
            print(f"  Final Cosine Sim: {final_metrics['step_cosine_sims'][-1]:.3f}")
