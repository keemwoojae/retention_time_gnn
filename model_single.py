"""
Single-Head Model Trainer

Supports:
- Global normalization (entire dataset)
- Task-wise normalization (per system)
- Categorical features (system_id, chrom_type)
- Adam and L-BFGS optimizers
"""

import numpy as np
import time

import torch
import torch.nn as nn
from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SingleHeadTrainer:
    """
    Trainer for Single-Head model with Categorical Features.
    """

    def __init__(self, net, source_path, cuda, norm_stats):
        """
        Args:
            net: Single-head model (e.g., GINSinglePredictor)
            source_path: Path to pre-trained model (for fine-tuning)
            cuda: Device to use
            norm_stats: Dict with normalization statistics from dataset
                        {'mode', 'global_mean', 'global_std', 'task_means', 'task_stds'}
        """
        self.net = net.to(cuda)
        self.source_path = source_path
        self.target_path = ''
        self.cuda = cuda

        self.norm_mode = norm_stats.get('mode', 'global')
        self.global_mean = norm_stats.get('global_mean', 0.0)
        self.global_std = norm_stats.get('global_std', 1.0)
        self.task_means = norm_stats.get('task_means', {})
        self.task_stds = norm_stats.get('task_stds', {})

    def load(self, model_path):
        """Load model weights."""
        self.net.load_state_dict(torch.load(model_path, map_location=self.cuda))

    def load_pretrained_encoder(self):
        """Load pre-trained encoder from source model."""
        self.net.load_pretrained_encoder(self.source_path)

    def _normalize_labels(self, labels, system_ids=None):
        """
        Normalize labels based on normalization mode.

        Args:
            labels: (B, 1) tensor
            system_ids: (B,) tensor, required for task_wise normalization

        Returns:
            Normalized labels
        """
        labels_cpu = labels.cpu() if labels.is_cuda else labels

        if self.norm_mode == 'global':
            normalized = (labels_cpu - self.global_mean) / self.global_std
        else:  # task_wise
            system_ids_cpu = system_ids.cpu() if system_ids.is_cuda else system_ids
            normalized = labels_cpu.clone()
            unique_systems = torch.unique(system_ids_cpu)

            for sys_id in unique_systems:
                sys_id_val = sys_id.item()
                if sys_id_val in self.task_means and sys_id_val in self.task_stds:
                    mask = system_ids_cpu == sys_id
                    mean = self.task_means[sys_id_val]
                    std = self.task_stds[sys_id_val]
                    normalized[mask] = (labels_cpu[mask] - mean) / std

        return normalized

    def _denormalize_preds(self, preds, system_ids=None):
        """
        Denormalize predictions based on normalization mode.

        Args:
            preds: (B, 1) tensor or numpy array
            system_ids: (B,) tensor or numpy array, required for task_wise normalization

        Returns:
            Denormalized predictions
        """
        if isinstance(preds, torch.Tensor):
            denormalized = preds.clone()
        else:
            denormalized = preds.copy()

        if self.norm_mode == 'global':
            denormalized = preds * self.global_std + self.global_mean
        else:  # task_wise
            if isinstance(system_ids, torch.Tensor):
                unique_systems = torch.unique(system_ids).cpu().numpy()
            else:
                unique_systems = np.unique(system_ids)

            for sys_id in unique_systems:
                sys_id_val = int(sys_id)
                if sys_id_val in self.task_means and sys_id_val in self.task_stds:
                    mask = system_ids == sys_id
                    mean = self.task_means[sys_id_val]
                    std = self.task_stds[sys_id_val]
                    denormalized[mask] = preds[mask] * std + mean

        return denormalized

    def training(self, train_loader, val_loader, method='finetune', opt='adam',
                 encoder_lr=1e-5, head_lr=1e-4, max_epochs=500, verbose=True,
                 freeze_encoder=False, weight_decay=1e-5, patience=30):
        """
        Train the single-head model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (can be None for lbfgs)
            method: 'scratch', 'finetune', or 'feature'
            opt: 'adam' or 'lbfgs'
            encoder_lr: Learning rate for encoder (used in finetune mode)
            head_lr: Learning rate for head
            max_epochs: Maximum number of epochs
            verbose: Print training progress
            freeze_encoder: Whether to freeze encoder (overrides method setting)
            weight_decay: Weight decay for optimizer
            patience: Early stopping patience
        """
        # Setup based on method
        if method == 'scratch':
            pass
        elif method == 'finetune':
            self.load_pretrained_encoder()
            self.net.reset_head()
        elif method == 'feature':
            self.load_pretrained_encoder()
            self.net.reset_head()
            freeze_encoder = True

        if freeze_encoder:
            self.net.freeze_encoder()
            print('[SingleHeadTrainer] Encoder frozen')

        trainable_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f'[SingleHeadTrainer] Trainable parameters: {trainable_params}')
        print(f'[SingleHeadTrainer] Normalization mode: {self.norm_mode}')

        loss_fn = nn.HuberLoss()

        if opt == 'adam':
            self._train_adam(
                train_loader, val_loader, loss_fn,
                encoder_lr, head_lr, max_epochs, verbose,
                freeze_encoder, weight_decay, patience
            )
        elif opt == 'lbfgs':
            self._train_lbfgs(
                train_loader, loss_fn, max_epochs, verbose, weight_decay
            )

    def _train_adam(self, train_loader, val_loader, loss_fn,
                    encoder_lr, head_lr, max_epochs, verbose,
                    freeze_encoder, weight_decay, patience):
        """Train with Adam optimizer."""

        if freeze_encoder:
            optimizer = Adam(
                self.net.get_head_params(),
                lr=head_lr,
                weight_decay=weight_decay
            )
        else:
            optimizer = Adam([
                {'params': self.net.get_encoder_params(), 'lr': encoder_lr},
                {'params': self.net.get_head_params(), 'lr': head_lr}
            ], weight_decay=weight_decay)

        lr_scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=20,
            min_lr=1e-7, verbose=True
        )

        val_log = np.zeros(max_epochs)

        for epoch in range(max_epochs):
            # Training
            self.net.train()
            start_time = time.time()
            epoch_loss = 0.0
            n_batches = 0

            for batchidx, batchdata in enumerate(train_loader):
                g, node_feats, edge_feats, labels, system_ids, chrom_types, cat_feats = batchdata

                g = g.to(self.cuda)
                node_feats = node_feats.to(self.cuda)
                edge_feats = edge_feats.to(self.cuda)
                cat_feats = cat_feats.to(self.cuda)
                system_ids = system_ids.to(self.cuda)

                # Normalize labels
                labels_norm = self._normalize_labels(labels, system_ids).to(self.cuda)

                # Forward pass with categorical features
                preds = self.net(g, node_feats, edge_feats, cat_feats)

                # Compute loss
                loss = loss_fn(preds, labels_norm)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.detach().item()
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches

            if verbose:
                print(f'--- Epoch {epoch + 1}, lr encoder={optimizer.param_groups[0]["lr"]:.2e}, '
                      f'loss={avg_train_loss:.4f}, time={(time.time()-start_time)/60:.2f}min')

            # Validation
            if val_loader is not None:
                start_time = time.time()
                val_loss = self._validate(val_loader, loss_fn)
                lr_scheduler.step(val_loss)
                val_log[epoch] = val_loss

                if verbose:
                    best_epoch = np.argmin(val_log[:epoch + 1])
                    print(f'--- Validation loss={val_loss:.4f} (BEST={val_log[best_epoch]:.4f} @ epoch {best_epoch + 1}), '
                          f'time={(time.time()-start_time)/60:.2f}min')

                # Save best model
                if np.argmin(val_log[:epoch + 1]) == epoch:
                    torch.save(self.net.state_dict(), self.target_path)

                # Early stopping
                if np.argmin(val_log[:epoch + 1]) <= epoch - patience:
                    print(f'[SingleHeadTrainer] Early stopping at epoch {epoch + 1}')
                    break
            else:
                torch.save(self.net.state_dict(), self.target_path)

        print(f'[SingleHeadTrainer] Training terminated at epoch {epoch + 1}')

        if val_loader is not None:
            self.load(self.target_path)

    def _train_lbfgs(self, train_loader, loss_fn, max_epochs, verbose, weight_decay):
        """Train with L-BFGS optimizer."""

        # Load all data
        all_data = None
        for batchdata in train_loader:
            all_data = batchdata
            break

        g, node_feats, edge_feats, labels, system_ids, chrom_types, cat_feats = all_data
        g = g.to(self.cuda)
        node_feats = node_feats.to(self.cuda)
        edge_feats = edge_feats.to(self.cuda)
        cat_feats = cat_feats.to(self.cuda)
        system_ids = system_ids.to(self.cuda)
        labels_norm = self._normalize_labels(labels, system_ids).to(self.cuda)

        def loss_calc():
            preds = self.net(g, node_feats, edge_feats, cat_feats)
            loss = loss_fn(preds, labels_norm)
            loss += weight_decay * torch.stack([p.square().sum() for p in self.net.parameters() if p.requires_grad]).sum()
            return loss

        def closure():
            optimizer.zero_grad()
            loss = loss_calc()
            loss.backward()
            return loss

        optimizer = LBFGS(self.net.parameters(), lr=1, max_iter=1)

        loss_log = np.zeros(max_epochs)
        for epoch in range(max_epochs):
            self.net.train()
            optimizer.step(closure)

            loss_log[epoch] = loss_calc().detach().cpu().numpy()
            if np.isnan(loss_log[epoch]):
                loss_log[epoch] = 1e5

            optimizer.param_groups[0]['lr'] -= 1 / max_epochs

            if np.argmin(loss_log[:epoch + 1]) == epoch:
                torch.save(self.net.state_dict(), self.target_path)

            if verbose and (epoch + 1) % 50 == 0:
                print(f'--- L-BFGS Iter {epoch + 1}, loss={loss_log[epoch]:.4f}')

        best_iter = np.argmin(loss_log) + 1
        print(f'[SingleHeadTrainer] L-BFGS training terminated, best iter={best_iter}')
        self.load(self.target_path)

    def _validate(self, val_loader, loss_fn):
        """Compute validation loss."""
        self.net.eval()
        total_loss = 0.0
        n_samples = 0

        with torch.no_grad():
            for batchdata in val_loader:
                g, node_feats, edge_feats, labels, system_ids, chrom_types, cat_feats = batchdata

                g = g.to(self.cuda)
                node_feats = node_feats.to(self.cuda)
                edge_feats = edge_feats.to(self.cuda)
                cat_feats = cat_feats.to(self.cuda)
                system_ids = system_ids.to(self.cuda)

                labels_norm = self._normalize_labels(labels, system_ids).to(self.cuda)

                preds = self.net(g, node_feats, edge_feats, cat_feats)
                loss = loss_fn(preds, labels_norm)

                total_loss += loss.item() * labels.size(0)
                n_samples += labels.size(0)

        return total_loss / n_samples

    def inference(self, data_loader):
        """
        Run inference on data.

        Args:
            data_loader: Data loader

        Returns:
            Predictions (denormalized)
        """
        self.net.eval()
        all_preds = []
        all_system_ids = []

        with torch.no_grad():
            for batchdata in data_loader:
                g, node_feats, edge_feats, labels, system_ids, chrom_types, cat_feats = batchdata

                g = g.to(self.cuda)
                node_feats = node_feats.to(self.cuda)
                edge_feats = edge_feats.to(self.cuda)
                cat_feats = cat_feats.to(self.cuda)

                preds = self.net(g, node_feats, edge_feats, cat_feats)
                all_preds.append(preds.cpu().numpy())
                all_system_ids.append(system_ids.numpy())

        all_preds = np.vstack(all_preds)
        all_system_ids = np.concatenate(all_system_ids)

        # Denormalize predictions
        all_preds = self._denormalize_preds(all_preds, all_system_ids)

        return all_preds


def compute_system_metrics(y_true, y_pred, system_ids, system_names):
    """
    Compute metrics for each system.

    Args:
        y_true: Ground truth labels (N, 1)
        y_pred: Predictions (N, 1)
        system_ids: System IDs (N,)
        system_names: List of system names

    Returns:
        Dict of metrics per system
    """
    from sklearn.metrics import mean_absolute_error, r2_score

    metrics = {}
    unique_systems = np.unique(system_ids)

    for sys_id in unique_systems:
        mask = system_ids == sys_id
        y_t = y_true[mask].flatten()
        y_p = y_pred[mask].flatten()

        mae = mean_absolute_error(y_t, y_p)
        medae = np.median(np.abs(y_t - y_p))
        r2 = r2_score(y_t, y_p)

        sys_name = system_names[sys_id] if sys_id < len(system_names) else f'System_{sys_id}'
        metrics[sys_name] = {
            'MAE': mae,
            'MedAE': medae,
            'R2': r2,
            'n_samples': int(mask.sum())
        }

    return metrics
