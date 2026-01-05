"""
Multi-Task Learning Trainer

Supports:
- Masked Loss: Only compute loss for samples with ground truth labels
- Task-wise normalization
- Encoder freezing / separate learning rates
- Adam and L-BFGS optimizers
"""

import numpy as np
import time

import torch
import torch.nn as nn
from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import ReduceLROnPlateau


class MTLTrainer:
    """
    Trainer for Multi-Task Learning with Hard Parameter Sharing.
    """

    def __init__(self, net, source_path, cuda, task_means=None, task_stds=None):
        """
        Args:
            net: MTL model (e.g., GINMTLPredictor)
            source_path: Path to pre-trained model (for fine-tuning)
            cuda: Device to use
            task_means: Dict of task-wise means for normalization
            task_stds: Dict of task-wise stds for normalization
        """
        self.net = net.to(cuda)
        self.source_path = source_path
        self.target_path = ''
        self.cuda = cuda

        self.task_means = task_means if task_means else {}
        self.task_stds = task_stds if task_stds else {}

    def load(self, model_path):
        """Load model weights."""
        self.net.load_state_dict(torch.load(model_path, map_location=self.cuda))

    def load_pretrained_encoder(self):
        """Load pre-trained encoder from source model."""
        self.net.load_pretrained_encoder(self.source_path)

    def _normalize_labels(self, labels, task_ids):
        """
        Normalize labels using task-specific mean/std.

        Args:
            labels: (B, 1) tensor
            task_ids: (B,) tensor

        Returns:
            Normalized labels (on CPU)
        """
        # Ensure all tensors are on CPU for indexing
        labels_cpu = labels.cpu() if labels.is_cuda else labels
        task_ids_cpu = task_ids.cpu() if task_ids.is_cuda else task_ids

        normalized = labels_cpu.clone()
        unique_tasks = torch.unique(task_ids_cpu)

        for task_id in unique_tasks:
            task_id_val = task_id.item()
            if task_id_val in self.task_means and task_id_val in self.task_stds:
                mask = task_ids_cpu == task_id
                mean = self.task_means[task_id_val]
                std = self.task_stds[task_id_val]
                normalized[mask] = (labels_cpu[mask] - mean) / std

        return normalized

    def _denormalize_preds(self, preds, task_ids):
        """
        Denormalize predictions using task-specific mean/std.

        Args:
            preds: (B, 1) tensor or numpy array
            task_ids: (B,) tensor or numpy array

        Returns:
            Denormalized predictions
        """
        if isinstance(preds, torch.Tensor):
            denormalized = preds.clone()
        else:
            denormalized = preds.copy()

        if isinstance(task_ids, torch.Tensor):
            unique_tasks = torch.unique(task_ids).cpu().numpy()
        else:
            unique_tasks = np.unique(task_ids)

        for task_id in unique_tasks:
            task_id_val = int(task_id)
            if task_id_val in self.task_means and task_id_val in self.task_stds:
                mask = task_ids == task_id
                mean = self.task_means[task_id_val]
                std = self.task_stds[task_id_val]
                denormalized[mask] = preds[mask] * std + mean

        return denormalized

    def masked_loss(self, preds, labels, task_ids, loss_fn, reduction='mean'):
        """
        Compute masked loss where each sample only contributes to its own task's loss.

        For MTL, when model outputs predictions for all heads (B, n_tasks),
        we only compute loss for the head corresponding to each sample's task.

        Args:
            preds: (B, n_tasks) or (B, 1) predictions
            labels: (B, 1) ground truth labels
            task_ids: (B,) task IDs
            loss_fn: Loss function
            reduction: 'mean' or 'sum'

        Returns:
            Scalar loss
        """
        batch_size = labels.size(0)

        if preds.size(1) == self.net.n_tasks:
            # Model returned all head outputs, select appropriate ones
            selected_preds = torch.zeros(batch_size, 1, device=preds.device)
            for i in range(batch_size):
                task_id = task_ids[i].item()
                selected_preds[i, 0] = preds[i, task_id]
        else:
            # Model already selected appropriate outputs
            selected_preds = preds

        loss = loss_fn(selected_preds, labels)

        return loss

    def training(self, train_loader, val_loader, method='finetune', opt='adam',
                 encoder_lr=1e-5, head_lr=1e-4, max_epochs=500, verbose=True,
                 freeze_encoder=False, weight_decay=1e-5, patience=30):
        """
        Train the MTL model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (can be None for lbfgs)
            method: 'scratch', 'finetune', or 'feature'
                - scratch: Train from random initialization
                - finetune: Load pre-trained encoder, train all with different LRs
                - feature: Load pre-trained encoder, freeze encoder, train heads only
            opt: 'adam' or 'lbfgs'
            encoder_lr: Learning rate for encoder (used in finetune mode)
            head_lr: Learning rate for heads
            max_epochs: Maximum number of epochs
            verbose: Print training progress
            freeze_encoder: Whether to freeze encoder (overrides method setting)
            weight_decay: Weight decay for optimizer
            patience: Early stopping patience
        """
        # Setup based on method
        if method == 'scratch':
            pass  # Start from random initialization
        elif method == 'finetune':
            self.load_pretrained_encoder()
            self.net.reset_heads()
        elif method == 'feature':
            self.load_pretrained_encoder()
            self.net.reset_heads()
            freeze_encoder = True

        if freeze_encoder:
            self.net.freeze_encoder()
            print('[MTLTrainer] Encoder frozen')

        trainable_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f'[MTLTrainer] Trainable parameters: {trainable_params}')

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

        # Setup optimizer with different learning rates
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

        # Get dataset sizes
        if hasattr(train_loader.dataset, 'label'):
            train_size = len(train_loader.dataset)
        else:
            train_size = len(train_loader.dataset.indices)

        val_log = np.zeros(max_epochs)

        for epoch in range(max_epochs):
            # Training
            self.net.train()
            start_time = time.time()
            epoch_loss = 0.0
            n_batches = 0

            for batchidx, batchdata in enumerate(train_loader):
                g, node_feats, edge_feats, labels, task_ids = batchdata

                g = g.to(self.cuda)
                node_feats = node_feats.to(self.cuda)
                edge_feats = edge_feats.to(self.cuda)
                task_ids = task_ids.to(self.cuda)

                # Normalize labels per task
                labels_norm = self._normalize_labels(labels, task_ids).to(self.cuda)

                # Forward pass with task_ids for selective head prediction
                preds = self.net(g, node_feats, edge_feats, task_ids)

                # Compute masked loss
                loss = self.masked_loss(preds, labels_norm, task_ids, loss_fn)

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
                    print(f'[MTLTrainer] Early stopping at epoch {epoch + 1}')
                    break
            else:
                # No validation, save every epoch
                torch.save(self.net.state_dict(), self.target_path)

        print(f'[MTLTrainer] Training terminated at epoch {epoch + 1}')

        # Load best model
        if val_loader is not None:
            self.load(self.target_path)

    def _train_lbfgs(self, train_loader, loss_fn, max_epochs, verbose, weight_decay):
        """Train with L-BFGS optimizer."""

        # Load all data
        all_data = None
        for batchdata in train_loader:
            all_data = batchdata
            break

        g, node_feats, edge_feats, labels, task_ids = all_data
        g = g.to(self.cuda)
        node_feats = node_feats.to(self.cuda)
        edge_feats = edge_feats.to(self.cuda)
        task_ids = task_ids.to(self.cuda)
        labels_norm = self._normalize_labels(labels, task_ids).to(self.cuda)

        def loss_calc():
            preds = self.net(g, node_feats, edge_feats, task_ids)
            loss = self.masked_loss(preds, labels_norm, task_ids, loss_fn)
            # L2 regularization
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

            # Learning rate decay
            optimizer.param_groups[0]['lr'] -= 1 / max_epochs

            # Save best model
            if np.argmin(loss_log[:epoch + 1]) == epoch:
                torch.save(self.net.state_dict(), self.target_path)

            if verbose and (epoch + 1) % 50 == 0:
                print(f'--- L-BFGS Iter {epoch + 1}, loss={loss_log[epoch]:.4f}')

        best_iter = np.argmin(loss_log) + 1
        print(f'[MTLTrainer] L-BFGS training terminated, best iter={best_iter}')
        self.load(self.target_path)

    def _validate(self, val_loader, loss_fn):
        """Compute validation loss."""
        self.net.eval()
        total_loss = 0.0
        n_samples = 0

        with torch.no_grad():
            for batchdata in val_loader:
                g, node_feats, edge_feats, labels, task_ids = batchdata

                g = g.to(self.cuda)
                node_feats = node_feats.to(self.cuda)
                edge_feats = edge_feats.to(self.cuda)
                task_ids = task_ids.to(self.cuda)

                labels_norm = self._normalize_labels(labels, task_ids).to(self.cuda)

                preds = self.net(g, node_feats, edge_feats, task_ids)
                loss = self.masked_loss(preds, labels_norm, task_ids, loss_fn, reduction='sum')

                total_loss += loss.item() * labels.size(0)
                n_samples += labels.size(0)

        return total_loss / n_samples

    def inference(self, data_loader, task_id=None):
        """
        Run inference on data.

        Args:
            data_loader: Data loader
            task_id: If provided, use this task_id for all samples.
                     If None, use task_ids from data loader.

        Returns:
            Predictions (denormalized)
        """
        self.net.eval()
        all_preds = []
        all_task_ids = []

        with torch.no_grad():
            for batchdata in data_loader:
                g, node_feats, edge_feats, labels, task_ids = batchdata

                g = g.to(self.cuda)
                node_feats = node_feats.to(self.cuda)
                edge_feats = edge_feats.to(self.cuda)

                if task_id is not None:
                    # Use specified task_id for all samples
                    batch_task_ids = torch.full((labels.size(0),), task_id, dtype=torch.long)
                else:
                    batch_task_ids = task_ids

                batch_task_ids = batch_task_ids.to(self.cuda)

                preds = self.net(g, node_feats, edge_feats, batch_task_ids)
                all_preds.append(preds.cpu().numpy())
                all_task_ids.append(batch_task_ids.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_task_ids = np.concatenate(all_task_ids)

        # Denormalize predictions
        all_preds = self._denormalize_preds(all_preds, all_task_ids)

        return all_preds

    def inference_all_heads(self, data_loader):
        """
        Run inference using all heads (for ensemble prediction).

        Returns:
            Predictions from all heads: (N, n_tasks)
        """
        self.net.eval()
        all_preds = []

        with torch.no_grad():
            for batchdata in data_loader:
                g, node_feats, edge_feats, labels, task_ids = batchdata

                g = g.to(self.cuda)
                node_feats = node_feats.to(self.cuda)
                edge_feats = edge_feats.to(self.cuda)

                # Get predictions from all heads
                preds = self.net.forward_all_heads(g, node_feats, edge_feats)
                all_preds.append(preds.cpu().numpy())

        all_preds = np.vstack(all_preds)

        # Denormalize each task's predictions
        for task_id in range(self.net.n_tasks):
            if task_id in self.task_means and task_id in self.task_stds:
                mean = self.task_means[task_id]
                std = self.task_stds[task_id]
                all_preds[:, task_id] = all_preds[:, task_id] * std + mean

        return all_preds


def compute_task_metrics(y_true, y_pred, task_ids, task_names):
    """
    Compute metrics for each task.

    Args:
        y_true: Ground truth labels (N, 1)
        y_pred: Predictions (N, 1)
        task_ids: Task IDs (N,)
        task_names: List of task names

    Returns:
        Dict of metrics per task
    """
    from sklearn.metrics import mean_absolute_error, r2_score

    metrics = {}
    unique_tasks = np.unique(task_ids)

    for task_id in unique_tasks:
        mask = task_ids == task_id
        y_t = y_true[mask].flatten()
        y_p = y_pred[mask].flatten()

        mae = mean_absolute_error(y_t, y_p)
        medae = np.median(np.abs(y_t - y_p))
        r2 = r2_score(y_t, y_p)

        task_name = task_names[task_id] if task_id < len(task_names) else f'Task_{task_id}'
        metrics[task_name] = {
            'MAE': mae,
            'MedAE': medae,
            'R2': r2,
            'n_samples': int(mask.sum())
        }

    return metrics
