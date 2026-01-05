"""
Multi-Task Learning Experiment Script

Usage:
    python run_mtl.py --tasks FEM_long FEM_short RIKEN --method finetune --opt adam

This script trains a multi-task learning model with:
- Hard Parameter Sharing (shared GIN encoder + task-specific heads)
- Masked Loss (only compute loss for samples with ground truth)
- Task-wise normalization
- Pre-training → Fine-tuning strategy
"""

import numpy as np
import os
import csv
import torch
from argparse import ArgumentParser

from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_absolute_error, r2_score

from dataset_mtl import MTLDataset, collate_mtl_graphs, TaskBalancedSampler
from gnn import GINMTLPredictor
from model_mtl import MTLTrainer, compute_task_metrics


# Available datasets
AVAILABLE_DATASETS = [
    'Cao_HILIC', 'Eawag_XBridgeC18', 'FEM_lipids', 'FEM_long',
    'FEM_orbitrap_plasma', 'FEM_orbitrap_urine', 'FEM_short',
    'IPB_Halle', 'LIFE_new', 'LIFE_old', 'MassBank1', 'MassBank2',
    'MetaboBase', 'MPI_Symmetry', 'MTBLS87', 'PFR-TK72', 'RIKEN',
    'RIKEN_PlaSMA', 'UFZ_Phenomenex', 'UniToyama_Atlantis'
]


def parse_args():
    parser = ArgumentParser(description='Multi-Task Learning for Retention Time Prediction')

    # Dataset arguments
    parser.add_argument('--tasks', '-t', type=str, nargs='+', required=True,
                        choices=AVAILABLE_DATASETS,
                        help='List of dataset names to use as tasks')
    parser.add_argument('--cvid', '-k', type=int, choices=list(range(10)), default=0,
                        help='Cross-validation fold ID (0-9)')
    parser.add_argument('--seed', '-s', type=int, default=134,
                        help='Random seed')

    # Model arguments
    parser.add_argument('--num_layers', type=int, default=5,
                        help='Number of GIN layers')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--readout', type=str, default='mean',
                        choices=['sum', 'mean', 'max', 'attention', 'set2set'],
                        help='Readout method')
    parser.add_argument('--predictor_hidden', type=int, default=256,
                        help='Hidden dimension for prediction heads')

    # Training arguments
    parser.add_argument('--method', '-m', type=str, default='finetune',
                        choices=['scratch', 'finetune', 'feature'],
                        help='Training method: scratch, finetune (load encoder, train all), '
                             'feature (load encoder, freeze, train heads only)')
    parser.add_argument('--opt', '-o', type=str, default='adam',
                        choices=['adam', 'lbfgs'],
                        help='Optimizer')
    parser.add_argument('--encoder_lr', type=float, default=1e-5,
                        help='Learning rate for encoder (finetune mode)')
    parser.add_argument('--head_lr', type=float, default=1e-4,
                        help='Learning rate for heads')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=500,
                        help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    parser.add_argument('--balanced_batch', action='store_true',
                        help='Use task-balanced sampling for batches')

    # Path arguments
    parser.add_argument('--source_path', type=str, default='./model/model_SMRT.pt',
                        help='Path to pre-trained model')
    parser.add_argument('--output_dir', type=str, default='./model_mtl/',
                        help='Directory to save models')
    parser.add_argument('--result_dir', type=str, default='./exps_results_mtl/',
                        help='Directory to save results')

    # Other arguments
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        cuda = torch.device(f'cuda:{args.gpu}')
        torch.cuda.manual_seed(args.seed)
    else:
        cuda = torch.device('cpu')
        print('[Warning] CUDA not available, using CPU')

    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    # Task configuration
    task_names = args.tasks
    n_tasks = len(task_names)
    print(f'\n{"="*60}')
    print(f'Multi-Task Learning Experiment')
    print(f'{"="*60}')
    print(f'Tasks ({n_tasks}): {task_names}')
    print(f'CV Fold: {args.cvid}')
    print(f'Method: {args.method}')
    print(f'Optimizer: {args.opt}')
    print(f'Device: {cuda}')
    print(f'{"="*60}\n')

    # Load datasets
    print('[1] Loading datasets...')
    train_dataset = MTLDataset(task_names, cv_id=args.cvid, split='trn', seed=args.seed)
    test_dataset = MTLDataset(task_names, cv_id=args.cvid, split='tst', seed=args.seed)

    # Get feature dimensions
    node_in_feats = train_dataset.get_node_feat_dim()
    edge_in_feats = train_dataset.get_edge_feat_dim()

    print(f'\nTotal training samples: {len(train_dataset)}')
    print(f'Total test samples: {len(test_dataset)}')
    print(f'Node features: {node_in_feats}, Edge features: {edge_in_feats}')

    # Create model
    print('\n[2] Creating model...')
    model = GINMTLPredictor(
        node_in_feats=node_in_feats,
        edge_in_feats=edge_in_feats,
        n_tasks=n_tasks,
        num_layers=args.num_layers,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        readout=args.readout,
        predictor_hidden_feats=args.predictor_hidden
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')

    # Create data loaders
    print('\n[3] Creating data loaders...')

    if args.opt == 'adam':
        # Split training into train/val
        val_size = int(np.round(1/9 * len(train_dataset)))
        train_size = len(train_dataset) - val_size

        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )

        if args.balanced_batch:
            # Use task-balanced sampling
            # Note: This requires the full dataset, need to handle subsets
            train_loader = DataLoader(
                train_subset,
                batch_size=min(args.batch_size, train_size),
                shuffle=True,
                collate_fn=collate_mtl_graphs,
                drop_last=True
            )
        else:
            train_loader = DataLoader(
                train_subset,
                batch_size=min(args.batch_size, train_size),
                shuffle=True,
                collate_fn=collate_mtl_graphs,
                drop_last=True
            )

        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_mtl_graphs
        )

        print(f'Train/Val split: {train_size}/{val_size}')

    else:  # lbfgs
        train_loader = DataLoader(
            train_dataset,
            batch_size=len(train_dataset),
            shuffle=False,
            collate_fn=collate_mtl_graphs
        )
        val_loader = None

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_mtl_graphs
    )

    # Create trainer
    print('\n[4] Setting up trainer...')
    target_path = os.path.join(
        args.output_dir,
        f'model_mtl_{"_".join(task_names[:3])}{"_etc" if n_tasks > 3 else ""}_cv{args.cvid}.pt'
    )

    trainer = MTLTrainer(
        net=model,
        source_path=args.source_path,
        cuda=cuda,
        task_means=train_dataset.task_means,
        task_stds=train_dataset.task_stds
    )
    trainer.target_path = target_path

    # Training
    print('\n[5] Training...')
    print(f'Source model: {args.source_path}')
    print(f'Target path: {target_path}')

    trainer.training(
        train_loader=train_loader,
        val_loader=val_loader,
        method=args.method,
        opt=args.opt,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        max_epochs=args.max_epochs,
        verbose=args.verbose,
        weight_decay=args.weight_decay,
        patience=args.patience
    )

    # Evaluation
    print('\n[6] Evaluating on test set...')

    # Get predictions
    test_preds = trainer.inference(test_loader)
    test_labels = test_dataset.label
    test_task_ids = test_dataset.task_id

    # Compute overall metrics
    overall_mae = mean_absolute_error(test_labels, test_preds)
    overall_medae = np.median(np.abs(test_labels - test_preds))
    overall_r2 = r2_score(test_labels, test_preds)

    print(f'\n{"="*60}')
    print('OVERALL TEST RESULTS')
    print(f'{"="*60}')
    print(f'MAE: {overall_mae:.4f}')
    print(f'MedAE: {overall_medae:.4f}')
    print(f'R2: {overall_r2:.4f}')

    # Compute per-task metrics
    task_metrics = compute_task_metrics(test_labels, test_preds, test_task_ids, task_names)

    print(f'\n{"="*60}')
    print('PER-TASK TEST RESULTS')
    print(f'{"="*60}')
    for task_name, metrics in task_metrics.items():
        print(f'\n{task_name} (n={metrics["n_samples"]}):')
        print(f'  MAE: {metrics["MAE"]:.4f}')
        print(f'  MedAE: {metrics["MedAE"]:.4f}')
        print(f'  R2: {metrics["R2"]:.4f}')

    # Save results
    print('\n[7] Saving results...')
    save_results(args, task_names, task_metrics, overall_mae, overall_medae, overall_r2)

    print(f'\n{"="*60}')
    print('DONE!')
    print(f'{"="*60}')

    return task_metrics


def save_results(args, task_names, task_metrics, overall_mae, overall_medae, overall_r2):
    """Save results to CSV files."""

    # Save overall results
    overall_file = os.path.join(args.result_dir, 'mtl_overall.csv')
    file_exists = os.path.isfile(overall_file)

    with open(overall_file, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(['Tasks', 'N_Tasks', 'CV_Fold', 'Method', 'Optimizer',
                           'MAE', 'MedAE', 'R2'])

        writer.writerow([
            '+'.join(task_names), len(task_names), args.cvid,
            args.method, args.opt, overall_mae, overall_medae, overall_r2
        ])

    print(f'Overall results saved to {overall_file}')

    # Save per-task results
    for task_name, metrics in task_metrics.items():
        task_file = os.path.join(args.result_dir, f'mtl_{task_name}.csv')
        file_exists = os.path.isfile(task_file)

        with open(task_file, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(['Task', 'CV_Fold', 'Method', 'Optimizer',
                               'N_Samples', 'MAE', 'MedAE', 'R2', 'Joint_Tasks'])

            writer.writerow([
                task_name, args.cvid, args.method, args.opt,
                metrics['n_samples'], metrics['MAE'], metrics['MedAE'], metrics['R2'],
                '+'.join(task_names)
            ])

    print(f'Per-task results saved to {args.result_dir}')


if __name__ == '__main__':
    main()
