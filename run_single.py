"""
Single-Head Model Experiment Script

Usage:
    # Basic: system_id as categorical feature (always included), task-wise normalization
    python run_single.py --tasks FEM_long FEM_short RIKEN --normalization task_wise

    # With chromatography type (HILIC/RP) as additional feature
    python run_single.py --tasks FEM_long Cao_HILIC MTBLS87 --include_chrom_type --normalization task_wise

    # Compare normalization: global vs task-wise
    python run_single.py --tasks FEM_long FEM_short RIKEN --normalization global

This script trains a single-head model with:
- System ID as categorical feature (always included for task distinction)
- Optional: Chromatography type (HILIC/RP) as additional categorical feature
- Task-wise or global normalization
- Pre-training → Fine-tuning strategy
"""

import numpy as np
import os
import csv
import torch
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score

from dataset_single import SingleHeadDataset, collate_single_graphs, CHROMATOGRAPHY_TYPE
from gnn import GINSinglePredictor
from model_single import SingleHeadTrainer, compute_system_metrics


# Available datasets
AVAILABLE_DATASETS = [
    'Cao_HILIC', 'Eawag_XBridgeC18', 'FEM_lipids', 'FEM_long',
    'FEM_orbitrap_plasma', 'FEM_orbitrap_urine', 'FEM_short',
    'IPB_Halle', 'LIFE_new', 'LIFE_old', 'MassBank1', 'MassBank2',
    'MetaboBase', 'MPI_Symmetry', 'MTBLS87', 'PFR-TK72', 'RIKEN',
    'RIKEN_PlaSMA', 'UFZ_Phenomenex', 'UniToyama_Atlantis'
]


def parse_args():
    parser = ArgumentParser(description='Single-Head Model for Retention Time Prediction')

    # Dataset arguments
    parser.add_argument('--tasks', '-t', type=str, nargs='+', required=True,
                        choices=AVAILABLE_DATASETS,
                        help='List of dataset names to combine')
    parser.add_argument('--cvid', '-k', type=int, choices=list(range(10)), default=0,
                        help='Cross-validation fold ID (0-9)')
    parser.add_argument('--seed', '-s', type=int, default=134,
                        help='Random seed')

    # Normalization and categorical feature arguments
    parser.add_argument('--normalization', '-n', type=str, default='task_wise',
                        choices=['task_wise', 'global'],
                        help='Normalization mode: task_wise or global')
    parser.add_argument('--include_chrom_type', action='store_true',
                        help='Include chromatography type (HILIC/RP) as additional categorical feature')
    parser.add_argument('--cat_embedding_dim', type=int, default=0,
                        help='If > 0, project categorical features to this dimension')

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
                        help='Hidden dimension for prediction head')

    # Training arguments
    parser.add_argument('--method', '-m', type=str, default='finetune',
                        choices=['scratch', 'finetune', 'feature'],
                        help='Training method')
    parser.add_argument('--opt', '-o', type=str, default='adam',
                        choices=['adam', 'lbfgs'],
                        help='Optimizer')
    parser.add_argument('--encoder_lr', type=float, default=1e-5,
                        help='Learning rate for encoder')
    parser.add_argument('--head_lr', type=float, default=1e-4,
                        help='Learning rate for head')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=500,
                        help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')

    # Path arguments
    parser.add_argument('--source_path', type=str, default='./model/model_SMRT.pt',
                        help='Path to pre-trained model')
    parser.add_argument('--output_dir', type=str, default='./model_single/',
                        help='Directory to save models')
    parser.add_argument('--result_dir', type=str, default='./exps_results_single/',
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

    # Categorical feature settings
    # system_id is always included (essential for one-head model to distinguish tasks)
    include_system_id = True
    include_chrom_type = args.include_chrom_type

    # Configuration
    system_names = args.tasks
    n_systems = len(system_names)

    print(f'\n{"="*60}')
    print(f'Single-Head Model Experiment')
    print(f'{"="*60}')
    print(f'Systems ({n_systems}): {system_names}')
    print(f'CV Fold: {args.cvid}')
    print(f'Normalization: {args.normalization}')
    print(f'Categorical features: system_id (always), chrom_type={include_chrom_type}')
    print(f'Method: {args.method}')
    print(f'Optimizer: {args.opt}')
    print(f'Device: {cuda}')
    print(f'{"="*60}\n')

    # Print chromatography type info
    print('Chromatography types:')
    for name in system_names:
        chrom = CHROMATOGRAPHY_TYPE.get(name, 'RP')
        print(f'  {name}: {chrom}')
    print()

    # Load datasets
    print('[1] Loading datasets...')
    train_dataset = SingleHeadDataset(
        system_names,
        cv_id=args.cvid,
        split='trn',
        seed=args.seed,
        normalization=args.normalization,
        include_system_id=include_system_id,
        include_chrom_type=include_chrom_type
    )
    test_dataset = SingleHeadDataset(
        system_names,
        cv_id=args.cvid,
        split='tst',
        seed=args.seed,
        normalization=args.normalization,
        include_system_id=include_system_id,
        include_chrom_type=include_chrom_type
    )

    # Get feature dimensions
    node_in_feats = train_dataset.get_node_feat_dim()
    edge_in_feats = train_dataset.get_edge_feat_dim()
    cat_feat_dim = train_dataset.get_cat_feat_dim()

    print(f'\nTotal training samples: {len(train_dataset)}')
    print(f'Total test samples: {len(test_dataset)}')
    print(f'Node features: {node_in_feats}, Edge features: {edge_in_feats}')
    print(f'Categorical features dim: {cat_feat_dim}')

    # Create model
    print('\n[2] Creating model...')
    model = GINSinglePredictor(
        node_in_feats=node_in_feats,
        edge_in_feats=edge_in_feats,
        cat_feat_dim=cat_feat_dim,
        num_layers=args.num_layers,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        readout=args.readout,
        predictor_hidden_feats=args.predictor_hidden,
        cat_embedding_dim=args.cat_embedding_dim
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

        train_loader = DataLoader(
            train_subset,
            batch_size=min(args.batch_size, train_size),
            shuffle=True,
            collate_fn=collate_single_graphs,
            drop_last=True
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_single_graphs
        )

        print(f'Train/Val split: {train_size}/{val_size}')

    else:  # lbfgs
        train_loader = DataLoader(
            train_dataset,
            batch_size=len(train_dataset),
            shuffle=False,
            collate_fn=collate_single_graphs
        )
        val_loader = None

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_single_graphs
    )

    # Create trainer
    print('\n[4] Setting up trainer...')

    cat_str = ''
    if include_system_id:
        cat_str += '_sysid'
    if include_chrom_type:
        cat_str += '_chrom'
    if not cat_str:
        cat_str = '_nocat'

    target_path = os.path.join(
        args.output_dir,
        f'model_single_{"_".join(system_names[:3])}{"_etc" if n_systems > 3 else ""}'
        f'_{args.normalization}{cat_str}_cv{args.cvid}.pt'
    )

    trainer = SingleHeadTrainer(
        net=model,
        source_path=args.source_path,
        cuda=cuda,
        norm_stats=train_dataset.get_normalization_stats()
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
    test_system_ids = test_dataset.system_id

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

    # Compute per-system metrics
    system_metrics = compute_system_metrics(test_labels, test_preds, test_system_ids, system_names)

    print(f'\n{"="*60}')
    print('PER-SYSTEM TEST RESULTS')
    print(f'{"="*60}')
    for sys_name, metrics in system_metrics.items():
        chrom = CHROMATOGRAPHY_TYPE.get(sys_name, 'RP')
        print(f'\n{sys_name} ({chrom}, n={metrics["n_samples"]}):')
        print(f'  MAE: {metrics["MAE"]:.4f}')
        print(f'  MedAE: {metrics["MedAE"]:.4f}')
        print(f'  R2: {metrics["R2"]:.4f}')

    # Save results
    print('\n[7] Saving results...')
    save_results(args, system_names, system_metrics, overall_mae, overall_medae, overall_r2,
                 include_chrom_type)

    print(f'\n{"="*60}')
    print('DONE!')
    print(f'{"="*60}')

    return system_metrics


def save_results(args, system_names, system_metrics, overall_mae, overall_medae, overall_r2,
                 include_chrom_type):
    """Save results to CSV files."""

    # system_id is always included
    cat_features_str = 'system_id'
    if include_chrom_type:
        cat_features_str += '+chrom_type'

    # Save overall results
    overall_file = os.path.join(args.result_dir, 'single_overall.csv')
    file_exists = os.path.isfile(overall_file)

    with open(overall_file, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(['Systems', 'N_Systems', 'CV_Fold', 'Seed', 'Method', 'Optimizer',
                           'Normalization', 'Cat_Features', 'MAE', 'MedAE', 'R2'])

        writer.writerow([
            '+'.join(system_names), len(system_names), args.cvid, args.seed,
            args.method, args.opt, args.normalization, cat_features_str,
            overall_mae, overall_medae, overall_r2
        ])

    print(f'Overall results saved to {overall_file}')

    # Save per-system results
    for sys_name, metrics in system_metrics.items():
        sys_file = os.path.join(args.result_dir, f'single_{sys_name}.csv')
        file_exists = os.path.isfile(sys_file)

        with open(sys_file, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(['System', 'CV_Fold', 'Seed', 'Method', 'Optimizer',
                               'Normalization', 'Cat_Features',
                               'N_Samples', 'MAE', 'MedAE', 'R2', 'Joint_Systems'])

            writer.writerow([
                sys_name, args.cvid, args.seed, args.method, args.opt,
                args.normalization, cat_features_str,
                metrics['n_samples'], metrics['MAE'], metrics['MedAE'], metrics['R2'],
                '+'.join(system_names)
            ])

    print(f'Per-system results saved to {args.result_dir}')


if __name__ == '__main__':
    main()
