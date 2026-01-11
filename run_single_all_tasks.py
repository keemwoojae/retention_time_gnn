"""
Run Single-Head Model Experiments for All 20 Tasks

Experiments:
1. With chrom_type + task-wise normalization
2. With chrom_type + global normalization
3. Without chrom_type + task-wise normalization
4. Without chrom_type + global normalization

Each experiment runs with 2 seeds × 10 CV folds = 20 runs per condition
"""

import subprocess
import sys
import os
from itertools import product

# All 20 datasets
ALL_TASKS = [
    'Cao_HILIC', 'Eawag_XBridgeC18', 'FEM_lipids', 'FEM_long',
    'FEM_orbitrap_plasma', 'FEM_orbitrap_urine', 'FEM_short',
    'IPB_Halle', 'LIFE_new', 'LIFE_old', 'MassBank1', 'MassBank2',
    'MetaboBase', 'MPI_Symmetry', 'MTBLS87', 'PFR-TK72', 'RIKEN',
    'RIKEN_PlaSMA', 'UFZ_Phenomenex', 'UniToyama_Atlantis'
]

# Experiment configurations
SEEDS = [134, 42]
CV_FOLDS = list(range(10))

# Conditions: (include_chrom_type, normalization)
CONDITIONS = [
    (True, 'task_wise'),   # with chrom_type, task-wise norm
    (True, 'global'),      # with chrom_type, global norm
    (False, 'task_wise'),  # without chrom_type, task-wise norm
    (False, 'global'),     # without chrom_type, global norm
]


def run_experiment(tasks, seed, cvid, include_chrom_type, normalization,
                   method='finetune', opt='lbfgs', verbose=False):
    """Run a single experiment."""

    cmd = [
        sys.executable, 'run_single.py',
        '--tasks', *tasks,
        '--seed', str(seed),
        '--cvid', str(cvid),
        '--normalization', normalization,
        '--method', method,
        '--opt', opt,
    ]

    if include_chrom_type:
        cmd.append('--include_chrom_type')

    if verbose:
        cmd.append('-v')

    print(f"\n{'='*60}")
    print(f"Running: seed={seed}, cvid={cvid}, chrom_type={include_chrom_type}, norm={normalization}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=not verbose)

    if result.returncode != 0:
        print(f"[ERROR] Experiment failed!")
        if not verbose:
            print(result.stderr.decode())
        return False

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run all single-head experiments')
    parser.add_argument('--method', '-m', type=str, default='finetune',
                        choices=['scratch', 'finetune', 'feature'])
    parser.add_argument('--opt', '-o', type=str, default='lbfgs',
                        choices=['adam', 'lbfgs'])
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print commands without running')
    args = parser.parse_args()

    # Create output directories
    os.makedirs('./model_single/', exist_ok=True)
    os.makedirs('./exps_results_single/', exist_ok=True)

    total_runs = len(SEEDS) * len(CV_FOLDS) * len(CONDITIONS)
    current_run = 0
    failed_runs = []

    print(f"\n{'#'*60}")
    print(f"# Single-Head Model: All 20 Tasks Experiment")
    print(f"# Total runs: {total_runs}")
    print(f"# Seeds: {SEEDS}")
    print(f"# CV folds: {len(CV_FOLDS)}")
    print(f"# Conditions: {len(CONDITIONS)}")
    print(f"{'#'*60}\n")

    for include_chrom, norm in CONDITIONS:
        condition_name = f"chrom={include_chrom}_norm={norm}"
        print(f"\n{'*'*60}")
        print(f"* Condition: {condition_name}")
        print(f"{'*'*60}")

        for seed in SEEDS:
            for cvid in CV_FOLDS:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] seed={seed}, cvid={cvid}, {condition_name}")

                if args.dry_run:
                    chrom_flag = '--include_chrom_type' if include_chrom else ''
                    print(f"  CMD: python run_single.py --tasks {' '.join(ALL_TASKS[:3])}... "
                          f"--seed {seed} --cvid {cvid} --normalization {norm} {chrom_flag}")
                    continue

                success = run_experiment(
                    ALL_TASKS, seed, cvid, include_chrom, norm,
                    method=args.method, opt=args.opt, verbose=args.verbose
                )

                if not success:
                    failed_runs.append((seed, cvid, include_chrom, norm))

    # Summary
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT COMPLETE")
    print(f"# Successful: {total_runs - len(failed_runs)}/{total_runs}")
    if failed_runs:
        print(f"# Failed runs:")
        for run in failed_runs:
            print(f"#   seed={run[0]}, cvid={run[1]}, chrom={run[2]}, norm={run[3]}")
    print(f"{'#'*60}")

    print("\nNow run 'python aggregate_single_results.py' to aggregate results.")


if __name__ == '__main__':
    main()
