#!/usr/bin/env python3
"""
Parameterized DPSS training for systematic NW sensitivity experiments.

Usage:
    python train_dpss_sweep.py --nw 20 --seed 42
    python train_dpss_sweep.py --nw 15 --seed 123 --optimized

For full sweep:
    for NW in 10 12 15 18 20 22 25; do
        for SEED in 42 123 456; do
            python train_dpss_sweep.py --nw $NW --seed $SEED
        done
    done
"""

import os
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

from models.ste_encoder import STEEncoder
from utils.ace_data import prepare_ace_data
from utils.training import train_epoch, evaluate_detailed, create_optimizer, EarlyStopping


def parse_args():
    parser = argparse.ArgumentParser(description='Train STE with DPSS temporal encoding')

    # DPSS parameters
    parser.add_argument('--nw', type=int, required=True,
                        help='Time-bandwidth product NW (e.g., 10, 15, 20, 25)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--k', type=int, default=40,
                        help='Number of DPSS sequences (capped at 2*NW-1)')
    parser.add_argument('--concentration-threshold', type=float, default=0.85,
                        help='Eigenvalue threshold for sequence selection')
    parser.add_argument('--optimized', action='store_true',
                        help='Use optimized DPSS with learnable projection')

    # Data parameters
    parser.add_argument('--data-path', type=str,
                        default='/scratch/local/arra4944_images/ace/',
                        help='Path to ACE dataset')
    parser.add_argument('--train-frac', type=float, default=0.01)
    parser.add_argument('--val-frac', type=float, default=0.01)
    parser.add_argument('--test-frac', type=float, default=0.01)

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=40000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--max-epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=15)

    # Regularization
    parser.add_argument('--ortho-weight', type=float, default=1e-3)
    parser.add_argument('--ortho-weight-space', type=float, default=1e-3)
    parser.add_argument('--ortho-weight-time', type=float, default=1e-3)
    parser.add_argument('--time-grad-weight', type=float, default=1e-6)

    # Output
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                        help='Directory for saving checkpoints and results')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tag for this run
    opt_tag = 'opt' if args.optimized else 'base'
    run_tag = f"dpss_NW{args.nw}_K{args.k}_{opt_tag}_seed{args.seed}"
    print(f"\n{'='*60}")
    print(f"Training: {run_tag}")
    print(f"{'='*60}")

    # Load data using shared utility
    train_ds, val_ds, test_ds, normalizer = prepare_ace_data(
        args.data_path, args.train_frac, args.val_frac, args.test_frac, seed=args.seed
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Effective K (capped at Shannon number)
    k_effective = min(args.k, 2 * args.nw - 1)

    # Create model
    model = STEEncoder(
        spatial_L=20,
        temporal_type='dpss',
        temporal_dim=k_effective,
        combination='concatenate',
        hidden_dim=1024,
        num_layers=4,
        output_dim=8,
        ortho_weight=args.ortho_weight,
        ortho_weight_space=args.ortho_weight_space,
        ortho_weight_time=args.ortho_weight_time,
        ortho_exponent=1,
        normality_flag=False,
        time_grad_penalty_weight=args.time_grad_weight,
        dpss_N=365 * 4,  # 1460 timesteps
        dpss_NW=args.nw,
        dpss_optimized=args.optimized,
        dpss_concentration_threshold=args.concentration_threshold,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Create optimizer
    optimizer = create_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(patience=args.patience)

    # Training loop
    best_ckpt_path = output_dir / f"best_{run_tag}.pt"
    history = []

    for epoch in range(args.max_epochs):
        # Train
        tr_mse, tr_reg = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_metrics = evaluate_detailed(model, val_loader, device, normalizer)
        val_rmse = val_metrics['rmse_mean']

        # Log
        history.append({
            'epoch': epoch + 1,
            'train_mse': tr_mse,
            'train_reg': tr_reg,
            'val_rmse': val_rmse,
            'val_mae': val_metrics['mae_mean']
        })

        print(f"Epoch {epoch+1}/{args.max_epochs}: "
              f"Train MSE={tr_mse:.4f}, Reg={tr_reg:.6f}, "
              f"Val RMSE={val_rmse:.4f}")

        # Early stopping
        if early_stopping(val_rmse):
            torch.save(model.state_dict(), best_ckpt_path)
            print("  -> Saved best model")

        if early_stopping.should_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Final test evaluation
    print("\nLoading best checkpoint for final evaluation...")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_metrics = evaluate_detailed(model, test_loader, device, normalizer)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS: {run_tag}")
    print(f"{'='*60}")
    print(f"Test RMSE: {test_metrics['rmse_mean']:.4f} K")
    print(f"Test MAE:  {test_metrics['mae_mean']:.4f} K")
    print(f"Per-variable RMSE: {[f'{r:.3f}' for r in test_metrics['rmse_per_var']]}")

    # Save results
    results = {
        'run_tag': run_tag,
        'args': vars(args),
        'k_effective': k_effective,
        'n_params': n_params,
        'best_val_rmse': early_stopping.best_score,
        'test_rmse': test_metrics['rmse_mean'],
        'test_mae': test_metrics['mae_mean'],
        'test_rmse_per_var': test_metrics['rmse_per_var'],
        'test_mae_per_var': test_metrics['mae_per_var'],
        'epochs_trained': len(history),
        'timestamp': datetime.now().isoformat(),
        'history': history
    }

    results_path = output_dir / f"results_{run_tag}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"Checkpoint saved to: {best_ckpt_path}")


if __name__ == "__main__":
    main()
