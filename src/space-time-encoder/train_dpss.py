#!/usr/bin/env python3
"""
Training script for STE+DPSS on ACE, with sequential hyperparam runs.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from models.ste_encoder import STEEncoder
from utils.ace_data import prepare_ace_data
from utils.training import train_epoch, evaluate, create_optimizer, EarlyStopping


# Top-level config
SEED = 42  # Default seed (can be overridden by bash runner)

BASE = {
    'data_path': '/scratch/local/arra4944_images/ace/',
    'train_frac': 0.01,
    'val_frac': 0.01,
    'test_frac': 0.01,
    'batch_size': 40000,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'max_epochs': 500,
    'patience': 15,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # Regularization knobs
    'ortho_weight_final': 1e-3,
    'ortho_weight_space': 1e-3,
    'ortho_weight_time': 1e-3,
    'ortho_exponent': 1,
    'normality_flag': False,
    'time_grad_penalty_weight': 1e-6,

    # DPSS defaults
    'dpss_N': 365 * 4,  # 1460
    'dpss_concentration_threshold': 0.85,
}

# Grid of DPSS runs
RUNS = [
    # (NW, K, optimized)
    (10, 32, False),
    (12, 40, False),
    (16, 40, False),
    (20, 40, False),
    (12, 40, True),
    (16, 40, True),
    (20, 40, True),
]


def main():
    print("=" * 50)
    print("STE + DPSS Training on ACE")
    print("=" * 50)

    # Set seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load data once using shared utility
    train_ds, val_ds, test_ds, normalizer = prepare_ace_data(
        BASE['data_path'],
        BASE['train_frac'],
        BASE['val_frac'],
        BASE['test_frac'],
        seed=SEED
    )

    train_loader = DataLoader(train_ds, batch_size=BASE['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BASE['batch_size'])
    test_loader = DataLoader(test_ds, batch_size=BASE['batch_size'])

    for (NW, K, OPT) in RUNS:
        print("\n" + "-" * 50)
        tag = f"dpss_NW{NW}_K{K}_{'opt' if OPT else 'base'}"
        print(f"Run: {tag}")

        model = STEEncoder(
            spatial_L=20,
            temporal_type='dpss',
            temporal_dim=K,
            combination='concatenate',
            hidden_dim=1024,
            num_layers=4,
            output_dim=8,
            ortho_weight=BASE['ortho_weight_final'],
            ortho_weight_space=BASE['ortho_weight_space'],
            ortho_weight_time=BASE['ortho_weight_time'],
            ortho_exponent=BASE['ortho_exponent'],
            normality_flag=BASE['normality_flag'],
            time_grad_penalty_weight=BASE['time_grad_penalty_weight'],
            dpss_N=BASE['dpss_N'],
            dpss_NW=NW,
            dpss_optimized=OPT,
            dpss_concentration_threshold=BASE['dpss_concentration_threshold'],
        ).to(BASE['device'])

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        optimizer = create_optimizer(model, lr=BASE['learning_rate'], weight_decay=BASE['weight_decay'])
        early_stopping = EarlyStopping(patience=BASE['patience'])

        BEST_CKPT = f"best_ste_{tag}_seed{SEED}.pt"

        for epoch in range(BASE['max_epochs']):
            print(f"\nEpoch {epoch+1}/{BASE['max_epochs']}  [{tag}]")

            tr_mse, tr_reg = train_epoch(model, train_loader, optimizer, BASE['device'])
            va_rmse, va_mae = evaluate(model, val_loader, BASE['device'], normalizer)

            print(f"Train MSE: {tr_mse:.4f}, Reg: {tr_reg:.4f}")
            print(f"Val RMSE: {va_rmse:.4f}, MAE: {va_mae:.4f}")

            if early_stopping(va_rmse):
                torch.save(model.state_dict(), BEST_CKPT)
                print("  -> Saved best model")

            if early_stopping.should_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Final test
        print("\nTesting best checkpoint...")
        model.load_state_dict(torch.load(BEST_CKPT, map_location=BASE['device']))
        te_rmse, te_mae = evaluate(model, test_loader, BASE['device'], normalizer)
        print(f"[{tag}_seed{SEED}] Test RMSE: {te_rmse:.4f} K   Test MAE: {te_mae:.4f} K")

        # Reset early stopping for next run
        early_stopping = EarlyStopping(patience=BASE['patience'])


if __name__ == "__main__":
    main()
