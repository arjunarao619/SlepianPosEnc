#!/usr/bin/env python3
"""
Training script for STE (Space-Time Encoder) on ACE dataset.

This is the primary training script that matches the STE paper setup.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from models.ste_encoder import STEEncoder
from utils.ace_data import prepare_ace_data
from utils.training import train_epoch, evaluate, create_optimizer, EarlyStopping


# Configuration
USE_REG = True  # Set to True if we want to regularize according to the STE paper

config = {
    'data_path': '/scratch/local/arra4944_images/ace/',
    'train_frac': 0.01,
    'val_frac': 0.01,
    'test_frac': 0.01,
    'batch_size': 40000,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'max_epochs': 500,
    'patience': 15,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # Regularization hyperparams
    'ortho_weight_final': 1e-3,
    'ortho_weight_space': 1e-3,
    'ortho_weight_time': 1e-3,
    'ortho_exponent': 1,
    'normality_flag': False,
    'time_grad_penalty_weight': 1e-6,
}

if not USE_REG:
    config.update({
        'ortho_weight_final': 0.0,
        'ortho_weight_space': 0.0,
        'ortho_weight_time': 0.0,
        'time_grad_penalty_weight': 0.0,
    })
    CKPT_NAME = 'best_ste_model_noreg.pt'
else:
    CKPT_NAME = 'best_ste_model_reg.pt'


def main():
    """Main training loop - matching STE setup."""
    print("=" * 50)
    print("STE Training on ACE Dataset")
    print("=" * 50)

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data using shared utility
    train_ds, val_ds, test_ds, normalizer = prepare_ace_data(
        config['data_path'],
        config['train_frac'],
        config['val_frac'],
        config['test_frac'],
        seed=42
    )

    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'])

    # Create model (matching paper architecture)
    model = STEEncoder(
        spatial_L=20,
        temporal_type='',  # No temporal encoding in base STE
        temporal_dim=40,
        combination='concatenate',
        hidden_dim=1024,
        num_layers=4,
        output_dim=8,
        ortho_weight=config['ortho_weight_final'],
        ortho_weight_space=config['ortho_weight_space'],
        ortho_weight_time=config['ortho_weight_time'],
        ortho_exponent=config['ortho_exponent'],
        normality_flag=config['normality_flag'],
        time_grad_penalty_weight=config['time_grad_penalty_weight'],
    ).to(config['device'])

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer with weight decay only on head
    optimizer = create_optimizer(
        model,
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Training with early stopping
    early_stopping = EarlyStopping(patience=config['patience'])

    for epoch in range(config['max_epochs']):
        print(f"\nEpoch {epoch+1}/{config['max_epochs']}")

        mse_loss, reg_loss = train_epoch(model, train_loader, optimizer, config['device'])
        val_rmse, val_mae = evaluate(model, val_loader, config['device'], normalizer)

        print(f"Train MSE: {mse_loss:.4f}, Reg Loss: {reg_loss:.4f}")
        print(f"Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}")

        if early_stopping(val_rmse):
            torch.save(model.state_dict(), CKPT_NAME)
            print("  -> Saved best model")

        if early_stopping.should_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Final test evaluation
    print("\n" + "=" * 50)
    print("Final Test Evaluation")
    model.load_state_dict(torch.load(CKPT_NAME, map_location=config['device']))
    test_rmse, test_mae = evaluate(model, test_loader, config['device'], normalizer)
    print(f"Test RMSE: {test_rmse:.4f} K")
    print(f"Test MAE: {test_mae:.4f} K")


if __name__ == "__main__":
    main()
