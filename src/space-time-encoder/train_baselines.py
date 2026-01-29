#!/usr/bin/env python3
"""
Training script for STE baseline temporal encodings on ACE dataset.

Usage:
    python train_baselines.py --temporal_type legendre --seed 42
    python train_baselines.py --temporal_type fourier --use_reg --seed 42
"""

import os
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.ste_encoder import STEEncoder
from utils.ace_data import prepare_ace_data
from utils.training import train_epoch, evaluate, create_optimizer, EarlyStopping


def parse_args():
    p = argparse.ArgumentParser("Train STE baselines for ACE")
    p.add_argument("--data_path", type=str, default="/scratch/local/arra4944_images/ace/")
    p.add_argument("--temporal_type", choices=["no_time", "time_copy", "triangle", "monomial", "legendre", "fourier"],
                   default="legendre")
    p.add_argument("--use_reg", action="store_true", help="Enable regularization")
    p.add_argument("--ortho_weight", type=float, default=None,
                   help="Override ortho weight (default: 1e-3 if --use_reg, else 0)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.01)
    p.add_argument("--val_frac", type=float, default=0.01)
    p.add_argument("--test_frac", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=40000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--max_epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Thread settings for reproducibility
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Set regularization weights
    if args.ortho_weight is not None:
        ortho_weight_final = args.ortho_weight
        reg_suffix = f"alpha{args.ortho_weight}"
    elif args.use_reg:
        ortho_weight_final = 1.0
        reg_suffix = "reg"
    else:
        ortho_weight_final = 0.0
        reg_suffix = "noreg"

    # Load data
    train_ds, val_ds, test_ds, normalizer = prepare_ace_data(
        args.data_path, args.train_frac, args.val_frac, args.test_frac, seed=args.seed
    )

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=16)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=16)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, num_workers=16)

    # Create model
    model = STEEncoder(
        spatial_L=20,
        temporal_type=args.temporal_type,
        temporal_dim=40,
        combination='concatenate',
        hidden_dim=1024,
        num_layers=4,
        output_dim=8,
        ortho_weight=ortho_weight_final,
        ortho_weight_space=0.0,
        ortho_weight_time=0.0,
        normality_flag=False,
        ortho_exponent=1,
        time_grad_penalty_weight=0.0,
    ).to(args.device)

    optimizer = create_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(patience=args.patience)

    best_path = Path(f"best_{args.temporal_type}_{reg_suffix}_seed{args.seed}.pt")

    print(f"Training {args.temporal_type} (alpha={ortho_weight_final})")
    print(f"Checkpoint: {best_path}")

    for epoch in range(args.max_epochs):
        mse, reg = train_epoch(model, train_dl, optimizer, args.device)
        val_rmse, val_mae = evaluate(model, val_dl, args.device, normalizer)

        print(f"Epoch {epoch+1}: TrainMSE={mse:.4f} Reg={reg:.4f} | Val RMSE={val_rmse:.4f} MAE={val_mae:.4f}")

        if early_stopping(val_rmse):
            torch.save(model.state_dict(), best_path)
            print(f"  -> Saved best model")

        if early_stopping.should_stop:
            print(f"Early stopping @ epoch {epoch+1}")
            break

    # Final test evaluation
    print("\n" + "=" * 50)
    print("Final Test Evaluation")
    model.load_state_dict(torch.load(best_path, map_location=args.device))
    test_rmse, test_mae = evaluate(model, test_dl, args.device, normalizer)
    print(f"Test RMSE: {test_rmse:.4f} K")
    print(f"Test MAE:  {test_mae:.4f} K")


if __name__ == "__main__":
    main()
