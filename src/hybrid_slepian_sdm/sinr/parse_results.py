"""
Parse and display S&T and IUCN results for all trained models.
"""

import os
import numpy as np
from pathlib import Path

experiments_dir = Path('experiments')

def load_results(exp_path):
    """Load SNT and IUCN results from an experiment directory."""
    results = {}

    snt_path = exp_path / 'results_snt.npy'
    iucn_path = exp_path / 'results_iucn.npy'

    if snt_path.exists():
        snt = np.load(snt_path, allow_pickle=True).item()
        results['snt_map'] = snt.get('mean_average_precision', None)
        results['snt_n_species'] = snt.get('num_eval_species_w_valid_ap', None)
    else:
        results['snt_map'] = None
        results['snt_n_species'] = None

    if iucn_path.exists():
        iucn = np.load(iucn_path, allow_pickle=True).item()
        results['iucn_map'] = iucn.get('mean_average_precision', None)
        results['iucn_n_species'] = iucn.get('num_eval_species_w_valid_ap', None)
    else:
        results['iucn_map'] = None
        results['iucn_n_species'] = None

    return results

def main():
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*70 + "\n")

    # Header
    print(f"{'Model':<20} {'SNT mAP':>10} {'SNT #sp':>10} {'IUCN mAP':>10} {'IUCN #sp':>10}")
    print("-"*70)

    all_results = []

    # Iterate through experiment directories
    for exp_dir in sorted(experiments_dir.iterdir()):
        if exp_dir.is_dir():
            results = load_results(exp_dir)
            results['name'] = exp_dir.name
            all_results.append(results)

            snt_str = f"{results['snt_map']:.4f}" if results['snt_map'] is not None else "N/A"
            snt_n = str(results['snt_n_species']) if results['snt_n_species'] is not None else "N/A"
            iucn_str = f"{results['iucn_map']:.4f}" if results['iucn_map'] is not None else "N/A"
            iucn_n = str(results['iucn_n_species']) if results['iucn_n_species'] is not None else "N/A"

            print(f"{exp_dir.name:<20} {snt_str:>10} {snt_n:>10} {iucn_str:>10} {iucn_n:>10}")

    print("-"*70)

    # Find best models
    valid_snt = [r for r in all_results if r['snt_map'] is not None]
    valid_iucn = [r for r in all_results if r['iucn_map'] is not None]

    if valid_snt:
        best_snt = max(valid_snt, key=lambda x: x['snt_map'])
        print(f"\nBest SNT mAP:  {best_snt['name']} ({best_snt['snt_map']:.4f})")

    if valid_iucn:
        best_iucn = max(valid_iucn, key=lambda x: x['iucn_map'])
        print(f"Best IUCN mAP: {best_iucn['name']} ({best_iucn['iucn_map']:.4f})")

    print()

if __name__ == "__main__":
    main()
