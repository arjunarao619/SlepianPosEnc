"""
Parse and display S&T and IUCN results for all trained models.
"""

import os
import numpy as np
import torch
from pathlib import Path

experiments_dir = Path('experiments')

def load_results(exp_path):
    """Load SNT and IUCN results from an experiment directory."""
    results = {}

    model_path = exp_path / 'model.pt'
    snt_path = exp_path / 'results_snt.npy'
    iucn_path = exp_path / 'results_iucn.npy'

    # Get model info
    results['has_model'] = model_path.exists()
    results['model_type'] = None
    results['input_enc'] = None
    results['input_dim'] = None

    if model_path.exists():
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            params = checkpoint.get('params', {})
            results['model_type'] = params.get('model', None)
            results['input_enc'] = params.get('input_enc', None)
            results['input_dim'] = params.get('input_dim', None)
        except:
            pass

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

    # Determine status
    if results['has_model'] and results['snt_map'] is not None and results['iucn_map'] is not None:
        results['status'] = 'Complete'
    elif results['has_model'] and (results['snt_map'] is not None or results['iucn_map'] is not None):
        results['status'] = 'Partial'
    elif results['has_model']:
        results['status'] = 'Needs Eval'
    else:
        results['status'] = 'Running/Empty'

    return results

def main():
    print("\n" + "="*110)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*110 + "\n")

    all_results = []

    # Iterate through experiment directories
    for exp_dir in sorted(experiments_dir.iterdir()):
        if exp_dir.is_dir():
            results = load_results(exp_dir)
            results['name'] = exp_dir.name
            all_results.append(results)

    # Separate by model type
    linnet_results = [r for r in all_results if r['model_type'] == 'LinNet']
    resnet_results = [r for r in all_results if r['model_type'] == 'ResidualFCNet']
    other_results = [r for r in all_results if r['model_type'] not in ['LinNet', 'ResidualFCNet']]

    def print_table(title, data):
        print(f"\n{title}")
        print("-"*110)
        print(f"{'Experiment':<25} {'Encoding':<15} {'Dim':>6} {'SNT mAP':>10} {'#sp':>6} {'IUCN mAP':>10} {'#sp':>6} {'Status':<12}")
        print("-"*110)

        for r in sorted(data, key=lambda x: x['name']):
            snt_str = f"{r['snt_map']:.4f}" if r['snt_map'] is not None else "-"
            snt_n = str(r['snt_n_species']) if r['snt_n_species'] is not None else "-"
            iucn_str = f"{r['iucn_map']:.4f}" if r['iucn_map'] is not None else "-"
            iucn_n = str(r['iucn_n_species']) if r['iucn_n_species'] is not None else "-"
            enc = r['input_enc'] if r['input_enc'] else "-"
            dim = str(r['input_dim']) if r['input_dim'] else "-"

            print(f"{r['name']:<25} {enc:<15} {dim:>6} {snt_str:>10} {snt_n:>6} {iucn_str:>10} {iucn_n:>6} {r['status']:<12}")

    if linnet_results:
        print_table("LinNet Models (Linear classifier on encoding)", linnet_results)

    if resnet_results:
        print_table("ResidualFCNet Models (Deep network on encoding)", resnet_results)

    if other_results:
        print_table("Other/Incomplete Models", other_results)

    # Find best models
    print("\n" + "="*110)
    print("BEST RESULTS")
    print("="*110)

    valid_snt = [r for r in all_results if r['snt_map'] is not None]
    valid_iucn = [r for r in all_results if r['iucn_map'] is not None]

    if valid_snt:
        best_snt = max(valid_snt, key=lambda x: x['snt_map'])
        print(f"\nBest SNT mAP:  {best_snt['name']} ({best_snt['snt_map']:.4f})")

    if valid_iucn:
        best_iucn = max(valid_iucn, key=lambda x: x['iucn_map'])
        print(f"Best IUCN mAP: {best_iucn['name']} ({best_iucn['iucn_map']:.4f})")

    # Check what's missing for comparison
    print("\n" + "="*110)
    print("MISSING EXPERIMENTS FOR COMPLETE LinNet vs ResNet COMPARISON")
    print("="*110)

    # Get base names
    linnet_bases = set()
    resnet_bases = set()

    for r in linnet_results:
        linnet_bases.add(r['name'])

    for r in resnet_results:
        # Remove _res suffix to get base name
        base = r['name'].replace('_res', '')
        resnet_bases.add(base)

    # Find LinNet experiments without ResNet counterpart
    missing_resnet = []
    for name in linnet_bases:
        if name not in resnet_bases:
            missing_resnet.append(name + "_res")

    # Find ResNet experiments without LinNet counterpart
    missing_linnet = []
    for name in resnet_bases:
        if name not in linnet_bases:
            missing_linnet.append(name)

    if missing_resnet:
        print(f"\nLinNet experiments missing ResNet counterpart ({len(missing_resnet)}):")
        for name in sorted(missing_resnet):
            print(f"  - {name}")
    else:
        print("\nAll LinNet experiments have ResNet counterparts!")

    if missing_linnet:
        print(f"\nResNet experiments missing LinNet counterpart ({len(missing_linnet)}):")
        for name in sorted(missing_linnet):
            print(f"  - {name}")

    # Summary
    print("\n" + "="*110)
    print("STATUS SUMMARY")
    print("="*110)

    complete = [r for r in all_results if r['status'] == 'Complete']
    partial = [r for r in all_results if r['status'] == 'Partial']
    needs_eval = [r for r in all_results if r['status'] == 'Needs Eval']
    running = [r for r in all_results if r['status'] == 'Running/Empty']

    print(f"\nComplete: {len(complete)}")
    print(f"Partial: {len(partial)}")
    print(f"Needs Eval: {len(needs_eval)}")
    print(f"Running/Empty: {len(running)}")

    if running:
        print(f"\nRunning/Empty experiments: {[r['name'] for r in running]}")

    print()

if __name__ == "__main__":
    main()
