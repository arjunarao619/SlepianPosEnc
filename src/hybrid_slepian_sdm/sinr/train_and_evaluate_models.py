import os
import sys
import numpy as np
import torch

import train
import eval


def run_experiment(experiment_name, input_enc, model='ResidualFCNet', sh_L=10,
                   slepian_L_global=10, slepian_L_regional=40, hard_cap=100):
    """Run a single experiment with given parameters."""
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Encoding: {input_enc}, Model: {model}")
    if input_enc == 'sh':
        print(f"SH L={sh_L}")
    elif input_enc == 'slepian':
        print(f"L_global={slepian_L_global}, L_regional={slepian_L_regional}")
    print(f"{'='*60}\n")

    train_params = {
        'experiment_name': experiment_name,
        'log_frequency': 100,
        'species_set': 'all',
        'hard_cap_num_per_class': hard_cap,
        'num_aux_species': 0,
        'loss': 'an_full',
        'input_enc': input_enc,
        'model': model,
        'sh_L': sh_L,
        'slepian_L_global': slepian_L_global,
        'slepian_L_regional': slepian_L_regional,
    }

    train.launch_training_run(train_params)

    for eval_type in ['snt', 'iucn']:
        eval_params = {
            'exp_base': './experiments',
            'experiment_name': experiment_name,
            'eval_type': eval_type,
        }
        if eval_type == 'iucn':
            eval_params['device'] = torch.device('cpu')
        results = eval.launch_eval_run(eval_params)
        np.save(os.path.join('./experiments', experiment_name, f'results_{eval_type}.npy'), results)
        print(f"{eval_type} mAP: {results.mean():.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python train_and_evaluate_models.py <experiment_name> <input_enc> [options]")
        print("  Options: --model, --sh_L, --L_global, --L_regional, --hard_cap")
        print("Examples:")
        print("  python train_and_evaluate_models.py sin_cos sin_cos")
        print("  python train_and_evaluate_models.py sh_10 sh --sh_L 10")
        print("  python train_and_evaluate_models.py slepian_only_l40 slepian --L_global 0 --L_regional 40")
        sys.exit(1)

    experiment_name = sys.argv[1]
    input_enc = sys.argv[2]

    # Parse optional arguments
    kwargs = {}
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == '--model':
            kwargs['model'] = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == '--sh_L':
            kwargs['sh_L'] = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '--L_global':
            kwargs['slepian_L_global'] = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '--L_regional':
            kwargs['slepian_L_regional'] = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '--hard_cap':
            kwargs['hard_cap'] = int(sys.argv[i+1])
            i += 2
        else:
            i += 1

    run_experiment(experiment_name, input_enc, **kwargs)
