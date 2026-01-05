import os
import numpy as np
import torch

import train
import eval

train_params = {}

train_params['experiment_name'] = 'slepian_l40_sh10_res' # This will be the name of the directory where results for this run are saved.
train_params['log_frequency'] = 100
'''
species_set
- Which set of species to train on.
- Valid values: 'all', 'snt_birds'
'''
train_params['species_set'] = 'all'

'''
hard_cap_num_per_class
- Maximum number of examples per class to use for training.
- Valid values: positive integers or -1 (indicating no cap).
'''
train_params['hard_cap_num_per_class'] = 100

'''
num_aux_species
- Number of random additional species to add.
- Valid values: Nonnegative integers. Should be zero if params['species_set'] == 'all'.
'''
train_params['num_aux_species'] = 0

'''
input_enc
- Type of inputs to use for training.
- Valid values: 'sin_cos', 'env', 'sin_cos_env', 'sh'
'''
train_params['input_enc'] = 'slepian'  # Change to 'sh' for spherical harmonics
train_params['slepian_L_regional'] = 40
train_params['model'] = 'ResidualFCNet'
'''
sh_L
- Maximum degree for spherical harmonics encoding (only used when input_enc='sh')
- Output dimension = L^2 (e.g., L=10 -> 100 features)
'''
train_params['sh_L'] = 10

'''
loss
- Which loss to use for training.
- Valid values: 'an_full', 'an_slds', 'an_ssdl', 'an_full_me', 'an_slds_me', 'an_ssdl_me'
'''
train_params['loss'] = 'an_full'

# train:
train.launch_training_run(train_params)

# evaluate:
for eval_type in ['snt', 'iucn']:
    eval_params = {}
    eval_params['exp_base'] = './experiments'
    eval_params['experiment_name'] = train_params['experiment_name']
    eval_params['eval_type'] = eval_type
    if eval_type == 'iucn':
        eval_params['device'] = torch.device('cpu') # for memory reasons
    cur_results = eval.launch_eval_run(eval_params)
    np.save(os.path.join(eval_params['exp_base'], train_params['experiment_name'], f'results_{eval_type}.npy'), cur_results)

'''
Note that train_params and eval_params do not contain all of the parameters of interest. Instead,
there are default parameter sets for training and evaluation (which can be found in setup.py).
In this script we create dictionaries of key-value pairs that are used to override the defaults
as needed.
'''
