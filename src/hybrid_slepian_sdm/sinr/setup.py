import copy
import torch

def apply_overrides(params, overrides):
    params = copy.deepcopy(params)
    for param_name in overrides:
        if param_name not in params:
            print(f'override failed: no parameter named {param_name}')
            raise ValueError
        params[param_name] = overrides[param_name]
    return params

def get_default_params_train(overrides={}):

    params = {}

    '''
    misc
    '''
    params['device'] = 'cuda' # cuda, cpu
    params['save_base'] = './experiments/'
    params['experiment_name'] = 'demo'
    params['timestamp'] = False

    '''
    data
    '''
    params['species_set'] = 'all' # all, snt_birds
    params['hard_cap_seed'] = 9472
    params['hard_cap_num_per_class'] = -1 # -1 for no hard capping
    params['aux_species_seed'] = 8099
    params['num_aux_species'] = 0 # for snt_birds case, how many other species to add in

    '''
    data files
    '''
    params['obs_file'] = 'geo_prior_train.csv'
    params['taxa_file'] = 'geo_prior_train_meta.json'

    '''
    model
    '''
    params['model'] = 'ResidualFCNet'  # ResidualFCNet, LinNet
    params['num_filts'] = 256  # embedding dimension
    params['input_enc'] = 'sin_cos'  # sin_cos, env, sin_cos_env, sh, slepian, slepian_env
    params['depth'] = 4
    params['sh_L'] = 10  # Max degree for SH encoding (output dim = L^2 = 100)

    '''
    slepian encoding (only used when input_enc='slepian' or 'slepian_env')
    '''
    params['slepian_L_global'] = 10       # Global SH baseline (100 features)
    params['slepian_L_regional'] = 80     # Regional Slepian max degree
    params['slepian_lambda_thresh'] = 0.1 # Eigenvalue threshold for mode selection
    params['slepian_grid_resolution'] = 0.5  # Grid resolution in degrees
    params['slepian_cache_dir'] = './cache/slepian'
    params['slepian_caps'] = [
        {'name': 'us', 'center': (-95.0, 40.0), 'radius': 25.0},
        {'name': 'europe', 'center': (10.0, 50.0), 'radius': 20.0},
    ]

    '''
    loss
    '''
    params['loss'] = 'an_full' # an_full, an_ssdl, an_slds
    params['pos_weight'] = 2048

    '''
    optimization
    '''
    params['batch_size'] = 2048
    params['lr'] = 0.0005
    params['lr_decay'] = 0.98
    params['num_epochs'] = 10

    '''
    saving
    '''
    params['log_frequency'] = 512

    params = apply_overrides(params, overrides)

    return params

def get_default_params_eval(overrides={}):

    params = {}

    '''
    misc
    '''
    params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['seed'] = 2022
    params['exp_base'] = './experiments'
    params['ckp_name'] = 'model.pt'
    params['eval_type'] = 'snt' # snt, iucn, geo_prior, geo_feature
    params['experiment_name'] = 'demo'

    '''
    geo prior
    '''
    params['batch_size'] = 2048

    '''
    geo feature
    '''
    params['cell_size'] = 25

    params = apply_overrides(params, overrides)

    return params
