import os
import numpy as np
import json
import pandas as pd
from calendar import monthrange
import torch
import utils

class LocationDataset(torch.utils.data.Dataset):
    def __init__(self, locs, labels, classes, class_to_taxa, input_enc, device, sh_L=10, slepian_params=None):

        # handle input encoding:
        self.input_enc = input_enc

        # Load environmental raster if needed
        if self.input_enc in ['env', 'sin_cos_env', 'slepian_env', 'slepian_direct_env']:
            raster = load_env()
        else:
            raster = None

        # Load Slepian raster if needed (for interpolation-based encoding)
        if self.input_enc in ['slepian', 'slepian_env']:
            if slepian_params is None:
                raise ValueError("slepian_params required for Slepian encoding. "
                                 "Pass the full params dict containing slepian_* keys.")
            slepian_raster = load_slepian(slepian_params)
        else:
            slepian_raster = None

        # Load Slepian coefficients if needed (for direct computation)
        if self.input_enc in ['slepian_direct', 'slepian_direct_env']:
            if slepian_params is None:
                raise ValueError("slepian_params required for Slepian direct encoding. "
                                 "Pass the full params dict containing slepian_* keys.")
            slepian_data = load_slepian_direct(slepian_params)
        else:
            slepian_data = None

        # Create encoder with both rasters and/or direct slepian data
        self.enc = utils.CoordEncoder(input_enc, raster=raster, L=sh_L,
                                       slepian_raster=slepian_raster, slepian_data=slepian_data)

        # define some properties:
        self.locs = locs
        self.loc_feats = self.enc.encode(self.locs, normalize=True)
        self.labels = labels
        self.classes = classes
        self.class_to_taxa = class_to_taxa

        # useful numbers:
        self.num_classes = len(np.unique(labels))
        self.input_dim = self.loc_feats.shape[1]

        # Move rasters to device
        if self.enc.raster is not None:
            self.enc.raster = self.enc.raster.to(device)
        if self.enc.slepian_raster is not None:
            self.enc.slepian_raster = self.enc.slepian_raster.to(device)

    def __len__(self):
        return self.loc_feats.shape[0]

    def __getitem__(self, index):
        loc_feat  = self.loc_feats[index, :]
        loc       = self.locs[index, :]
        class_id  = self.labels[index]
        return loc_feat, loc, class_id

def load_env():
    with open('paths.json', 'r') as f:
        paths = json.load(f)
    raster = load_context_feats(os.path.join(paths['env'],'bioclim_elevation_scaled.npy'))
    return raster


def load_slepian(params):
    """
    Load or compute Slepian feature raster.

    Handles caching: if cache exists, loads from disk; otherwise computes
    from scratch (expensive, 10-30 minutes) and caches for future use.

    Args:
        params: Dict with slepian parameters:
            - slepian_L_global: Global SH max degree (default 10)
            - slepian_L_regional: Regional Slepian max degree (default 80)
            - slepian_lambda_thresh: Eigenvalue threshold (default 0.1)
            - slepian_grid_resolution: Grid resolution in degrees (default 0.5)
            - slepian_cache_dir: Cache directory (default './cache/slepian')
            - slepian_caps: List of cap definitions

    Returns:
        torch.Tensor: Slepian feature raster [H, W, C]
    """
    cache_dir = params.get('slepian_cache_dir', './cache/slepian')
    os.makedirs(cache_dir, exist_ok=True)

    # Generate cache path based on configuration
    cache_path = utils.get_slepian_cache_path(params, cache_dir)

    if os.path.exists(cache_path):
        # Fast path: load from cache
        cache_data = utils.load_slepian_raster(cache_path, verbose=True)
        return cache_data['raster']
    else:
        # Slow path: compute from scratch
        print("\n" + "="*70)
        print("Slepian raster not found in cache. Computing from scratch...")
        print("This is a one-time computation that will be cached for future use.")
        print("="*70)

        result = utils.compute_slepian_raster(
            L_global=params.get('slepian_L_global', 10),
            L_regional=params.get('slepian_L_regional', 80),
            caps=params.get('slepian_caps', None),
            lambda_thresh=params.get('slepian_lambda_thresh', 0.1),
            grid_resolution=params.get('slepian_grid_resolution', 0.5),
            cache_path=cache_path,
            verbose=True
        )
        return result['raster']


def load_slepian_direct(params):
    """
    Load or compute Slepian coefficients for direct feature computation.

    This stores only the Slepian SH coefficients (much smaller than full raster),
    allowing exact feature computation at any location without interpolation.

    Args:
        params: Dict with slepian parameters

    Returns:
        Dict with 'shcoeffs_list' (reconstructed SHCoeffs objects) and 'metadata'
    """
    try:
        import pyshtools as pysh
    except ImportError:
        raise ImportError("pyshtools required for direct Slepian computation")

    cache_dir = params.get('slepian_cache_dir', './cache/slepian')
    os.makedirs(cache_dir, exist_ok=True)

    # Generate cache path for direct coefficients
    cache_path = utils.get_slepian_direct_cache_path(params, cache_dir)

    if os.path.exists(cache_path):
        # Fast path: load from cache
        print(f"Loading Slepian coefficients from {cache_path}")
        slepian_data = torch.load(cache_path, map_location='cpu', weights_only=False)
    else:
        # Slow path: compute from scratch
        print("\n" + "="*70)
        print("Slepian coefficients not found in cache. Computing from scratch...")
        print("This is a one-time computation that will be cached for future use.")
        print("="*70)

        slepian_data = utils.compute_slepian_coefficients(
            L_global=params.get('slepian_L_global', 10),
            L_regional=params.get('slepian_L_regional', 80),
            caps=params.get('slepian_caps', None),
            lambda_thresh=params.get('slepian_lambda_thresh', 0.1),
            cache_path=cache_path,
            verbose=True
        )

    # Reconstruct SHCoeffs objects from stored coefficient arrays
    meta = slepian_data.get('metadata', {})
    coefficients = slepian_data.get('coefficients', [])

    print(f"  L_global={meta.get('L_global', '?')}, "
          f"L_regional={meta.get('L_regional', '?')}, "
          f"total_dim={meta.get('total_dim', '?')}")
    print("  Reconstructing SHCoeffs objects...")

    shcoeffs_list = []
    for cap_coeffs in coefficients:
        cap_shcoeffs = []
        for coeff_array in cap_coeffs:
            # Reconstruct SHCoeffs from stored array
            # coeff_array shape: (2, L+1, L+1)
            shcoeffs = pysh.SHCoeffs.from_array(coeff_array, normalization='ortho')
            cap_shcoeffs.append(shcoeffs)
        shcoeffs_list.append(cap_shcoeffs)

    # Return data with reconstructed SHCoeffs
    return {
        'shcoeffs_list': shcoeffs_list,
        'metadata': meta
    }


def load_context_feats(data_path):
    context_feats = np.load(data_path).astype(np.float32)
    context_feats = torch.from_numpy(context_feats)
    return context_feats

def load_inat_data(ip_file, taxa_of_interest=None):

    print('\nLoading  ' + ip_file)
    data = pd.read_csv(ip_file)

    # remove outliers
    num_obs = data.shape[0]
    data = data[((data['latitude'] <= 90) & (data['latitude'] >= -90) & (data['longitude'] <= 180) & (data['longitude'] >= -180) )]
    if (num_obs - data.shape[0]) > 0:
        print(num_obs - data.shape[0], 'items filtered due to invalid locations')

    if 'accuracy' in data.columns:
        data.drop(['accuracy'], axis=1, inplace=True)

    if 'positional_accuracy' in data.columns:
        data.drop(['positional_accuracy'], axis=1, inplace=True)

    if 'geoprivacy' in data.columns:
        data.drop(['geoprivacy'], axis=1, inplace=True)

    if 'observed_on' in data.columns:
        data.rename(columns = {'observed_on':'date'}, inplace=True)

    num_obs_orig = data.shape[0]
    data = data.dropna()
    size_diff = num_obs_orig - data.shape[0]
    if size_diff > 0:
        print(size_diff, 'observation(s) with a NaN entry out of' , num_obs_orig, 'removed')

    # keep only taxa of interest:
    if taxa_of_interest is not None:
        num_obs_orig = data.shape[0]
        data = data[data['taxon_id'].isin(taxa_of_interest)]
        print(num_obs_orig - data.shape[0], 'observation(s) out of' , num_obs_orig, 'from different taxa removed')

    print('Number of unique classes {}'.format(np.unique(data['taxon_id'].values).shape[0]))

    locs = np.vstack((data['longitude'].values, data['latitude'].values)).T.astype(np.float32)
    taxa = data['taxon_id'].values.astype(np.int64)

    if 'user_id' in data.columns:
        users = data['user_id'].values.astype(np.int64)
        _, users = np.unique(users, return_inverse=True)
    elif 'observer_id' in data.columns:
        users = data['observer_id'].values.astype(np.int64)
        _, users = np.unique(users, return_inverse=True)
    else:
        users = np.ones(taxa.shape[0], dtype=np.int64)*-1

    # Note - assumes that dates are in format YYYY-MM-DD
    years  = np.array([int(d_str[:4])   for d_str in data['date'].values])
    months = np.array([int(d_str[5:7])  for d_str in data['date'].values])
    days   = np.array([int(d_str[8:10]) for d_str in data['date'].values])
    days_per_month = np.cumsum([0] + [monthrange(2018, mm)[1] for mm in range(1, 12)])
    dates  = days_per_month[months-1] + days-1
    dates  = np.round((dates) / 364.0, 4).astype(np.float32)
    if 'id' in data.columns:
        obs_ids = data['id'].values
    elif 'observation_uuid' in data.columns:
        obs_ids = data['observation_uuid'].values

    return locs, taxa, users, dates, years, obs_ids

def choose_aux_species(current_species, num_aux_species, aux_species_seed, taxa_file):
    if num_aux_species == 0:
        return []
    with open('paths.json', 'r') as f:
        paths = json.load(f)
    data_dir = paths['train']
    taxa_file = os.path.join(data_dir, taxa_file)
    with open(taxa_file, 'r') as f:
        inat_large_metadata = json.load(f)
    aux_species_candidates = [x['taxon_id'] for x in inat_large_metadata]
    aux_species_candidates = np.setdiff1d(aux_species_candidates, current_species)
    print(f'choosing {num_aux_species} species to add from {len(aux_species_candidates)} candidates')
    rng = np.random.default_rng(aux_species_seed)
    idx_rand_aux_species = rng.permutation(len(aux_species_candidates))
    aux_species = list(aux_species_candidates[idx_rand_aux_species[:num_aux_species]])
    return aux_species

def get_taxa_of_interest(species_set='all', num_aux_species=0, aux_species_seed=123, taxa_file=None, taxa_file_snt=None):
    if species_set == 'all':
        return None
    if species_set == 'snt_birds':
        assert taxa_file_snt is not None
        with open(taxa_file_snt, 'r') as f: #
            taxa_subsets = json.load(f)
        taxa_of_interest = list(taxa_subsets['snt_birds'])
    else:
        raise NotImplementedError
    # optionally add some other species back in:
    aux_species = choose_aux_species(taxa_of_interest, num_aux_species, aux_species_seed, taxa_file)
    taxa_of_interest.extend(aux_species)
    return taxa_of_interest

def get_idx_subsample_observations(labels, hard_cap=-1, hard_cap_seed=123):
    if hard_cap == -1:
        return np.arange(len(labels))
    print(f'subsampling (up to) {hard_cap} per class for the training set')
    class_counts = {id: 0 for id in np.unique(labels)}
    ss_rng = np.random.default_rng(hard_cap_seed)
    idx_rand = ss_rng.permutation(len(labels))
    idx_ss = []
    for i in idx_rand:
        class_id = labels[i]
        if class_counts[class_id] < hard_cap:
            idx_ss.append(i)
            class_counts[class_id] += 1
    idx_ss = np.sort(idx_ss)
    print(f'final training set size: {len(idx_ss)}')
    return idx_ss

def get_train_data(params):
    with open('paths.json', 'r') as f:
        paths = json.load(f)
    data_dir = paths['train']
    obs_file  = os.path.join(data_dir, params['obs_file'])
    taxa_file = os.path.join(data_dir, params['taxa_file'])
    taxa_file_snt = os.path.join(data_dir, 'taxa_subsets.json')

    taxa_of_interest = get_taxa_of_interest(params['species_set'], params['num_aux_species'], params['aux_species_seed'], params['taxa_file'], taxa_file_snt)

    locs, labels, _, _, _, _ = load_inat_data(obs_file, taxa_of_interest)
    unique_taxa, class_ids = np.unique(labels, return_inverse=True)
    class_to_taxa = unique_taxa.tolist()

    # load class names
    class_info_file = json.load(open(taxa_file, 'r'))
    class_names_file = [cc['latin_name'] for cc in class_info_file]
    taxa_ids_file = [cc['taxon_id'] for cc in class_info_file]
    classes = dict(zip(taxa_ids_file, class_names_file))

    idx_ss = get_idx_subsample_observations(labels, params['hard_cap_num_per_class'], params['hard_cap_seed'])

    locs = torch.from_numpy(np.array(locs)[idx_ss]) # convert to Tensor

    labels = torch.from_numpy(np.array(class_ids)[idx_ss])

    # Get sh_L parameter (default to 10 for backward compatibility)
    sh_L = params.get('sh_L', 10)

    # Pass slepian_params if using any slepian encoding (raster or direct)
    slepian_params = params if params['input_enc'] in ['slepian', 'slepian_env', 'slepian_direct', 'slepian_direct_env'] else None

    ds = LocationDataset(locs, labels, classes, class_to_taxa, params['input_enc'], params['device'],
                         sh_L=sh_L, slepian_params=slepian_params)

    return ds
