"""Core utility functions."""

import torch
import numpy as np
import math
import datetime


def normalize_coords(locs):
    """Normalize lon/lat from degrees to [-1, 1]."""
    locs[:, 0] /= 180.0
    locs[:, 1] /= 90.0
    return locs


def bilinear_interpolate(loc_ip, data, remove_nans_raster=True):
    """Bilinear interpolation from raster. loc_ip: Nx2 in [-1,1], data: HxWxC."""
    assert data is not None

    loc = (loc_ip.clone() + 1) / 2.0
    loc[:, 1] = 1 - loc[:, 1]

    assert not torch.any(torch.isnan(loc))

    if remove_nans_raster:
        data[torch.isnan(data)] = 0.0

    loc[:, 0] *= (data.shape[1] - 1)
    loc[:, 1] *= (data.shape[0] - 1)

    loc_int = torch.floor(loc).long()
    xx = loc_int[:, 0]
    yy = loc_int[:, 1]
    xx_plus = xx + 1
    xx_plus[xx_plus > (data.shape[1] - 1)] = data.shape[1] - 1
    yy_plus = yy + 1
    yy_plus[yy_plus > (data.shape[0] - 1)] = data.shape[0] - 1

    loc_delta = loc - torch.floor(loc)
    dx = loc_delta[:, 0].unsqueeze(1)
    dy = loc_delta[:, 1].unsqueeze(1)

    interp_val = (data[yy, xx, :] * (1 - dx) * (1 - dy) +
                  data[yy, xx_plus, :] * dx * (1 - dy) +
                  data[yy_plus, xx, :] * (1 - dx) * dy +
                  data[yy_plus, xx_plus, :] * dx * dy)

    return interp_val


def rand_samples(batch_size, device, rand_type='uniform'):
    """Random background locations. Returns Nx2 in [-1, 1]."""
    if rand_type == 'spherical':
        rand_loc = torch.rand(batch_size, 2).to(device)
        theta1 = 2.0 * math.pi * rand_loc[:, 0]
        theta2 = torch.acos(2.0 * rand_loc[:, 1] - 1.0)
        lat = 1.0 - 2.0 * theta2 / math.pi
        lon = (theta1 / math.pi) - 1.0
        rand_loc = torch.cat((lon.unsqueeze(1), lat.unsqueeze(1)), 1)
    elif rand_type == 'uniform':
        rand_loc = torch.rand(batch_size, 2).to(device) * 2.0 - 1.0
    return rand_loc


def get_time_stamp():
    """Current timestamp string."""
    cur_time = str(datetime.datetime.now())
    date, time = cur_time.split(' ')
    h, m, s = time.split(':')
    s = s.split('.')[0]
    return '{}-{}-{}-{}'.format(date, h, m, s)


def coord_grid(grid_size, split_ids=None, split_of_interest=None):
    """Generate evenly spaced coordinate grid. Returns Nx2 array."""
    feats = np.zeros((grid_size[0], grid_size[1], 2), dtype=np.float32)
    mg = np.meshgrid(np.linspace(-180, 180, feats.shape[1]),
                     np.linspace(90, -90, feats.shape[0]))
    feats[:, :, 0] = mg[0]
    feats[:, :, 1] = mg[1]
    if split_ids is None or split_of_interest is None:
        return feats.reshape(feats.shape[0] * feats.shape[1], 2)
    else:
        ind_y, ind_x = np.where(split_ids == split_of_interest)
        return feats[ind_y, ind_x, :]


def create_spatial_split(raster, mask, train_amt=1.0, cell_size=25):
    """Checkerboard train/test split. Returns 0=invalid, 1=train, 2=test."""
    split_ids = np.ones((raster.shape[0], raster.shape[1]))
    start = cell_size
    for ii in np.arange(0, split_ids.shape[0], cell_size):
        if start == 0:
            start = cell_size
        else:
            start = 0
        for jj in np.arange(start, split_ids.shape[1], cell_size * 2):
            split_ids[ii:ii + cell_size, jj:jj + cell_size] = 2
    split_ids = split_ids * mask
    if train_amt < 1.0:
        tr_y, tr_x = np.where(split_ids == 1)
        inds = np.random.choice(len(tr_y), int(len(tr_y) * (1.0 - train_amt)), replace=False)
        split_ids[tr_y[inds], tr_x[inds]] = 0
    return split_ids


def average_precision_score_faster(y_true, y_scores):
    """Fast AP score, drop-in replacement for sklearn."""
    num_positives = y_true.sum()
    inds = np.argsort(y_scores)[::-1]
    y_true_s = y_true[inds]

    false_pos_c = np.cumsum(1.0 - y_true_s)
    true_pos_c = np.cumsum(y_true_s)
    recall = true_pos_c / num_positives
    false_neg = np.maximum(true_pos_c + false_pos_c, np.finfo(np.float32).eps)
    precision = true_pos_c / false_neg

    recall_e = np.hstack((0, recall, 1))
    recall_e = (recall_e[1:] - recall_e[:-1])[:-1]
    return (recall_e * precision).sum()
