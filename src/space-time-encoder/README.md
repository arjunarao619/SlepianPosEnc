# Space-Time Encoder for ACE Temperature Prediction

Spherical harmonics (spatial) + temporal encodings for atmospheric temperature regression on the ACE dataset.

## Requirements

```
torch>=2.0
numpy
scipy
netCDF4
```

```bash
pip install torch numpy scipy netCDF4
```

## Data Format

Set the data path:
```bash
export ACE_DATA_PATH="/path/to/ace/"  # default: /scratch/local/arra4944_images/ace/
```

Expected files (12 monthly NetCDF files for year 2021):
```
2021010100.nc  2021020100.nc  ...  2021120100.nc
```

Each file must contain:
| Variable | Description |
|----------|-------------|
| `grid_xt` | Longitude (0-360, auto-shifted to -180 to 180) |
| `grid_yt` | Latitude |
| `time` | Time coordinate |
| `air_temperature_0` | Temperature at ~26 hPa |
| `air_temperature_1` | Temperature at ~99 hPa |
| `air_temperature_2` | Temperature at ~203 hPa |
| `air_temperature_3` | Temperature at ~337 hPa |
| `air_temperature_4` | Temperature at ~504 hPa |
| `air_temperature_5` | Temperature at ~690 hPa |
| `air_temperature_6` | Temperature at ~850 hPa |
| `air_temperature_7` | Temperature at ~964 hPa |

## Training Baselines

Single run:
```bash
python train_baselines.py --temporal_type legendre --seed 42
```

Temporal types: `no_time`, `time_copy`, `triangle`, `monomial`, `legendre`, `fourier`

Full sweep (6 types x 3 seeds = 18 experiments):
```bash
./run_baselines.sh
```

Checkpoints saved to `checkpoints_baselines/`.

## Training DPSS Encoder

Single run:
```bash
python train_dpss_sweep.py --nw 20 --seed 42 --optimized
```

| Argument | Description |
|----------|-------------|
| `--nw` | Time-bandwidth product (e.g., 10, 15, 20, 25, 30, 35) |
| `--k` | Number of DPSS sequences (default: 2*NW-1) |
| `--optimized` | Use learnable projection layer |

Full sweep (6 NW values x 3 seeds = 18 experiments):
```bash
./run_dpss.sh
```

Checkpoints saved to `checkpoints/`.

## Evaluation

Evaluate all checkpoints:
```bash
python test_ace.py
```

Outputs `test_results.csv` with per-variable RMSE and MAE for all 8 temperature levels.

## Model Architecture

```
Input: [lon, lat, time]
    |
    +-- Spatial: Spherical Harmonics (L=20) --> 400 dims --> MLP --> 512 dims
    |
    +-- Temporal: DPSS/Legendre/Fourier/etc. --> 40 dims --> MLP --> 512 dims
    |
    +-- Concatenate --> 1024 dims
    |
    +-- Head: 4-layer FCNet (1024 hidden) --> 8 temperature predictions
```
