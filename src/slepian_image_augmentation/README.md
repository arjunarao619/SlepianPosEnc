# Slepian Image Augmentation

Compares Slepian vs Spherical Harmonic positional encodings for augmenting satellite image embeddings in building density regression.

## Task

**Goal**: Predict building density from satellite imagery at multiple spatial scales.

**Approach**: Augment image embeddings (AlphaEarth or Galileo) with positional encodings, then train a regression model. We compare:
- **Baseline**: Image embeddings only
- **+SH**: Image embeddings + Spherical Harmonics
- **+Slepian**: Image embeddings + Slepian functions (region-concentrated)

**Dataset**: Google Open Buildings footprints + Sentinel-2 imagery across 4 regions (South Florida, Dhaka, Maharashtra, Mexico City).

## Setup

### 1. Download Open Buildings Data

1. Go to https://sites.research.google/gr/open-buildings/
2. Download CSV files for your regions of interest
3. Name files by region: `southflorida.csv`, `dhaka.csv`, `maharashtra.csv`, `mexicocity.csv`

### 2. Configure Earth Engine

Edit `scripts/prepare_data.sh` to set your Earth Engine project ID:
```bash
EE_PROJECT="your-ee-project"  # Your Earth Engine project ID
```

## Usage

### Step 1: Prepare Data
```bash
./scripts/prepare_data.sh <output_dir> <csv_dir>
```

**Arguments:**
- `<output_dir>`: Where to save processed data (parquets, embeddings, etc.)
- `<csv_dir>`: Directory containing your Open Buildings CSVs

**Example:**

If your CSVs are organized as:
```
/data/openbuildings_csvs/
├── southflorida.csv
├── dhaka.csv
├── maharashtra.csv
└── mexicocity.csv
```

Run:
```bash
./scripts/prepare_data.sh /scratch/openbuildings /data/openbuildings_csvs
```

The script expects CSV filenames to match the `REGIONS` variable in the script (e.g., `REGIONS="southflorida dhaka"` expects `southflorida.csv` and `dhaka.csv`).

This runs:
1. Create grid + extract AlphaEarth embeddings (Earth Engine)
2. Download Sentinel-2 patches
3. Extract Galileo embeddings
4. Merge embeddings
5. Compute multi-scale targets (σ = 0, 1, 2, 3, 4, 5, 10, 20, 40 km)

### Step 2: Run Experiments
```bash
./scripts/run_parallel_experiment.sh <data_dir>
```

Example:
```bash
./scripts/run_parallel_experiment.sh /scratch/openbuildings
```

This runs 48 experiments in parallel:
- 4 regions × 2 embeddings × 2 L_SH × 3 L_Slepian

## Output

Results saved to `<data_dir>/results/`:
- `*.csv` - Per-experiment metrics (R², RMSE per smoothing scale)

CSV columns:
| Column | Description |
|--------|-------------|
| `R2_ALPHAEARTH` / `R2_GALILEO` | Baseline (embedding only) |
| `R2_SH` | With Spherical Harmonics |
| `R2_Slepian` | With Slepian functions |
| `ΔR2_Slepian` | Improvement over baseline |
| `sigma_km` | Smoothing scale |

## File Structure

```
slepian_image_augmentation/
├── scripts/
│   ├── prepare_data.sh              # Data pipeline
│   ├── run_parallel_experiment.sh   # Training pipeline
│   └── precompute_design_matrices.py
├── create_buildings_dataset.py
├── sentinel2_openbuildings_download.py
├── generate_galileo_all_regions.py
├── merge_galileo_embeddings.py
├── prep_multiscale_targets.py
├── train_slepian_vs_sh_multiscale_v3.py
└── galileo/
```

## Customization

Edit `scripts/run_parallel_experiment.sh` to change:
```bash
REGIONS="..."           # Regions to process
L_SH_VALUES="10 40"     # SH maximum degrees
L_SLEPIAN_VALUES="40 80 120"  # Slepian maximum degrees
SEED=123                # Random seed
MAX_PARALLEL_JOBS=30    # Parallelism
```
