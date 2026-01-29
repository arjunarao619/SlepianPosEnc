# Supervised Geographic Classification

Compares Slepian vs Spherical Harmonic positional encodings on geographic prediction tasks.

## Tasks

| Task | Type | Samples | Metric | Data Source |
|------|------|---------|--------|-------------|
| California Housing | Regression | 20,640 | R-squared | scikit-learn (auto-download) |
| Japan Prefectures | Classification (47 classes) | 4,700 | Accuracy | Generated from centroids |
| Arctic MSS | Regression | ~500,000 | R-squared | External download |

## Setup

### California Housing & Japan Prefectures

No setup required. Data is automatically downloaded/generated on first run.

### Arctic Mean Sea Surface (MSS)

Download preprocessed data from [Google Drive](https://drive.google.com/drive/folders/17rwMtEc5vwRKEjNolreUBL2Yk4OSTvr4?usp=sharing) (from Chen et al., arXiv:2412.11350).

```bash
export MSS_DATA_PATH=/path/to/downloaded/mss/data
```

## Usage

Run from the `scripts/` directory:

```bash
cd scripts/

# California Housing
bash run_california_experiments.sh    # Slepian + Vanilla SH
bash run_california_baselines.sh      # Baseline encoders

# Japan Prefectures
bash run_japan_experiments.sh         # Slepian + Vanilla SH
bash run_japan_baselines.sh           # Baseline encoders

# Arctic MSS (requires MSS_DATA_PATH)
bash run_mss_experiments.sh           # Slepian + Vanilla SH
bash run_mss_baselines.sh             # Baseline encoders

# Gaussian Process baselines
bash run_all_gp_baselines.sh

# Resolution sweeps
bash run_resolution_sweep_cali.sh
bash run_japan_resolution_sweep.sh
bash run_mss_resolution_sweep.sh
```

## Output

Results are saved to `results/{task}/`:
- `aggregated_results.csv` - Metrics across runs
- `figures/` - Visualizations

## File Structure

```
supervised_geographic_classification/
├── scripts/                          # Experiment runners
│   ├── run_{task}_experiments.sh     # Slepian experiments
│   ├── run_{task}_baselines.sh       # Baseline comparisons
│   ├── run_{task}_gp_baselines.sh    # GP baselines
│   └── run_{task}_resolution_sweep.sh
├── utils/                            # Shared utilities
│   ├── training.py                   # train_model, evaluate_model
│   ├── geo.py                        # Geographic utilities
│   ├── data.py                       # Data loading
│   └── configs.py                    # Encoder configurations
├── cached_slepian_demo_cali.py       # California Slepian (cached features)
├── japan_prefecture_slepian.py       # Japan Slepian (cached features)
├── train_mss_slepian.py              # MSS Slepian (cached features)
├── train_california_sh_vanilla.py    # California vanilla SH
├── train_mss_sh_vanilla.py           # MSS vanilla SH
├── train_mss_baselines.py            # MSS baseline encoders
├── run_baselines_californiahousing.py # California baseline encoders
├── run_baselines_japanprefecture.py  # Japan baseline encoders
└── train_gp_baselines.py             # Gaussian Process baselines
```

## Configuration

Training parameters in shell scripts:

| Parameter | California | Japan | MSS |
|-----------|------------|-------|-----|
| Architectures | mlp, resmlp, siren, glu | mlp, resmlp, siren, glu | mlp, resmlp, siren, glu |
| Epochs | 500 | 200 | 200 |
| Batch size | 512 | 256 | 2048 |
| Runs | 5 | 3 | 5 |

Slepian bandwidths tested: L = 40, 64, 80, 120

## Baseline Encoders

From `pe_baselines/`:
- **Direct**: Raw (lat, lon) coordinates
- **Cartesian3D**: 3D unit sphere embedding
- **Wrap**: Sinusoidal wrapping
- **Grid**: Learnable spatial grid
- **SphereC/M/+**: Mac Aodha et al. sphere encodings
- **Theory**: Theoretical positional encoding
- **Wavelets**: Spherical wavelets
