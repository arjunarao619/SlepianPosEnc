# Localized, High Resolution Geographic Representations with Slepian Functions

<p align="center">
  <img src="visualizations/README_gif.gif" style="display: block; margin: 0 auto; width: 80%;">
</p>
<p align="center"> <sup>Slepian functions concentrate representational capacity where it matters. Left: Global spherical harmonics (L=10) spread energy uniformly across the sphere. Right: Our hybrid encoder progressively adds Slepian modes concentrated on a region of interest (India), building fine-grained local detail while preserving global context.<sup></p>

## Requirements

**Core (Part 1 & 2)**
```
python >= 3.9
torch >= 2.0
numpy
scipy
pandas
matplotlib
scikit-learn
tqdm
pyshtools          # for Slepian functions
```

**Additional for Part 2 (Image-Augmented)**
```
earthengine-api    # Google Earth Engine
seaborn
wandb
geopandas          # optional, for spatial data
shapely            # optional
cartopy            # optional, for map visualization
```

## Project Structure

```
SlepianPosEnc/
├── src/
│   ├── nn/                              # Neural network architectures
│   ├── pe_baselines/                    # Baseline positional encoders
│   ├── spherical_harmonics_ylm.py       # Spherical harmonics implementation
│   ├── supervised_geographic_classification/
│   │   ├── scripts/                     # Part 1 experiment runners
│   │   └── *.py                         # Training scripts
│   └── slepian_image_augmentation/
│       ├── scripts/                     # Part 2 experiment runners
│       └── *.py                         # Dataset creation & training
├── results/                             # Outputs (created on run)
├── cache/                               # Feature cache (created on run)
└── data/                                # Datasets (created on run)
```

---

## Part 1: Supervised Geographic Prediction

Three geographic prediction tasks comparing Slepian encodings against baseline positional encoders.

| Task | Type | Samples | Metric |
|------|------|---------|--------|
| California Housing | Regression | 20,640 | R-squared |
| Japan Prefectures | Classification (47 classes) | 4,700 | Accuracy |
| Arctic MSS | Regression | ~500,000 | R-squared |

### Dataset Preparation

**California Housing**

No preparation needed. Data is automatically downloaded from scikit-learn on first run.

**Japan Prefectures**

No preparation needed. The dataset is synthetically generated on first run using prefecture centroid coordinates.

**Arctic Mean Sea Surface (MSS)**

Download the preprocessed Arctic MSS data from:
https://drive.google.com/drive/folders/17rwMtEc5vwRKEjNolreUBL2Yk4OSTvr4?usp=sharing

This dataset is from Chen et al., "Deep Random Features for Scalable Interpolation of Spatiotemporal Data" (arXiv:2412.11350, 2024).

After downloading, set the environment variable before running experiments:
```bash
export MSS_DATA_PATH=/path/to/downloaded/mss/data
```

### Running Experiments

All commands are run from the project root directory.

**California Housing**
```bash
cd src/supervised_geographic_classification/scripts
bash run_california_experiments.sh    # Slepian + Vanilla SH
bash run_california_baselines.sh      # Baseline encoders
```

**Japan Prefectures**
```bash
cd src/supervised_geographic_classification/scripts
bash run_japan_experiments.sh         # Slepian + Vanilla SH (generates data on first run)
bash run_japan_baselines.sh           # Baseline encoders
```

**Arctic MSS**
```bash
export MSS_DATA_PATH=/path/to/mss/data
cd src/supervised_geographic_classification/scripts
bash run_mss_experiments.sh           # Slepian + Vanilla SH
bash run_mss_baselines.sh             # Baseline encoders
```

### Results

Aggregated results are saved to:
```
results/
├── california/aggregated_results.csv
├── japan/aggregated_results.csv
└── mss/aggregated_results.csv
```

Figures are saved to `results/{task}/figures/`.

### Configuration

Training parameters can be modified in the bash scripts:

| Parameter | California | Japan | MSS |
|-----------|------------|-------|-----|
| Architectures | mlp, resmlp, siren, glu | mlp, resmlp, siren, glu | mlp, resmlp, siren, glu |
| Epochs | 500 | 200 | 200 |
| Batch size | 512 | 256 | 2048 |
| Runs | 5 | 3 | 5 |

Slepian bandwidths tested: L = 40, 64, 80, 120

---

## Part 2: Image-Augmented Geographic Prediction

Building density prediction using satellite image embeddings (AlphaEarth) combined with Slepian positional encodings.

| Task | Type | Regions | Metric |
|------|------|---------|--------|
| OpenBuildings Density | Regression | Maharashtra, Dhaka, Mexico City | R-squared |

### Dataset Preparation

**Step 1: Download OpenBuildings Data**

Download building footprint CSVs from Google Open Buildings:
https://sites.research.google/open-buildings/

Select your region of interest and download the CSV files.

**Step 2: Create Dataset with AlphaEarth Embeddings**

Requires Google Earth Engine authentication:
```bash
earthengine authenticate
```

Then create the gridded dataset:
```bash
cd src/slepian_image_augmentation
python create_buildings_dataset.py \
    --csvs maharashtra=/path/to/maharashtra.csv dhaka=/path/to/dhaka.csv \
    --output-dir /path/to/output \
    --samples-per-region 30000
```

**Step 3: Add Multi-scale Targets (Optional)**

```bash
python prep_multiscale_targets.py \
    --data-dir /path/to/output \
    --region maharashtra \
    --sigmas-km 0,1,2,5,10,20
```

### Running Experiments

Set the data path and run experiments:
```bash
export BUILDINGS_DATA_PATH=/path/to/buildings/data
cd src/slepian_image_augmentation/scripts
bash run_buildings_experiments.sh
```

### Results

Results are saved to:
```
results/buildings/aggregated_results.csv
```

### Configuration

| Parameter | Value |
|-----------|-------|
| Regions | maharashtra, dhaka, mexicocity |
| Cap radius | 7.5 degrees |
| SH bandwidths | 10, 40 |
| Slepian bandwidths | 40, 64, 96, 128 |
| Seeds | 123, 456, 789, 1011, 1213 |
| Epochs | 120 |
| Batch size | 512 |
