# Slepian Positional Encodings for Geographic Machine Learning

<p align="center">
  <img src="visualizations/README_gif.gif" style="display: block; margin: 0 auto; width: 80%;">
</p>
<p align="center"><sup>Slepian functions concentrate representational capacity where it matters. Left: Global spherical harmonics (L=10) spread energy uniformly. Right: Our hybrid encoder adds Slepian modes concentrated on a region of interest (India), building fine-grained local detail while preserving global context.</sup></p>

## Overview

This repository implements **Hybrid Slepian positional encodings** for geographic machine learning tasks. Slepian functions are bandlimited spherical harmonics that concentrate energy within a specified region, enabling high-resolution local representations without sacrificing global context.

## Project Structure

```
SlepianPosEnc/
├── src/
│   ├── nn/                              # Neural network architectures (MLP, SIREN, ResMLP, GLU)
│   ├── pe_baselines/                    # Baseline positional encoders
│   ├── spherical_harmonics_ylm.py       # Spherical harmonics implementation
│   ├── supervised_geographic_classification/  # Geographic prediction experiments
│   └── slepian_image_augmentation/           # Image-augmented prediction
├── results/                             # Experiment outputs
├── cache/                               # Precomputed Slepian features
└── visualizations/                      # Figures
```

## Experiments

### 1. Supervised Geographic Prediction

Three tasks comparing Slepian vs baseline positional encoders:
- **California Housing** - Property value regression
- **Japan Prefectures** - 47-class geographic classification
- **Arctic MSS** - Sea surface height regression

See [`src/supervised_geographic_classification/README.md`](src/supervised_geographic_classification/README.md) for details.

### 2. Image-Augmented Geographic Prediction

Building density regression combining satellite image embeddings with Slepian encodings across 4 global regions.

See [`src/slepian_image_augmentation/README.md`](src/slepian_image_augmentation/README.md) for details.

## Requirements

```
python >= 3.9
torch >= 2.0
numpy, scipy, pandas
matplotlib, scikit-learn
pyshtools              # Slepian function computation
```

Additional for image augmentation:
```
earthengine-api        # Google Earth Engine
geopandas, cartopy     # Geospatial visualization
```

## Quick Start

```bash
# Clone with submodules
git clone --recursive https://github.com/your-repo/SlepianPosEnc.git
cd SlepianPosEnc

# Run California Housing experiment (no data download needed)
cd src/supervised_geographic_classification/scripts
bash run_california_experiments.sh
```

<!-- ## Citation

If you use this code, please cite: <update arxiv>
``` -->
