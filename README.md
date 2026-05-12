# Garmin Biometric Anomaly Detection for Principles of IoT Devices Course 
One-Class SVM anomaly detection trained on personal Garmin Forerunner 165 smartwatch data to detect biometric impersonation without labeled training data.

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

```bash
# Windows
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install garminconnect pandas scikit-learn matplotlib
```

## Run

### Fetch raw data from Garmin Connect
```bash
python fetch-garmin.py
```
This script prompts for user's Garmin credentials and pulls the last 65 days of biometric data (HRV, stress, sleep, body battery, steps, respiration, SpO2, RHR, activities) via the Garmin Connect API. Raw data saved to `./raw_data/` as JSON files.

> Note: Garmin's API may rate limit repeated login attempts... If authentication fails, wait and retry, or switch networks.

### Build datasets from raw data
```bash
python preprocessing.py
```
This script parses raw JSON files and builds three  CSVs saved to `./datasets/`:
- `dataset_A_cardiovascular.csv` — HRV, SpO2, respiration, activity HR, calories
- `dataset_B_stress_recovery.csv` — HRV, stress, body battery, steps, sleep hours
- `dataset_C_sleep.csv` — HRV, SpO2, respiration, sleep hours, sleep stage %


### Split/Train/Predict
```bash
python predict.py
```
Splits data (first 50 days = training, remainder = test). Trains a One-Class SVM independently on each dataset using the legitimate user's baseline data. Models saved to `./models/`, and scores test days. Impersonator days are labeled separately. Results saved to `./results/`.

> Update `IMPERSONATOR_A` and `IMPERSONATOR_B` date sets in `predict.py` before running if necessary. Currently they're hardcoded for the project LOL

### Generate visualizations
```bash
python visualize.py
```
Generates all figures to `./figures/`, including time-series baselines, scatter plots, correlation heatmap, OC-SVM decision scores, and baseline vs. impersonator comparison plots.

## Pipeline Overview

```
fetch-garmin.py --> preprocessing.py -> split_and_predict.py --> visualize.py
```

## Project Structure

```
.
├── raw_data/               # Raw JSON files from Garmin API
├── datasets/               # Preprocessed feature CSVs (A, B, C)
├── models/                 # Trained OC-SVM model files (.joblib) and metadata (.json)
├── results/                # Anomaly scoring results CSVs
├── figures/                # Generated visualization plots
├── fetch-garmin.py         # Garmin API data collection
├── preprocessing.py        # Data parsing and dataset engineering
├── predict.py              # OC-SVM training/testing and prediction
└── visualize.py            # Generate figures
```

## Notes
- Missing sensor values are imputed with the column median during preprocessing
- Days with < 3 valid features are excluded from datasets (so not all collected days are included)
- OC-SVM configured with RBF kernel, `nu=0.05`, `gamma="scale"`
- Decision scores below 0 are classified as anomalies; more negative = more anomalous