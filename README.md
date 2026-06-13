<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow">
  <img src="https://img.shields.io/badge/Notebook-Jupyter-lightgrey?logo=jupyter">
  <img src="https://img.shields.io/badge/Pace%20University-CS672-blue">
</p>

# 🚕 Deep Learning–Based Prediction of NYC Yellow Taxi Trip Duration Using Weather Data

A neural-network regression project that predicts **NYC Yellow Taxi trip duration** from trip and weather features, comparing three architectures — built for **CS672: Introduction to Deep Learning (Fall 2025)** at Pace University.

> ⚠️ **Leakage-free by design.** Quantities known only *after* a trip ends — `avg_mph` (= distance ÷ duration), `fare_amount`, `tip_amount` — are deliberately **excluded** because they encode the target. Reported errors therefore reflect realistic prediction from information available at (or estimable before) pickup.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Features](#-features)
- [Methodology](#-methodology)
- [Models & Results](#-models--results)
- [How to Run](#-how-to-run)
- [Environment](#-environment)
- [Team](#-team)

---

## 🔬 Overview

The goal is to predict `trip_duration_min` for NYC Yellow Taxi rides using trip characteristics enriched with daily weather data. Three TensorFlow/Keras models are trained and compared:

1. **Linear Regression** (no hidden layers)
2. **MLP** (two hidden layers)
3. **DNN** (deeper network with dropout)

Each is trained with two optimizers (Adam, RMSprop) and evaluated with MAE and MSE.

---

## 📊 Dataset

| Source | Details |
|---|---|
| **Taxi trips** | [NYC TLC — Yellow Taxi, January 2020](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) (Parquet) |
| **Weather** | Daily NYC weather (Jan 2020) via the **Meteostat** API (Wall Street station) |
| **Target** | `trip_duration_min` = drop-off − pickup, in minutes |
| **Sample used** | 5% of cleaned trips (~314,700 rows) for training efficiency |

**Cleaning filters:** trip duration 1–180 min, trip distance 0–50 mi, non-null distance/duration. Taxi and weather are merged on the pickup date.

---

## 🧩 Features

**Leakage-free feature set** (known or estimable at pickup):

| Group | Features |
|---|---|
| Trip | `passenger_count`, `trip_distance` |
| Time | `pickup_hour`, `pickup_weekday`, `is_weekend`, `is_rush_hour` |
| Weather | `tavg`, `tmin`, `tmax`, `prcp`, `snow`, `wdir`, `wspd`, `pres` |

> `trip_distance` is the metered distance; in production it would be replaced by a route-estimated distance available at pickup.

---

## 🔬 Methodology

1. **Data load & merge** — Yellow Taxi (Jan 2020) + daily Meteostat weather, joined on pickup date.
2. **Feature engineering** — duration target, time features, rainy flag, temperature buckets.
3. **EDA** — correlation heatmap, rain vs. clear-day duration, temperature effect, weather pair-plots.
4. **Time-aware split** — sorted by pickup time, **80% train / 20% validation** (no shuffling → no future leakage). Train ≈ 251,800 · Val ≈ 62,900.
5. **Scaling** — `StandardScaler` fit on training data only.
6. **Training** — MSE loss, MAE metric, 40 epochs, batch size 32, `EarlyStopping` (patience 5, restore best weights), optimizers Adam & RMSprop at lr = 0.001.

---

## 📈 Models & Results

| Model | Optimizer | Val MAE (min) | Val MSE |
|---|---|---|---|
| **MLP** | Adam | **3.31** | **25.49** |
| DNN | Adam | 3.32 | 27.38 |
| DNN | RMSprop | 3.46 | 28.14 |
| MLP | RMSprop | 3.55 | 28.88 |
| Linear | RMSprop | 4.22 | 37.48 |
| Linear | Adam | 4.23 | 37.58 |

### 🏆 Best Model
> **MLP (Adam, lr = 0.001) — Validation MAE ≈ 3.31 minutes** (MSE ≈ 25.49)

### Interpretation
- **MLP and DNN beat Linear Regression by ~0.9 minutes**, confirming a mild non-linear relationship between the features and trip duration.
- All neural models predict within **~3.3 minutes** — a realistic error once post-trip leakage is removed. (An earlier leaky version reported an artificial ~0.19 min, driven by `avg_mph`, which is computed *from* the target.)
- `trip_distance` is the dominant predictor; weather adds a small but measurable effect (trips run slightly longer on rainy days).

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/krishnamaniyar2209/nyc-taxi-dl.git
   cd nyc-taxi-dl
   ```
2. Open the notebook in **Google Colab** or Jupyter.
3. Run the install cell, then **Runtime → Restart session**, then run all cells top to bottom. Weather data is fetched automatically via Meteostat.

---

## 🛠️ Environment

```bash
pip install "meteostat==1.6.8" "pandas==2.2.2" tensorflow scikit-learn matplotlib seaborn
```
Python 3.10 · TensorFlow 2.x · pandas 2.2 · scikit-learn · seaborn · matplotlib · meteostat 1.6.8

---

## 📄 License

This project is licensed under the MIT License.
