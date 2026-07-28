<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow">
  <img src="https://img.shields.io/badge/Notebook-Jupyter-lightgrey?logo=jupyter">
  <img src="https://img.shields.io/badge/Pace%20University-CS672-blue">
</p>

# 🚕 Deep Learning Prediction of NYC Yellow Taxi Trip Duration Using Weather Data

A neural-network regression project that predicts **NYC Yellow Taxi trip duration** from trip and weather features, comparing three architectures across two optimizers. Built for **CS672: Introduction to Deep Learning (Fall 2025)** at Pace University.

> ⚠️ **Leakage-free by design.** Quantities known only *after* a trip ends, namely `avg_mph` (distance ÷ duration), `fare_amount`, and `tip_amount`, are deliberately **excluded** because they encode the target. Reported errors therefore reflect realistic prediction from information available at (or estimable before) pickup. An earlier leaky version of this project reported an artificial ~0.19 min MAE, driven entirely by `avg_mph`.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Features](#-features)
- [Methodology](#-methodology)
- [Models & Results](#-models--results)
- [Limitations & Next Steps](#-limitations--next-steps)
- [How to Run](#-how-to-run)
- [Environment](#-environment)
- [Author](#-author)

---

## 🔬 Overview

The goal is to predict `trip_duration_min` for NYC Yellow Taxi rides using trip characteristics enriched with daily weather data. Three TensorFlow/Keras architectures are trained and compared:

| Model | Architecture |
|---|---|
| **Linear Regression** | `Dense(1)`, no hidden layers |
| **MLP** | `Dense(64, relu)` → `Dense(32, relu)` → `Dense(1)` |
| **DNN** | `Dense(128, relu)` → `Dropout(0.3)` → `Dense(64, relu)` → `Dropout(0.2)` → `Dense(32, relu)` → `Dense(1)` |

Each is trained with two optimizers (Adam, RMSprop) at lr = 0.001, giving six experiments evaluated on MAE and MSE.

---

## 📊 Dataset

| Source | Details |
|---|---|
| **Taxi trips** | [NYC TLC, Yellow Taxi, January 2020](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) (Parquet) |
| **Weather** | Daily NYC weather (Jan 2020) via the **Meteostat** API, `Point(40.7060, -74.0086)`, Wall Street |
| **Target** | `trip_duration_min` = drop-off − pickup, in minutes |

### Data Pipeline

| Stage | Rows |
|---|---|
| Raw January 2020 trips | 6,405,008 |
| After cleaning filters | 6,294,171 (98.3% retained) |
| **5% sample for training** | **314,709** |
| Train (first 80% by time) | 251,767 |
| Validation (last 20% by time) | 62,942 |

**Cleaning filters:** trip duration in (1, 180] minutes, trip distance in (0, 50] miles, non-null distance and duration. Taxi and weather are merged on the pickup date.

---

## 🧩 Features

**Leakage-free feature set** (known or estimable at pickup), 14 features total:

| Group | Features |
|---|---|
| Trip | `passenger_count`, `trip_distance` |
| Time | `pickup_hour`, `pickup_weekday`, `is_weekend`, `is_rush_hour` |
| Weather | `tavg`, `tmin`, `tmax`, `prcp`, `snow`, `wdir`, `wspd`, `pres` |

> `trip_distance` is the metered distance. In production it would be replaced by a route-estimated distance available at pickup, which would introduce some additional error.

> `tsun` (sunshine duration) was requested from Meteostat but returned fully null and was dropped automatically.

---

## 🔬 Methodology

1. **Data load and merge.** Yellow Taxi (Jan 2020) joined to daily Meteostat weather on pickup date, with linear interpolation to fill weather gaps.
2. **Feature engineering.** Duration target, hour/weekday extraction, weekend and rush-hour indicators (rush hour = 7-9 and 16-19).
3. **EDA.** Correlation heatmap, rain versus clear-day duration, temperature effect, weather pair-plots.
4. **Time-aware split.** Sorted by pickup time, then **80% train / 20% validation with no shuffling**, so the model never trains on trips that occurred after those it is evaluated on.
5. **Scaling.** `StandardScaler` fit on training data only.
6. **Training.** MSE loss, MAE metric, 40 epochs, batch size 32, `EarlyStopping(patience=5, restore_best_weights=True)`, optimizers Adam and RMSprop at lr = 0.001.

---

## 📈 Models & Results

| Rank | Model | Optimizer | Train MAE | **Val MAE (min)** | Train MSE | Val MSE |
|---|---|---|---|---|---|---|
| 1 | **MLP** | Adam | 2.944 | **3.306** | 20.568 | **25.490** |
| 2 | DNN | Adam | 3.116 | 3.323 | 22.903 | 27.377 |
| 3 | DNN | RMSprop | 3.149 | 3.457 | 23.000 | 28.144 |
| 4 | MLP | RMSprop | 3.008 | 3.553 | 21.391 | 28.878 |
| 5 | Linear | RMSprop | 4.043 | 4.224 | 33.364 | 37.479 |
| 6 | Linear | Adam | 4.049 | 4.229 | 33.358 | 37.582 |

### 🏆 Best Model
> **MLP (Adam, lr = 0.001), Validation MAE ≈ 3.31 minutes** (MSE ≈ 25.49)

### Interpretation

**Non-linearity is worth about 0.9 minutes.** Both neural architectures beat linear regression by roughly 0.9 min MAE, confirming a real but moderate non-linear relationship between the features and trip duration. The relationship is not strongly non-linear; a linear model still gets within 4.2 minutes.

**The shallower network won.** The 2-layer MLP (3.306) outperformed the deeper 3-layer DNN with dropout (3.323). With 14 features and 251,767 training rows, the extra depth and regularization of the DNN bought nothing. Capacity was not the constraint here; feature information was.

**Optimizer choice matters only for the non-linear models.**

| Model | Adam | RMSprop | Gap |
|---|---|---|---|
| MLP | **3.306** | 3.553 | 0.247 |
| DNN | **3.323** | 3.457 | 0.134 |
| Linear | 4.229 | **4.224** | 0.005 |

Adam's advantage grows with architectural complexity and disappears entirely for the linear model, whose convex loss surface both optimizers solve equally well. A clean illustration of when optimizer tuning is worth the effort.

**Feature contributions.** `trip_distance` is the dominant predictor by a wide margin. Weather contributes a small effect, with trips running slightly longer on rainy days, though see the limitation on weather granularity below.

---

## ⚠️ Limitations & Next Steps

1. **No held-out test set.** The pipeline has train and validation only, and the best of six configurations was selected *using* validation MAE. The reported 3.31 minutes is therefore an optimistic estimate. A three-way time-ordered split would give an unbiased figure.
2. **Weather is effectively a date proxy.** January 2020 provides only 31 distinct daily weather observations across 314,709 rows. Combined with the time-ordered split, the training set covers roughly January 1 to 25 and validation covers January 25 to 31, meaning **the model extrapolates on every weather feature during validation**. Hourly weather, or several months of data, would be needed to test the weather hypothesis properly.
3. **Single month, single year.** January 2020 alone cannot capture seasonal effects, and the last week of the month is the entire validation window.
4. **Pickup and dropoff zones are unused.** `PULocationID` and `DOLocationID` are available and are almost certainly stronger predictors of duration than any weather variable, since route geography drives travel time.
5. **EarlyStopping never fired on the linear models.** Their `val_loss` decreased by roughly 0.0002 per epoch, enough to reset patience each time, so both ran the full 40 epochs while converging by epoch 3. Adding `min_delta=0.01` would stop them appropriately.
6. **`trip_distance` is metered, not estimated.** At true prediction time this value would come from a routing engine and carry its own error, which would raise the realistic MAE above 3.31.
7. **One learning rate tested.** Only lr = 0.001 was evaluated. A sweep would establish whether the Adam versus RMSprop gap survives tuning.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/krishnamaniyar2209/nyc-taxi-dl.git
   cd nyc-taxi-dl
   ```
2. Open the notebook in **Google Colab** or Jupyter (GPU recommended).
3. Run the install cell, then **Runtime → Restart session**, then run all cells top to bottom. Taxi data streams from the TLC CDN and weather is fetched automatically via Meteostat.

---

## 🛠️ Environment

```bash
pip install "meteostat==1.6.8" "pandas==2.2.2" tensorflow scikit-learn matplotlib seaborn
```

Python 3.10 · TensorFlow 2.20 · pandas 2.2.2 · scikit-learn · seaborn · matplotlib · meteostat 1.6.8

---

## 👤 Author

**Krishna Maniyar**, Data Analyst
- 🎓 Pace University, Seidenberg School of CSIS, MS in Data Science
- 📘 CS672: Introduction to Deep Learning (Fall 2025)
- 📧 krishnamaniyarkm22@gmail.com
- 🔗 [GitHub](https://github.com/krishnamaniyar2209) · [LinkedIn](https://www.linkedin.com/in/krishnamaniyar/) · [Portfolio](https://krishnamaniyar2209.github.io/)
