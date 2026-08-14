# 🚕 Deep Learning Prediction of NYC Yellow Taxi Trip Duration Using Weather Data

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow">
  <img src="https://img.shields.io/badge/Notebook-Jupyter-lightgrey?logo=jupyter">
  <img src="https://img.shields.io/badge/Pace%20University-CS672-blue">
</p>

A neural-network regression project that predicts **NYC Yellow Taxi trip duration** from trip and weather features, comparing three architectures across two optimizers. Built for **CS672: Introduction to Deep Learning (Fall 2025)** at Pace University.

> ⚠️ **Leakage-free by design.** Quantities known only *after* a trip ends, namely `avg_mph` (distance ÷ duration), `fare_amount`, and `tip_amount`, are deliberately **excluded** because they encode the target. Reported errors therefore reflect realistic prediction from information available at (or estimable before) pickup. An earlier leaky version of this project reported an artificial ~0.19 min MAE, driven entirely by `avg_mph`. That version is not committed here, so the figure cannot be reproduced from this repository.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Highlights](#-highlights)
- [Demo](#-demo)
- [Dataset](#-dataset)
- [Features](#-features)
- [Methodology](#-methodology)
- [Models & Results](#-models--results)
- [A Known Bug in the Training Loop](#-a-known-bug-in-the-training-loop)
- [Limitations & Next Steps](#️-limitations--next-steps)
- [How to Run](#-how-to-run)
- [Environment](#️-environment)
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

## ✨ Highlights

- Ran a controlled experiment across **3 architectures × 2 optimizers (6 configurations)**, comparing them on a held-out, time-ordered validation split rather than a random one — the harder, more realistic evaluation setup.
- Deliberately excluded post-trip fields (`avg_mph`, `fare_amount`, `tip_amount`) that would leak the target, after an earlier version of the project revealed how badly leakage can inflate results — and documented that finding rather than hiding it.
- **Found and diagnosed a subtle bug** in the training loop: a single `EarlyStopping` callback was reused across all six `model.fit()` calls, so later experiments were being judged against a stopping threshold inherited from an unrelated earlier model. This was traced using epoch-by-epoch validation-loss evidence, not just spotted by inspection.
- Reported results with the caveats a rigorous evaluation requires: which comparisons are confounded by unequal training budgets, which metric (final-epoch vs. best-epoch) was actually recorded, and where the validation split's limited date range restricts what conclusions the weather features can support.

---

## 🎥 Demo

*This project generates several plots during the notebook run — a correlation heatmap, a rainy-vs-clear boxplot, and train/validation loss curves for the best model. Embedding one or two of these directly here (e.g. the MAE comparison chart) would give anyone skimming the repo an instant sense of the result before they read a single line.*

```
![Validation MAE by model](docs/results_chart.png)
```

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

> `tsun` (sunshine duration) was requested from Meteostat but returned fully null and was dropped automatically, taking the feature count from 15 candidates to 14.

---

## 🔬 Methodology

1. **Data load and merge.** Yellow Taxi (Jan 2020) joined to daily Meteostat weather on pickup date, with linear interpolation to fill weather gaps.
2. **Feature engineering.** Duration target, hour/weekday extraction, weekend and rush-hour indicators (rush hour = 7-9 and 16-19).
3. **EDA.** Correlation heatmap, rain versus clear-day duration, temperature effect, weather pair-plots.
4. **Time-aware split.** Sorted by pickup time, then **80% train / 20% validation with no shuffling**, so the model never trains on trips that occurred after those it is evaluated on.
5. **Scaling.** `StandardScaler` fit on training data only.
6. **Training.** MSE loss, MAE metric, 40-epoch budget, batch size 32, `EarlyStopping(patience=5, restore_best_weights=True)`, optimizers Adam and RMSprop at lr = 0.001.

---

## 📈 Models & Results

| Rank | Model | Optimizer | Epochs run | Train MAE | **Val MAE (min)** | Train MSE | Val MSE |
|---|---|---|---|---|---|---|---|
| 1 | **MLP** | Adam | 18 | 2.944 | **3.306** | 20.568 | **25.490** |
| 2 | DNN | Adam | **5** | 3.116 | 3.323 | 22.903 | 27.377 |
| 3 | DNN | RMSprop | **5** | 3.149 | 3.457 | 23.000 | 28.144 |
| 4 | MLP | RMSprop | **5** | 3.008 | 3.553 | 21.391 | 28.878 |
| 5 | Linear | RMSprop | 8 | 4.043 | 4.224 | 33.364 | 37.479 |
| 6 | Linear | Adam | 40 | 4.049 | 4.229 | 33.358 | 37.582 |

> **Read the "Epochs run" column before the MAE column.** Three of the six configurations were cut off at five epochs by a callback defect, not by convergence. See [A Known Bug in the Training Loop](#-a-known-bug-in-the-training-loop). Comparisons between rows with different epoch counts are not like-for-like.

### 🏆 Best Model
> **MLP (Adam, lr = 0.001), Validation MAE ≈ 3.31 minutes** (MSE ≈ 25.49)

### Interpretation

**Non-linearity is worth about 0.9 minutes.** Both neural architectures beat linear regression by roughly 0.9 min MAE, confirming a real but moderate non-linear relationship between the features and trip duration. This conclusion survives the epoch-budget problem: the linear models got 40 and 8 epochs and still finished a full minute behind networks that trained for 5 and 18. The relationship is not strongly non-linear; a linear model still gets within 4.2 minutes.

**Whether depth helps is untested.** The 2-layer MLP (3.306) edged out the 3-layer DNN (3.323) by 0.017 minutes — but the MLP trained for **18 epochs and the DNN for 5**, and the DNN's validation loss was still falling when it stopped (28.38 → 27.50 → 27.38 across its last three epochs). This project therefore does **not** show that the extra depth and dropout bought nothing. It shows that a well-trained MLP beats an under-trained DNN, which is not an architectural finding. Rerunning with an equal budget is the first item in Next Steps.

**Optimizer choice.**

| Model | Adam | RMSprop | Gap | Equal budget? |
|---|---|---|---|---|
| MLP | **3.306** (18 ep) | 3.553 (5 ep) | 0.247 | ✗ |
| DNN | **3.323** (5 ep) | 3.457 (5 ep) | 0.133 | ✓ |
| Linear | 4.229 (40 ep) | **4.224** (8 ep) | 0.005 | ✗ |

Only the DNN pair trained for the same number of epochs, and there Adam wins by 0.133 minutes — the one defensible optimizer comparison in the set. The Linear pair is the other informative row: RMSprop matched Adam to within 0.005 minutes using a fifth of the epochs, which is what you would expect on a convex loss surface where both optimizers reach the same solution. The MLP gap of 0.247 is confounded by the 18-versus-5 epoch difference and should not be quoted.

**Feature contributions.** `trip_distance` is the dominant predictor by a wide margin. Weather contributes a small effect, with trips running slightly longer on rainy days, though see the limitation on weather granularity below.

---

## 🐛 A Known Bug in the Training Loop

A single `EarlyStopping` instance is created once and reused across all six `model.fit()` calls:

```python
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

for model_name, builder in model_builders.items():
    for opt_name, opt_class in optimizers_config:
        ...
        history = compile_and_train(..., callbacks=[early_stop])
```

The callback's `best` value carries over between runs, so later experiments are judged against a threshold set by an earlier, unrelated model.

**The evidence.** DNN/Adam's validation loss across its five epochs:

```
28.0388 → 27.8707 → 28.3811 → 27.4990 → 27.3774
```

It improved on epochs 2, 4, and 5 — its final epoch was its best — and training stopped anyway. With a correctly reset callback, `wait` would have been 0 and training would have continued. Stopping is only possible if `best` was inherited from the MLP/Adam run (~25.0), a value the DNN never reached, so `wait` incremented on every epoch. The same explains MLP/RMSprop and DNN/RMSprop halting at exactly five.

**The fix**, one line moved inside the loop:

```python
for model_name, builder in model_builders.items():
    for opt_name, opt_class in optimizers_config:
        for lr in learning_rates:
            early_stop = EarlyStopping(monitor="val_loss", patience=5,
                                       restore_best_weights=True)
            ...
```

**What it affects:** the MLP-versus-DNN conclusion and the MLP optimizer gap. It does **not** affect the leakage-free feature design, the time-aware split, the data pipeline counts, or the finding that both networks beat linear regression.

---

## ⚠️ Limitations & Next Steps

1. **Shared `EarlyStopping` truncated three of six runs.** Documented above. Rerunning with a fresh callback per experiment is the single most important fix, and it is what would turn the architecture comparison into a real result.
2. **Reported metrics are final-epoch, not best-epoch.** The results table records `history.history["val_mae"][-1]`, while `restore_best_weights=True` means the retained model holds the *best* epoch's weights. The saved `.keras` files are therefore better than the numbers attached to them, and the ranking could shift. Using `min(history.history["val_mae"])` would report what was actually kept.
3. **No held-out test set.** The pipeline has train and validation only, and the best of six configurations was selected *using* validation MAE. The reported 3.31 minutes is therefore an optimistic estimate. A three-way time-ordered split would give an unbiased figure.
4. **Weather is effectively a date proxy.** January 2020 provides only 31 distinct daily weather observations across 314,709 rows. Combined with the time-ordered split, the training set covers roughly January 1 to 25 and validation covers January 25 to 31, meaning **the model extrapolates on every weather feature during validation**. Hourly weather, or several months of data, would be needed to test the weather hypothesis properly.
5. **Single month, single year.** January 2020 alone cannot capture seasonal effects, and the last week of the month is the entire validation window.
6. **Pickup and dropoff zones are unused.** `PULocationID` and `DOLocationID` are available and are almost certainly stronger predictors of duration than any weather variable, since route geography drives travel time.
7. **The Adam linear model never early-stopped.** Its `val_loss` decreased by roughly 0.0001 per epoch, enough to reset patience each time, so it ran the full 40 epochs while converging by epoch 3. (The RMSprop linear model did stop, at epoch 8.) Adding `min_delta=0.01` would halt the Adam run appropriately.
8. **`trip_distance` is metered, not estimated.** At true prediction time this value would come from a routing engine and carry its own error, which would raise the realistic MAE above 3.31.
9. **One learning rate tested.** Only lr = 0.001 was evaluated. A sweep would establish whether the Adam versus RMSprop gap survives tuning.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/krishnamaniyar2209/nyc-taxi-dl.git
   cd nyc-taxi-dl
   ```
2. Open the notebook in **Google Colab** or Jupyter (GPU recommended).
3. Run the install cell, then **Runtime → Restart session**, then run all cells top to bottom. Taxi data streams from the TLC CDN and weather is fetched automatically via Meteostat.

> The final cell writes trained models to a relative `content/saved_models/` directory (`Linear_RMSprop_lr1e-03.keras`, `MLP_Adam_lr1e-03.keras`, `DNN_Adam_lr1e-03.keras`). These are generated locally and are not committed.

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
- 📧 maniyarkrishnakm22@gmail.com
- 🔗 [GitHub](https://github.com/krishnamaniyar2209) · [LinkedIn](https://www.linkedin.com/in/krishnamaniyar/) · [Portfolio](https://krishnamaniyar2209.github.io/)
