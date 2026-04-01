<p align="center">
  <img src="https://github.com/krishnam229/Deep-Learning/blob/main/banner.png" width="900">
</p>

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Jupyter Notebook](https://img.shields.io/badge/Notebook-Jupyter-lightgrey)

# Deep Learning–Based Prediction of NYC Yellow Taxi Trip Duration Using Weather Data

This repository contains two major projects completed for **CS672 – Introduction to Deep Learning (Fall 2025)** at Pace University:

**Deep Learning Regression Models for Trip Duration Prediction**

The goal is to understand factors influencing taxi ride durations and build neural network models that accurately predict trip time using enriched features, including NYC weather conditions.


# Deep Learning Regression for Trip Duration Prediction

This project builds and compares multiple neural network architectures to predict NYC Yellow Taxi trip duration using both **trip features** and **weather data**.

### Steps Performed
- Retrieved NYC climate data for January 2020 from Meteostat.  
- Merged weather data with taxi records based on date.  
- Scaled numeric features using StandardScaler.  
- Developed and trained three neural network models:
  1. **Linear Regression (no hidden layers)**  
  2. **Multi-Layer Perceptron (MLP)**  
  3. **Deep Neural Network (DNN)** with dropout and deeper architecture  

### Training Configuration
- **Loss Functions:** MSE, MAE  
- **Optimizers:** SGD, Adam, RMSprop  
- **Learning Rates:** 0.01, 0.001, 0.0001  
- **Epochs:** 100  
- **Batch Size:** 32  

Training and validation loss curves were analyzed to assess model stability and accuracy.

### Best Model
Based on validation MAE:

> **DNN with RMSprop (lr = 0.001)**  
> Showed the most stable and accurate performance.

---

# 3. Repository Structure

```text
📦 nyc-taxi-dl
│
├── NYC_Taxi_Trip_Duration_DeepLearning.ipynb
└── README.md                                  # Documentation
```

---

# 4. Model Performance Summary

| Model                     | Optimizer | Learning Rate | Val MAE | Val MSE |
|---------------------------|-----------|----------------|---------|---------|
| Linear Regression (TF)    | Adam      | 0.001          | 0.29    | 0.84    |
| MLP (2 Hidden Layers)     | RMSprop   | 0.001          | 0.24    | 0.56    |
| DNN (Deep Neural Network) | RMSprop   | 0.001          | **0.19** | **0.32** |

### Interpretation
- The **DNN model** performed best due to its ability to capture non-linear relationships.  
- The **MLP** showed moderate accuracy with mild overfitting.  
- **Linear Regression** underperformed, confirming that trip duration prediction benefits from deeper architectures.

---

# 5. Project Highlights

- Built a complete end-to-end **data science + deep learning pipeline**.  
- Performed detailed **EDA, data cleaning, and feature engineering**.  
- Integrated real **NYC weather data** to improve prediction accuracy.  
- Trained and compared **three neural network architectures**.  
- Tuned hyperparameters across optimizers, learning rates, and layers.  
- Evaluated models using **MAE** and **MSE**, visualizing training curves.  
- Best performance achieved using **DNN with RMSprop**.  
- Designed a clean, professional **GitHub structure and documentation**.

---

# 6. Environment & Dependencies

This project uses:

- Python 3.x  
- TensorFlow 2.x  
- pandas  
- numpy  
- scikit-learn  
- seaborn  
- matplotlib  

Install required packages:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
```

Kernel used during development: **Python (dl2020) – Anaconda environment**

---

# 7. How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/krishnam229/Deep-Learning.git
   ```
2. Open notebooks in **Jupyter Notebook** or **VS Code**.  
3. Ensure all dependencies are installed.  
4. Run all cells sequentially to reproduce the analysis and results.

---

# 8. Closing Notes

This repository demonstrates the complete pipeline for:

- Data ingestion  
- Exploratory analysis  
- Feature engineering  
- Deep learning model development  
- Performance evaluation  

Feel free to explore the notebooks and reach out with any questions.

---
