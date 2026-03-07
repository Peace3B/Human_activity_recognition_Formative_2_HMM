# Human Activity Recognition Using Hidden Markov Models

A machine learning project that classifies human activities — walking, sitting, standing, and more — from wearable motion sensor data using Gaussian Hidden Markov Models.

---

## Overview

Human activities unfold as sequences over time, making HMMs a natural fit for this task. The system extracts statistical and frequency-based features from accelerometer and gyroscope signals, then trains a dedicated Gaussian HMM per activity class. Classification is performed by selecting the model that returns the highest log-likelihood score for a given observation.

---

## Pipeline

**Data Loading → Window Segmentation → Feature Extraction → Scaling → PCA → HMM Training → Evaluation**

| Step | Description |
|------|-------------|
| Data Loading | Sensor recordings (ax, ay, az, gx, gy, gz) loaded from compressed CSVs |
| Segmentation | Continuous signals split into fixed-length time windows |
| Feature Extraction | Time-domain (mean, std, variance) and frequency-domain (dominant freq, SMA) features |
| Scaling | StandardScaler applied for consistent feature ranges |
| PCA | Dimensionality reduction to reduce noise and improve training efficiency |
| HMM Training | One Gaussian HMM trained per activity class |
| Evaluation | Confusion matrix, sensitivity, specificity, and accuracy |

---

## Results

The model demonstrates strong classification performance across most activity classes, with predicted activity sequences closely tracking ground truth over time. Activities with distinct motion profiles achieve the highest accuracy.

---

## Tech Stack

`Python` `NumPy` `Pandas` `Scikit-learn` `hmmlearn` `Matplotlib` `Jupyter Notebook`

---

## Project Structure

```
├── Formative_2_Hidden_Markov_Models.ipynb
├── dataset/
│   └── activity_data.zip
└── README.md
```

---

## Author

**Peace Keza and Niyonzima Stecie** · Software Engineering · African Leadership University
