

---

# **`Survivor Predictor: Titanic Survival Analytics Engine`**

**Neural Networks • Python • Kaggle Benchmark • 891 Passengers**

A foundational deep learning project for binary classification in high-stakes survival prediction. Implements a feedforward neural network to analyze historical passenger data, identifying correlative patterns in maritime disaster outcomes based on demographic, socioeconomic, and familial variables.

![Project](https://img.shields.io/badge/PROJECT-Titanic_Survival-00FF00?style=for-the-badge&logo=kaggle&logoColor=white&labelColor=0D1117)
![Model](https://img.shields.io/badge/MODEL-Neural_Network-00FF00?style=for-the-badge&logo=tensorflow&logoColor=white&labelColor=0D1117)
![Data](https://img.shields.io/badge/DATA-891_Passengers-00FF00?style=for-the-badge&logo=pandas&logoColor=white&labelColor=0D1117)
![Accuracy](https://img.shields.io/badge/ACCURACY-85%25-00FF00?style=for-the-badge&logo=python&logoColor=white&labelColor=0D1117)
![Python](https://img.shields.io/badge/PYTHON-3.9%2B-00FF00?style=for-the-badge&logo=python&logoColor=white&labelColor=0D1117)

---

## Interactive Demonstration

<p align="center">
  <img src="assets/survivor-demo.gif" alt="Titanic survival prediction inference engine" width="85%">
</p>

<p align="center">
  <sub>Real-time inference across 891 passenger records — class, gender, age, and familial relationships as predictive signals.</sub>
</p>

---

## Strategic Objective

The RMS Titanic disaster presents a canonical binary classification problem in data science. This project constructs a neural network capable of identifying survival determinants from demographic and socioeconomic features. The objective is not probabilistic forecasting of historical tragedy but rather systematic exploration of feature importance, model architecture optimization, and foundational deep learning methodology applied to structured tabular data.

---

## Dataset Specification

| Attribute | Detail |
|-----------|--------|
| **Source** | Titanic passenger manifest (Kaggle competition dataset) |
| **Record Volume** | 891 passengers |
| **Feature Count** | 12 raw attributes |
| **Target Variable** | Survival (binary: 0 = Deceased, 1 = Survived) |
| **Class Distribution** | 549 deceased (61.6%) / 342 survived (38.4%) |

### Feature Dictionary

| Feature | Type | Description |
|---------|------|-------------|
| `Pclass` | Ordinal | Ticket class — 1st (Upper), 2nd (Middle), 3rd (Lower) |
| `Sex` | Binary categorical | Male / Female |
| `Age` | Continuous | Age in years (partial missingness) |
| `SibSp` | Discrete | Number of siblings / spouses aboard |
| `Parch` | Discrete | Number of parents / children aboard |
| `Fare` | Continuous | Ticket price (British pounds) |
| `Embarked` | Nominal categorical | Port of embarkation — Cherbourg (C), Queenstown (Q), Southampton (S) |

### Sample Passenger Manifest

| Survival | Class | Passenger Name | Sex | Age | Fare |
|:--------:|:-----:|----------------|:---:|:---:|-----:|
| 0 | 3rd | Braund, Mr. Owen Harris | Male | 22 | £7.25 |
| 1 | 1st | Cumings, Mrs. John Bradley | Female | 38 | £71.28 |
| 1 | 3rd | Heikkinen, Miss. Laina | Female | 26 | £7.93 |

---

## Model Architecture

| Layer | Configuration | Activation | Regularization |
|-------|---------------|------------|-----------------|
| Input | 7 features | — | — |
| Hidden Layer 1 | Linear (7 → 64) | ReLU | Dropout (p=0.2) |
| Hidden Layer 2 | Linear (64 → 32) | ReLU | None |
| Output Layer | Linear (32 → 1) | Sigmoid | None |

**Loss Function:** Binary Cross-Entropy  
**Optimizer:** Adam (adaptive moment estimation)  
**Evaluation Metric:** Accuracy (primary), Precision/Recall (secondary)

---

## Model Performance

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 85.0% |
| **Benchmark Reference** | Kaggle Titanic competition baseline |
| **Feature Importance (Top 3)** | Sex (female survival advantage), Pclass (inverse correlation with mortality), Fare (positive correlation with survival) |

---

## Repository Architecture

```
Survivor-predictor/
├── README.md
├── requirements.txt
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── model.py
│   └── preprocess.py
├── notebooks/
│   └── exploratory_analysis.ipynb
├── data/
│   ├── train.csv
│   └── test.csv
├── models/
│   └── titanic_nn.pth
└── assets/
    └── survivor-demo.gif
```

---

## Execution Instructions

| Step | Action |
|------|--------|
| **1** | Install dependencies: `pip install -r requirements.txt` |
| **2** | Load dataset into `data/` directory |
| **3** | Execute preprocessing: `python src/preprocess.py` |
| **4** | Train model: `python src/train.py` |
| **5** | Generate predictions: `python src/predict.py` |

**Requirements:** Python 3.9+, PyTorch, Pandas, NumPy, Scikit-learn

---

## Key Analytical Findings

| Finding | Implication |
|---------|-------------|
| Female passengers demonstrated significantly higher survival probability across all classes | Gender-based evacuation protocol (women and children first) evident in data |
| First-class passengers exhibited ~3x higher survival rate than third-class | Socioeconomic status correlated with lifeboat access |
| Children (age < 12) showed elevated survival independent of class | Age-based prioritization in evacuation |
| Fare normalized by class revealed within-class variance in survival | Wealth stratification within passenger classes |

---

## Limitations & Future Work

| Limitation | Proposed Enhancement |
|------------|----------------------|
| Missing age values (20% of records) | Multiple imputation or probabilistic age modeling |
| No cabin location data | Incorporate deck-level proximity to lifeboats |
| Binary classification only | Extend to survival probability calibration |
| Historical single-event dataset | Transfer learning to other maritime disasters |
| No temporal evacuation sequencing | Add lifeboat launch order as feature |

---

