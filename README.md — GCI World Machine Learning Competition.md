# 🏈 GCI World — NFL Draft Prediction

A machine learning solution developed for the **GCI World Machine Learning Competition**, focused on predicting whether a football player would be drafted based on physical measurements, player information, and derived performance features.

The solution uses an ensemble of **XGBoost, LightGBM, and CatBoost**, combined through rank-based blending to produce the final predictions.

---

## 📌 Project Overview

The goal of the competition was to predict the probability that a player would be **drafted**.

The dataset contains player-level information such as:

- Age
- Height
- Weight
- 40-yard sprint
- Vertical jump
- Bench press repetitions
- Broad jump
- Shuttle
- 3-cone agility
- Position
- Position type
- Player type
- School

The target variable is:

```text
drafted
```

The final submission contains:

```text
id, Drafted
```

where `Drafted` represents the predicted probability of being drafted.

---

## 🧠 Approach

The solution follows a feature engineering + ensemble learning pipeline.

### 1. Feature Selection

The model uses a combination of physical measurements and categorical player information.

Numerical features include:

```text
age
height
weight
sprint_40yd
vertical_jump
bench_press_reps
broad_jump
shuttle
agility_3cone
```

Categorical features include:

```text
position
position_type
player_type
```

---

## ⚙️ Feature Engineering

Several additional features were created to capture relationships between physical attributes.

### BMI

Body Mass Index was calculated from height and weight:

```python
BMI = (weight * 703) / (height ** 2)
```

### Explosive Power

A combined measure of jumping performance:

```python
Explosive_Power = vertical_jump * broad_jump
```

### Speed / Weight Ratio

A feature combining sprint performance and body weight:

```python
Speed_Weight_Ratio = weight / sprint_40yd
```

### School Popularity

The frequency of a player's school in the training dataset was used as an additional feature:

```python
School_Popularity = school_frequency
```

This provides information about how frequently a school appears in the dataset.

---

## 🤖 Machine Learning Models

Three gradient boosting models were used:

### XGBoost

```text
n_estimators: 800
max_depth: 4
learning_rate: 0.015
subsample: 0.8
colsample_bytree: 0.6
```

### LightGBM

```text
n_estimators: 800
learning_rate: 0.015
max_depth: 5
num_leaves: 31
subsample: 0.8
colsample_bytree: 0.6
```

### CatBoost

```text
iterations: 800
learning_rate: 0.02
depth: 5
```

Using different boosting algorithms helps create model diversity and can improve the robustness of the final predictions.

---

## 🔄 Cross Validation

The training data was evaluated using **10-Fold Stratified Cross Validation**.

```python
StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)
```

Stratification was used to maintain a similar distribution of the target classes across folds.

---

## 🧩 Ensemble Strategy

Instead of directly averaging the predicted probabilities, the predictions from each model were converted into ranks.

For each model:

```python
rank = rankdata(predictions) / len(predictions)
```

The ranked predictions were then combined using weighted blending:

```text
XGBoost   → 45%
LightGBM  → 40%
CatBoost  → 15%
```

Final prediction:

```python
blended_ranks = (
    rank_xgb * 0.45 +
    rank_lgb * 0.40 +
    rank_cat * 0.15
)
```

The final predictions from all 10 folds were averaged to produce the submission.

---

## 📊 Pipeline

```text
Raw Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Feature Engineering
     │
     ├── BMI
     ├── Explosive Power
     ├── Speed/Weight Ratio
     └── School Popularity
     │
     ▼
Categorical Encoding
     │
     ▼
10-Fold Stratified CV
     │
     ├───────────────┐
     ▼               ▼
  XGBoost         LightGBM
     │               │
     └───────┬───────┘
             │
          CatBoost
             │
             ▼
      Rank Transformation
             │
             ▼
       Weighted Blending
             │
             ▼
      Final Predictions
             │
             ▼
     Competition Submission
```

---

## 🛠️ Technologies

The project was implemented in Python using:

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- SciPy

---

## 📁 Suggested Project Structure

```text
gci-world-ml/
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   │
│   └── processed/
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── submissions/
│   └── Rank_Blended_10Fold.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/gci-world-ml.git
cd gci-world-ml
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

Example `requirements.txt`:

```text
numpy
pandas
scikit-learn
scipy
xgboost
lightgbm
catboost
```

You can generate the file automatically from your environment with:

```bash
pip freeze > requirements.txt
```

However, for a competition repository, it is often cleaner to keep only the packages actually required by the project.

---

## ▶️ Running the Project

After installing the dependencies, place the competition data inside:

```text
data/raw/
```

Then run the training pipeline:

```bash
python src/train.py
```

The generated submission will be saved inside:

```text
submissions/
```

---

## 📈 Model Ensemble

| Model | Weight |
|------|------:|
| XGBoost | 45% |
| LightGBM | 40% |
| CatBoost | 15% |

The ensemble uses rank-based blending rather than simple probability averaging.

---

## 🎯 Key Takeaways

This project demonstrates several useful machine learning techniques:

- Feature engineering from domain-specific physical measurements
- Stratified K-Fold cross-validation
- Gradient boosting models
- Model diversity through heterogeneous ensembles
- Rank-based prediction blending
- Competition-oriented prediction pipelines

---

## 👤 Author

**Mario**

Machine Learning / Data Science Project

---

## 📄 License

This project is intended primarily for educational and competition purposes.