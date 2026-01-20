<!-- Navigation -->
<p align="center">
  <a href="../07_clustering/">⬅️ Prev: Clustering</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 ML Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../09_hyperparameter_tuning/">Next: Hyperparameter Tuning ➡️</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Model%20Selection&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/cross-validation-complete.svg" width="100%">

*Caption: Cross-validation provides robust model evaluation by training on different data splits. k-Fold is most common, stratified preserves class ratios, time series respects temporal order.*

---

## 📐 Mathematical Foundations

### Cross-Validation

```
k-Fold Cross-Validation:

1. Split data into k equal folds

2. For each fold i:
   • Train on all folds except i
   • Evaluate on fold i

3. Average scores across all folds

CV Score = (1/k) Σᵢ Score(foldᵢ)
CV Std = √((1/k) Σᵢ (Scoreᵢ - CV Score)²)

Common choices: k = 5 or k = 10

```

### Information Criteria

```
AIC (Akaike Information Criterion):
AIC = 2k - 2ln(L̂)
where k = number of parameters, L̂ = max likelihood

BIC (Bayesian Information Criterion):
BIC = k·ln(n) - 2ln(L̂)

Lower is better. BIC penalizes complexity more.

```

### Bias-Variance in Model Selection

```
Expected Error = Bias² + Variance + Noise

Simple model: High bias, low variance
Complex model: Low bias, high variance

Goal: Find the sweet spot (optimal complexity)

```

---

## 🎯 Validation Strategies

| Strategy | Use Case | How It Works |
|----------|----------|--------------|
| **k-Fold** | General | Rotate through k folds |
| **Stratified k-Fold** | Imbalanced | Preserve class ratios |
| **Leave-One-Out** | Small data | k = n |
| **Time Series Split** | Temporal data | Train on past, test on future |
| **Group k-Fold** | Grouped data | Keep groups together |
| **Holdout** | Large data | Simple train/test split |

---

## 💻 Code Examples

```python
import numpy as np
from sklearn.model_selection import (
    cross_val_score, KFold, StratifiedKFold, 
    TimeSeriesSplit, GridSearchCV, RandomizedSearchCV,
    train_test_split, learning_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Create dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Simple cross-validation
model = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# Stratified k-Fold (for classification)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    print(f"Fold {fold+1}: {score:.4f}")

# Time Series Split
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

# Grid Search with CV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X, y)
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10)
)
print(f"Learning curve computed with {len(train_sizes)} points")

```

---

## 🌍 ML Applications

| Application | Strategy | Why |
|-------------|----------|-----|
| **General ML** | 5-Fold CV | Balance between bias and variance |
| **Imbalanced Data** | Stratified CV | Preserve class distribution |
| **Time Series** | Time Series Split | Respect temporal order |
| **Small Dataset** | LOOCV | Maximum use of data |
| **Hyperparameter Tuning** | Nested CV | Unbiased evaluation |

---

## 📊 Common Mistakes

```
❌ Using test data for model selection
   → Data leakage, overoptimistic results

❌ Not stratifying with imbalanced data
   → Folds may have no positive samples

❌ Random splits for time series
   → Future information leaks to training

❌ Single train/test split for small data
   → High variance in evaluation

✅ Use nested CV for hyperparameter tuning
✅ Report mean ± std of CV scores
✅ Keep test set completely hidden until final evaluation

```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Cross-Validation Guide | [Scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html) |
| 📄 | Nested CV Paper | [Paper](https://jmlr.org/papers/v11/cawley10a.html) |
| 🎥 | Model Selection Explained | [YouTube](https://www.youtube.com/watch?v=fSytzGwwBVw) |
| 🇨🇳 | 交叉验证详解 | [知乎](https://zhuanlan.zhihu.com/p/24825503) |
| 🇨🇳 | 模型选择策略 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88333333) |

---

## 🔗 Where This Topic Is Used

| Application | How Model Selection Is Used |
|-------------|----------------------------|
| **ML Pipeline** | Validate before deployment |
| **AutoML** | Automated model selection |
| **Hyperparameter Tuning** | Find optimal settings |
| **Model Comparison** | Fair comparison across models |
| **Competition** | Local validation strategy |

---

⬅️ [Back: 07-Clustering](../07_clustering/) | ➡️ [Next: 09-Hyperparameter Tuning](../09_hyperparameter_tuning/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../07_clustering/">⬅️ Prev: Clustering</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 ML Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../09_hyperparameter_tuning/">Next: Hyperparameter Tuning ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
