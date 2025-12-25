<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Ensemble%20Methods&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/ensemble-methods-complete.svg" width="100%">

*Caption: Ensemble methods combine multiple weak learners to create a strong learner. Bagging reduces variance (Random Forest), Boosting reduces bias (XGBoost), Stacking learns optimal combination.*

---

## 📐 Mathematical Foundations

### Bagging (Bootstrap Aggregating)

```
Algorithm:
1. Create B bootstrap samples from training data
2. Train model fᵢ on each sample
3. Aggregate predictions:
   • Regression: f(x) = (1/B) Σᵢ fᵢ(x)
   • Classification: f(x) = mode(f₁(x), ..., f_B(x))

Variance reduction:
Var(f̄) = σ²/B  (if models independent)

Example: Random Forest = Bagging + Random feature subsets
```

### Boosting

```
AdaBoost:
1. Initialize weights: wᵢ = 1/n
2. For t = 1 to T:
   a. Train weak learner hₜ on weighted data
   b. Compute error: εₜ = Σᵢ wᵢ 𝟙[hₜ(xᵢ) ≠ yᵢ]
   c. Compute weight: αₜ = ½ log((1-εₜ)/εₜ)
   d. Update weights: wᵢ ← wᵢ exp(-αₜ yᵢ hₜ(xᵢ))
3. Final: H(x) = sign(Σₜ αₜ hₜ(x))

Gradient Boosting:
fₘ(x) = fₘ₋₁(x) + γₘ hₘ(x)
where hₘ fits the negative gradient (pseudo-residuals)
```

### Stacking

```
Level 0: Train base models {f₁, ..., fₖ}
Level 1: Train meta-model on base model predictions

Training:
1. Split data into folds
2. Generate out-of-fold predictions from base models
3. Train meta-model on [f₁(x), ..., fₖ(x)] → y

Prediction:
ŷ = g(f₁(x), ..., fₖ(x))
where g is the meta-learner
```

---

## 🎯 Method Comparison

| Method | Reduces | Training | Base Learners |
|--------|---------|----------|---------------|
| **Bagging** | Variance | Parallel | Independent |
| **Boosting** | Bias | Sequential | Dependent |
| **Stacking** | Both | Multi-level | Any |

---

## 💻 Code Examples

```python
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, BaggingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

# Create dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Random Forest (Bagging + Random Features)
rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    random_state=42
)
rf.fit(X, y)
print(f"Random Forest Accuracy: {rf.score(X, y):.4f}")

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X, y)
print(f"Gradient Boosting Accuracy: {gb.score(X, y):.4f}")

# XGBoost
xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
xgb.fit(X, y)
print(f"XGBoost Accuracy: {xgb.score(X, y):.4f}")

# AdaBoost
ada = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
ada.fit(X, y)
print(f"AdaBoost Accuracy: {ada.score(X, y):.4f}")

# Stacking
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=50)),
    ('gb', GradientBoostingClassifier(n_estimators=50)),
]
stacking = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(),
    cv=5
)
stacking.fit(X, y)
print(f"Stacking Accuracy: {stacking.score(X, y):.4f}")

# Feature importance from Random Forest
importances = rf.feature_importances_
print(f"Top 5 features: {np.argsort(importances)[-5:][::-1]}")
```

---

## 🌍 ML Applications

| Application | Method | Why |
|-------------|--------|-----|
| **Kaggle Competitions** | XGBoost, Stacking | Top performance |
| **Tabular Data** | Random Forest, GBM | Robust, little tuning |
| **Anomaly Detection** | Isolation Forest | Ensemble of trees |
| **Ranking** | LambdaMART (GBM) | Learning to rank |
| **Production ML** | Ensemble voting | Robustness |

---

## 📊 Popular Implementations

| Library | Algorithm | Strengths |
|---------|-----------|-----------|
| **XGBoost** | Gradient Boosting | Speed, regularization |
| **LightGBM** | Gradient Boosting | Large data, speed |
| **CatBoost** | Gradient Boosting | Categorical features |
| **Scikit-learn** | RF, AdaBoost, GB | Simplicity |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Random Forest | [Paper](https://link.springer.com/article/10.1023/A:1010933404324) |
| 📄 | XGBoost | [Paper](https://arxiv.org/abs/1603.02754) |
| 📄 | AdaBoost | [Paper](https://link.springer.com/article/10.1023/A:1007614523901) |
| 🇨🇳 | 集成学习详解 | [知乎](https://zhuanlan.zhihu.com/p/27689464) |
| 🇨🇳 | XGBoost原理 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88555555) |

---

## 🔗 Where This Topic Is Used

| Application | How Ensemble Methods Are Used |
|-------------|------------------------------|
| **Tabular ML** | XGBoost, LightGBM dominate |
| **Kaggle Competitions** | Stacking for top scores |
| **Production Systems** | Robust predictions |
| **AutoML** | Ensemble selection |
| **Neural Networks** | Model averaging, distillation |

---

⬅️ [Back: 05-Risk Minimization](../05-risk-minimization/) | ➡️ [Next: 07-Clustering](../07-clustering/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
