<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Hyperparameter%20Tuning&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/hyperparameter-tuning-complete.svg" width="100%">

*Caption: Hyperparameter tuning searches for optimal model settings. Grid search is exhaustive, Random search is efficient for high dimensions, Bayesian optimization learns from previous evaluations.*

---

## 📐 Mathematical Foundations

### Search Strategies

```
Grid Search:
• Exhaustive search over parameter grid
• Complexity: O(∏ᵢ |paramᵢ|)
• Best for: Few parameters, small ranges

Random Search:
• Sample randomly from parameter distributions
• More efficient for high dimensions
• Same budget often finds better solutions

Bayesian Optimization:
• Model f(params) → score with GP/TPE
• Use acquisition function to balance explore/exploit
• Most sample-efficient for expensive evaluations
```

### Bayesian Optimization

```
Algorithm:
1. Initialize with random evaluations
2. Fit surrogate model (GP) to (params, scores)
3. Select next params using acquisition function:
   • EI (Expected Improvement)
   • UCB (Upper Confidence Bound)
4. Evaluate and update
5. Repeat until budget exhausted

Acquisition functions:
EI(x) = E[max(f(x) - f(x⁺), 0)]
UCB(x) = μ(x) + κσ(x)
```

---

## 🎯 Method Comparison

| Method | Efficiency | Parallelizable | Best For |
|--------|------------|----------------|----------|
| **Grid Search** | Low | Yes | Few params |
| **Random Search** | Medium | Yes | Many params |
| **Bayesian** | High | Limited | Expensive evals |
| **Hyperband** | High | Yes | Deep learning |
| **Population-based** | High | Yes | Online tuning |

---

## 💻 Code Examples

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score
)
from sklearn.datasets import make_classification
from scipy.stats import randint, uniform

# Create dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X, y)
print(f"Grid Search Best: {grid_search.best_score_:.4f}")
print(f"Best params: {grid_search.best_params_}")

# Random Search (more efficient)
param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 30),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=100,  # Number of random combinations
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X, y)
print(f"Random Search Best: {random_search.best_score_:.4f}")

# Optuna (Bayesian Optimization)
try:
    import optuna
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        }
        
        model = RandomForestClassifier(**params, random_state=42)
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    print(f"Optuna Best: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
except ImportError:
    print("Optuna not installed")

# Learning Rate Schedule Search (Deep Learning)
# Example with PyTorch Lightning / Keras
learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
batch_sizes = [16, 32, 64, 128]
```

---

## 🌍 ML Applications

| Application | Strategy | Why |
|-------------|----------|-----|
| **Tabular ML** | Random Search | Many hyperparams |
| **Deep Learning** | Hyperband, PBT | Expensive training |
| **AutoML** | Bayesian + Ensemble | End-to-end optimization |
| **Neural Architecture** | NAS + HPO | Combined search |
| **Production** | Online tuning | Adapt to data drift |

---

## 📊 Common Hyperparameters

| Model | Key Hyperparameters |
|-------|---------------------|
| **Random Forest** | n_estimators, max_depth, min_samples_split |
| **XGBoost** | learning_rate, n_estimators, max_depth, subsample |
| **Neural Network** | learning_rate, batch_size, hidden_size, dropout |
| **SVM** | C, gamma (RBF), kernel |
| **Transformer** | lr, warmup, layers, heads, dropout |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Random Search Paper | [Paper](https://jmlr.org/papers/v13/bergstra12a.html) |
| 📄 | Hyperband | [Paper](https://arxiv.org/abs/1603.06560) |
| 📖 | Optuna | [Docs](https://optuna.readthedocs.io/) |
| 🇨🇳 | 超参数调优详解 | [知乎](https://zhuanlan.zhihu.com/p/29820676) |
| 🇨🇳 | 贝叶斯优化 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88222222) |

---

## 🔗 Where This Topic Is Used

| Application | How HPO Is Used |
|-------------|-----------------|
| **Model Training** | Find optimal learning rate, batch size |
| **AutoML** | Automated pipeline optimization |
| **Neural Architecture Search** | Combined with architecture search |
| **Production ML** | Continuous optimization |
| **Kaggle** | Competition edge |

---

⬅️ [Back: 08-Model Selection](../08-model-selection/) | ➡️ [Next: 10-Interpretability](../10-interpretability/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
