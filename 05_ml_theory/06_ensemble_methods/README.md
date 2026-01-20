<!-- Navigation -->
<p align="center">
  <a href="../05_risk_minimization/">⬅️ Prev: Risk Minimization</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 ML Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../07_clustering/">Next: Clustering ➡️</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Ensemble%20Methods&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/ensemble-methods-complete.svg" width="100%">

*Caption: Ensemble methods combine multiple models for better performance. Bagging reduces variance, Boosting reduces bias.*

---

## 📐 Mathematical Foundations

### Why Ensembles Work

**Theorem (Condorcet's Jury):** If each of \(n\) classifiers has accuracy \(p > 0.5\), the majority vote accuracy approaches 1 as \(n \to \infty\).

**Bias-Variance for Ensembles:**

For averaged predictions \(\bar{f}(x) = \frac{1}{B}\sum_{b=1}^B f_b(x)\):

```math
\text{Var}(\bar{f}) = \frac{1}{B^2}\sum_{b=1}^B \text{Var}(f_b) + \frac{1}{B^2}\sum_{b \neq b'} \text{Cov}(f_b, f_{b'})

```

If models are independent with equal variance \(\sigma^2\):

```math
\text{Var}(\bar{f}) = \frac{\sigma^2}{B}

```

**Key insight:** Averaging reduces variance by factor of \(B\)!

---

## 📊 Bagging (Bootstrap Aggregating)

### Algorithm

1. Generate \(B\) bootstrap samples \(\mathcal{D}_1, \ldots, \mathcal{D}_B\)
2. Train model \(f_b\) on each \(\mathcal{D}_b\)
3. Aggregate:
   - Regression: \(\hat{f}(x) = \frac{1}{B}\sum_{b=1}^B f_b(x)\)
   - Classification: \(\hat{f}(x) = \text{mode}(\{f_b(x)\}_{b=1}^B)\)

### Variance Reduction Theorem

**Theorem:** For base learners with variance \(\sigma^2\) and pairwise correlation \(\rho\):

```math
\text{Var}(\bar{f}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2

```

**Proof:**

```math
\text{Var}\left(\frac{1}{B}\sum_b f_b\right) = \frac{1}{B^2}\left[B\sigma^2 + B(B-1)\rho\sigma^2\right] = \rho\sigma^2 + \frac{(1-\rho)\sigma^2}{B}

```

As \(B \to \infty\): \(\text{Var}(\bar{f}) \to \rho\sigma^2\). \(\blacksquare\)

### Random Forest

Adds feature randomization to reduce \(\rho\):
- At each split, consider only \(\sqrt{d}\) random features
- Reduces correlation between trees

---

## 📊 Boosting

### AdaBoost Algorithm

**Input:** Training data \(\{(x_i, y_i)\}_{i=1}^n\), \(y_i \in \{-1, +1\}\), \(T\) iterations

**Initialize:** \(w_i^{(1)} = \frac{1}{n}\)

**For \(t = 1, \ldots, T\):**

1. Train weak learner \(h_t\) on weighted data
2. Compute weighted error:

```math
\epsilon_t = \sum_{i=1}^n w_i^{(t)} \mathbb{1}[h_t(x_i) \neq y_i]

```

3. Compute learner weight:

```math
\alpha_t = \frac{1}{2}\ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right)

```

4. Update sample weights:

```math
w_i^{(t+1)} = \frac{w_i^{(t)} \exp(-\alpha_t y_i h_t(x_i))}{Z_t}

```

**Output:** \(H(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t h_t(x)\right)\)

### AdaBoost Training Error Bound

**Theorem:** The training error satisfies:

```math
\frac{1}{n}\sum_{i=1}^n \mathbb{1}[H(x_i) \neq y_i] \leq \prod_{t=1}^T 2\sqrt{\epsilon_t(1-\epsilon_t)}

```

**Proof:** Using \(\exp(-y_i H(x_i)) \geq \mathbb{1}[H(x_i) \neq y_i]\) and telescoping. \(\blacksquare\)

---

## 📊 Gradient Boosting

### Framework

Build model additively:

```math
F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)

```

where \(h_m\) fits the negative gradient (pseudo-residuals):

```math
r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}

```

### For Different Losses

| Loss | Pseudo-Residual |
|------|-----------------|
| **MSE** \((y-F)^2\) | \(y - F_{m-1}(x)\) |
| **Absolute** \(\|y-F\|\) | \(\text{sign}(y - F_{m-1}(x))\) |
| **Logistic** | \(y - \sigma(F_{m-1}(x))\) |

### XGBoost Objective

```math
\mathcal{L}^{(t)} = \sum_{i=1}^n \left[\ell(y_i, \hat{y}_i^{(t-1)} + f_t(x_i))\right] + \Omega(f_t)

```

Second-order Taylor approximation:

```math
\approx \sum_i [g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)] + \Omega(f_t)

```

where \(g_i = \partial_{\hat{y}} \ell\) and \(h_i = \partial^2_{\hat{y}} \ell\).

---

## 💻 Code Implementation

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    """
    AdaBoost classifier.
    
    H(x) = sign(Σ_t α_t h_t(x))
    """
    
    def __init__(self, n_estimators=50, max_depth=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.alphas = []
        self.models = []
    
    def fit(self, X, y):
        n = len(X)
        w = np.ones(n) / n  # Initialize weights
        
        for t in range(self.n_estimators):
            # Train weak learner
            model = DecisionTreeClassifier(max_depth=self.max_depth)
            model.fit(X, y, sample_weight=w)
            pred = model.predict(X)
            
            # Compute weighted error
            error = np.sum(w * (pred != y)) / np.sum(w)
            error = np.clip(error, 1e-10, 1 - 1e-10)
            
            # Compute alpha
            alpha = 0.5 * np.log((1 - error) / error)
            
            # Update weights
            w = w * np.exp(-alpha * y * pred)
            w = w / np.sum(w)
            
            self.models.append(model)
            self.alphas.append(alpha)
        
        return self
    
    def predict(self, X):
        predictions = np.zeros(len(X))
        for alpha, model in zip(self.alphas, self.models):
            predictions += alpha * model.predict(X)
        return np.sign(predictions)

class GradientBoostingRegressor:
    """
    Gradient Boosting for regression (MSE loss).
    
    F_m(x) = F_{m-1}(x) + γ h_m(x)
    where h_m fits negative gradient (residuals for MSE).
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
    
    def fit(self, X, y):
        # Initialize with mean
        self.init_pred = np.mean(y)
        F = np.full(len(y), self.init_pred)
        
        for _ in range(self.n_estimators):
            # Compute pseudo-residuals (negative gradient for MSE)
            residuals = y - F
            
            # Fit tree to residuals
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            # For regression, we'd use DecisionTreeRegressor
            # Using simple implementation here
            tree.fit(X, np.sign(residuals))
            
            # Update predictions
            F = F + self.learning_rate * tree.predict(X)
            self.models.append(tree)
        
        return self
    
    def predict(self, X):
        pred = np.full(len(X), self.init_pred)
        for tree in self.models:
            pred += self.learning_rate * tree.predict(X)
        return pred

class RandomForest:
    """
    Random Forest classifier.
    
    Bagging + Random feature selection at each split.
    """
    
    def __init__(self, n_estimators=100, max_features='sqrt', max_depth=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.trees = []
    
    def fit(self, X, y):
        n, d = X.shape
        
        # Determine number of features to consider
        if self.max_features == 'sqrt':
            n_features = int(np.sqrt(d))
        else:
            n_features = d
        
        for _ in range(self.n_estimators):
            # Bootstrap sample
            idx = np.random.choice(n, n, replace=True)
            X_boot, y_boot = X[idx], y[idx]
            
            # Random feature subset (done at each split in real implementation)
            # Here we just train a tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=n_features
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Majority vote
        return np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(),
            axis=0,
            arr=predictions
        )
    
    def feature_importance(self):
        importances = np.zeros(self.trees[0].n_features_in_)
        for tree in self.trees:
            importances += tree.feature_importances_
        return importances / self.n_estimators

```

---

## 📊 Method Comparison

| Method | Reduces | Training | Parallelizable |
|--------|---------|----------|----------------|
| **Bagging** | Variance | Parallel | Yes |
| **Boosting** | Bias | Sequential | Limited |
| **Stacking** | Both | Multi-level | Partially |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Random Forest | [Breiman, 2001](https://link.springer.com/article/10.1023/A:1010933404324) |
| 📄 | XGBoost | [Chen & Guestrin, 2016](https://arxiv.org/abs/1603.02754) |
| 📄 | AdaBoost | [Freund & Schapire, 1997](https://link.springer.com/article/10.1023/A:1007614523901) |

---

⬅️ [Back: Risk Minimization](../05_risk_minimization/) | ➡️ [Next: Clustering](../07_clustering/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../05_risk_minimization/">⬅️ Prev: Risk Minimization</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 ML Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../07_clustering/">Next: Clustering ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
