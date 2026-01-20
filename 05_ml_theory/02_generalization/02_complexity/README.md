<!-- Navigation -->
<p align="center">
  <a href="../01_bias_variance/">⬅️ Prev: Bias-Variance</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 Generalization</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../03_overfitting/">Next: Overfitting ➡️</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Model%20Complexity&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/complexity.svg" width="100%">

*Caption: Model complexity determines the capacity to fit data. VC dimension, Rademacher complexity, and parameter count are key measures.*

---

## 📂 Overview

**Model complexity** measures how flexible a hypothesis class is. Understanding complexity is essential for predicting generalization and avoiding overfitting.

---

## 📐 VC Dimension

### Definition

**VC Dimension** is the largest number of points that can be **shattered** by a hypothesis class $\mathcal{H}$.

**Shattering:** A set $S = \{x_1, \ldots, x_n\}$ is shattered by $\mathcal{H}$ if for every labeling $\mathbf{y} \in \{0,1\}^n$, there exists $h \in \mathcal{H}$ such that $h(x_i) = y_i$ for all $i$.

$$
\text{VC}(\mathcal{H}) = \max\{n : \exists S \text{ of size } n \text{ shattered by } \mathcal{H}\}
$$

### Common VC Dimensions

| Hypothesis Class | VC Dimension | Proof Sketch |
|-----------------|--------------|--------------|
| Linear classifiers in $\mathbb{R}^d$ | $d + 1$ | Can shatter $d+1$ points in general position |
| Decision stumps | 2 | Axis-aligned threshold |
| Polynomial of degree $p$ | $\binom{d+p}{p}$ | Number of terms |
| Neural net with $W$ weights | $O(W \log W)$ | Via covering numbers |
| $k$-NN with $k=1$ | $\infty$ | Can memorize any labeling |

---

## 📐 Rademacher Complexity

### Definition

**Empirical Rademacher Complexity:**

$$
\hat{\mathcal{R}}_n(\mathcal{H}) = \mathbb{E}_{\boldsymbol{\sigma}}\left[\sup_{h \in \mathcal{H}} \frac{1}{n}\sum_{i=1}^n \sigma_i h(x_i)\right]
$$

where $\sigma_i \in \{-1, +1\}$ are i.i.d. Rademacher random variables.

**Intuition:** How well can $\mathcal{H}$ fit random noise?

### Properties

1. **Data-dependent** (unlike VC dimension)
2. **Tighter bounds** in practice
3. **Composition-friendly:** Can bound complex function classes

### Rademacher Complexity Examples

| Class | Rademacher Complexity |
|-------|----------------------|
| Finite class $\|\mathcal{H}\| = M$ | $O(\sqrt{\log M / n})$ |
| Linear classifiers $\|w\| \leq B$ | $O(B/\sqrt{n})$ |
| Lipschitz functions | $O(1/\sqrt{n})$ |

---

## 📐 Generalization Bounds

### VC Bound

**Theorem:** For hypothesis class $\mathcal{H}$ with VC dimension $d$, with probability $\geq 1 - \delta$:

$$
R(h) \leq \hat{R}(h) + \sqrt{\frac{d(\log(2n/d) + 1) + \log(4/\delta)}{n}}
$$

**Proof Sketch:**
1. Use growth function $\Pi_\mathcal{H}(n) \leq (en/d)^d$ (Sauer's lemma)
2. Apply union bound over effective hypotheses
3. Use Hoeffding's inequality $\blacksquare$

### Rademacher Bound

**Theorem:** With probability $\geq 1 - \delta$:

$$
R(h) \leq \hat{R}(h) + 2\mathcal{R}_n(\mathcal{H}) + \sqrt{\frac{\log(1/\delta)}{2n}}
$$

**Remark:** This is often tighter than VC bounds for specific data distributions.

---

## 📐 Double Descent

### Modern Phenomenon

Classical theory predicts: more parameters → more overfitting

**Double Descent:** After the interpolation threshold, test error *decreases* again!

$$
\text{Test Error} = \begin{cases} \text{Classical U-curve} & \text{if params} < n \\ \text{Decrease} & \text{if params} \gg n \end{cases}
$$

### Theoretical Explanation

In the overparameterized regime:
- **Minimum-norm interpolation:** Among all interpolating solutions, SGD finds the one with minimum norm
- **Implicit regularization:** Simple structure despite high capacity

---

## 💻 Code Implementation

```python
import numpy as np
from itertools import product
from sklearn.linear_model import Perceptron

def can_shatter(X, model_class):
    """
    Check if model class can shatter a set of points.
    
    Shattering: for every labeling, exists h achieving it.
    """
    n = len(X)
    
    for labels in product([0, 1], repeat=n):
        y = np.array(labels)
        model = model_class()
        try:
            model.fit(X, y)
            pred = model.predict(X)
            if not np.allclose(pred, y):
                return False
        except:
            return False
    return True

def empirical_rademacher(H, X, n_samples=1000):
    """
    Estimate empirical Rademacher complexity.
    
    R̂_n(H) = E_σ[sup_h (1/n) Σ σᵢh(xᵢ)]
    """
    n = len(X)
    max_correlations = []
    
    for _ in range(n_samples):

        # Generate Rademacher variables
        sigma = np.random.choice([-1, 1], size=n)
        
        # Find supremum over H
        max_corr = -np.inf
        for h in H:
            correlation = np.mean(sigma * h(X))
            max_corr = max(max_corr, correlation)
        
        max_correlations.append(max_corr)
    
    return np.mean(max_correlations)

def vc_generalization_bound(vc_dim, n, delta=0.05):
    """
    VC dimension generalization bound.
    
    R(h) ≤ R̂(h) + √(d(log(2n/d) + 1) + log(4/δ)) / n)
    """
    complexity_term = vc_dim * (np.log(2 * n / vc_dim) + 1) + np.log(4 / delta)
    return np.sqrt(complexity_term / n)

def rademacher_bound(rademacher_complexity, n, delta=0.05):
    """
    Rademacher complexity bound.
    
    R(h) ≤ R̂(h) + 2R_n(H) + √(log(1/δ)/(2n))
    """
    return 2 * rademacher_complexity + np.sqrt(np.log(1/delta) / (2*n))

# Example: VC dimension of linear classifiers
print("VC Dimension Examples:")
print("-" * 40)

# 3 points in 2D - linear classifier can shatter
X_3points = np.array([[0, 0], [1, 0], [0, 1]])
print(f"3 points in 2D: Can shatter = {can_shatter(X_3points, Perceptron)}")

# 4 points in 2D - XOR configuration cannot be shattered
X_4points = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
print(f"4 points in 2D (XOR): Can shatter = {can_shatter(X_4points, Perceptron)}")

# Generalization bounds
print("\nGeneralization Bounds:")
print("-" * 40)
for n in [100, 1000, 10000]:
    vc = 10  # Assume VC dim = 10
    bound = vc_generalization_bound(vc, n)
    print(f"n={n:5d}, VC={vc}: bound = {bound:.4f}")
```

---

## 📊 Complexity Measures Comparison

| Measure | Depends on Data | Computation | Tightness |
|---------|-----------------|-------------|-----------|
| VC Dimension | No | Hard (NP-hard in general) | Loose |
| Rademacher | Yes | Moderate | Tight |
| Parameter Count | No | Easy | Very loose |
| PAC-Bayes | Yes | Moderate | Tight |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Understanding ML (Shalev-Shwartz) | [Book](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) |
| 📄 | Rademacher Complexity | [Bartlett & Mendelson](https://www.jmlr.org/papers/v3/bartlett02a.html) |
| 📄 | Double Descent | [Belkin et al.](https://arxiv.org/abs/1912.02292) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../01_bias_variance/">⬅️ Prev: Bias-Variance</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 Generalization</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../03_overfitting/">Next: Overfitting ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
