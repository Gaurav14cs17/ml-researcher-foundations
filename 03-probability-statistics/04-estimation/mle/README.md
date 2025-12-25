<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Maximum%20Likelihood%20Estimation&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/mle.svg" width="100%">

*Caption: MLE finds the parameter θ that makes the observed data most probable. The likelihood function L(θ) = P(data|θ) is maximized, often using log-likelihood for numerical stability.*

---

## 📂 Overview

MLE is the workhorse of modern ML. When you minimize cross-entropy or MSE, you're doing MLE. Understanding this connection unifies many seemingly different training objectives.

---

## 📐 Mathematical Framework

### Definition

```
θ_MLE = argmax_θ P(D|θ)
      = argmax_θ ∏ᵢ P(xᵢ|θ)      (i.i.d. assumption)
      = argmax_θ Σᵢ log P(xᵢ|θ)  (log-likelihood)
```

### Why Log-Likelihood?

```
1. Products → Sums (numerical stability)
2. Avoids underflow for large datasets
3. Preserves argmax (log is monotonic)
4. Gradient is simpler to compute

L(θ) = ∏ᵢ P(xᵢ|θ)     →  very small numbers!
ℓ(θ) = Σᵢ log P(xᵢ|θ)  →  manageable sums
```

---

## 🎯 Common MLE Examples

### 1. Gaussian Distribution

```
Data: x₁, x₂, ..., xₙ ~ N(μ, σ²)

Log-likelihood:
ℓ(μ, σ²) = -n/2 log(2πσ²) - 1/(2σ²) Σᵢ(xᵢ - μ)²

MLE solutions:
μ_MLE = (1/n) Σᵢ xᵢ         ← Sample mean
σ²_MLE = (1/n) Σᵢ(xᵢ - μ)²  ← Sample variance (biased)
```

### 2. Bernoulli Distribution

```
Data: x₁, ..., xₙ ∈ {0, 1}  (k ones, n-k zeros)

Likelihood:
L(p) = p^k (1-p)^(n-k)

Log-likelihood:
ℓ(p) = k log(p) + (n-k) log(1-p)

MLE solution:
p_MLE = k/n  ← Sample proportion
```

### 3. Linear Regression as MLE

```
y = Xw + ε,  ε ~ N(0, σ²)

P(y|X,w,σ²) = N(y; Xw, σ²I)

Log-likelihood:
ℓ(w) ∝ -1/(2σ²) ||y - Xw||²

Maximizing ℓ(w) = Minimizing ||y - Xw||²
                = Least Squares!
```

### 4. Classification as MLE

```
y ∈ {0, 1}, P(y=1|x) = σ(w·x)

Log-likelihood (Binary):
ℓ(w) = Σᵢ [yᵢ log σ(w·xᵢ) + (1-yᵢ) log(1-σ(w·xᵢ))]
     = -Cross_Entropy(y, ŷ)

Minimizing Cross-Entropy = Maximizing Likelihood!
```

---

## 🔑 Properties of MLE

| Property | Description |
|----------|-------------|
| **Consistency** | θ̂_MLE → θ_true as n → ∞ |
| **Asymptotic Normality** | √n(θ̂ - θ) → N(0, I⁻¹(θ)) |
| **Efficiency** | Achieves Cramér-Rao lower bound |
| **Invariance** | g(θ̂_MLE) = MLE of g(θ) |

### Fisher Information

```
I(θ) = -E[∂²ℓ/∂θ²]  ← "Curvature" of log-likelihood

Var(θ̂_MLE) ≈ 1/I(θ)  ← Sharper peak = lower variance
```

---

## 💻 Code Examples

### NumPy MLE for Gaussian

```python
import numpy as np
from scipy.optimize import minimize

def gaussian_log_likelihood(params, data):
    mu, sigma = params
    n = len(data)
    ll = -n/2 * np.log(2*np.pi*sigma**2) - \
         np.sum((data - mu)**2) / (2*sigma**2)
    return -ll  # Minimize negative log-likelihood

# Generate data
data = np.random.normal(loc=5, scale=2, size=1000)

# Find MLE
result = minimize(gaussian_log_likelihood, x0=[0, 1], args=(data,))
mu_mle, sigma_mle = result.x
print(f"μ_MLE = {mu_mle:.3f}, σ_MLE = {sigma_mle:.3f}")
```

### PyTorch MLE (Classification)

```python
import torch
import torch.nn.functional as F

def cross_entropy_loss(logits, targets):
    """Cross-entropy = Negative log-likelihood"""
    return F.cross_entropy(logits, targets)

# Training loop is MLE!
for X, y in dataloader:
    logits = model(X)
    loss = cross_entropy_loss(logits, y)  # -log P(y|X,θ)
    loss.backward()
    optimizer.step()
```

---

## 🔗 Connection to Other Estimators

```
              Prior
                |
    MLE --------+--------> MAP (Maximum A Posteriori)
   P(D|θ)       |         P(θ|D) ∝ P(D|θ) × P(θ)
                |
                v
        Bayesian Inference
        P(θ|D) = ∫ ...
```

### MLE vs MAP vs Bayesian

| Method | Formula | Prior? | Point Estimate? |
|--------|---------|--------|-----------------|
| MLE | argmax P(D\|θ) | No | Yes |
| MAP | argmax P(θ\|D) | Yes | Yes |
| Bayesian | P(θ\|D) | Yes | No (distribution) |

---

## ⚠️ Limitations of MLE

| Issue | Example | Solution |
|-------|---------|----------|
| **Overfitting** | Perfect fit to training data | Regularization (→ MAP) |
| **Unbounded** | σ_MLE = 0 when xᵢ = μ for all i | Add prior on σ |
| **Local optima** | Non-convex likelihood | Multiple restarts |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Bayesian Estimation | [../bayesian/](../bayesian/) |
| 📖 | MAP Estimation | [../map/](../map/) |
| 🎥 | StatQuest: MLE | [YouTube](https://www.youtube.com/watch?v=XepXtl9YKwc) |
| 📄 | Bishop PRML Ch. 2 | Pattern Recognition and Machine Learning |
| 🇨🇳 | 最大似然估计详解 | [知乎](https://zhuanlan.zhihu.com/p/26614750) |
| 🇨🇳 | MLE与交叉熵的关系 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88393322) |
| 🇨🇳 | 极大似然估计推导 | [B站](https://www.bilibili.com/video/BV1aE411o7qd) |
| 🇨🇳 | 似然函数的理解 | [机器之心](https://www.jiqizhixin.com/)


## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

⬅️ [Back: Estimation](../)

---

⬅️ [Back: Map](../map/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
