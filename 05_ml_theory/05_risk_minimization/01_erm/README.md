<!-- Navigation -->
<p align="center">
  <a href="../">⬅️ Back: Risk Minimization</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../02_pac_learning/">Next: PAC Learning ➡️</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Empirical%20Risk%20Minimization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/erm.svg" width="100%">

*Caption: ERM finds the hypothesis that minimizes average loss on training data. It's the foundation of supervised learning.*

---

## 📂 Overview

**Empirical Risk Minimization (ERM)** is the most fundamental principle in machine learning. We approximate the unknown true risk (expected loss) with the empirical risk (training loss) and minimize it.

---

## 📐 Mathematical Definitions

### True (Population) Risk

```math
R(h) = \mathbb{E}_{(x,y) \sim \mathcal{D}}[\ell(h(x), y)] = \int \ell(h(x), y) \, dP(x, y)

```

where:
- \(h: \mathcal{X} \to \mathcal{Y}\) is a hypothesis
- \(\mathcal{D}\) is the unknown data distribution
- \(\ell: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_{\geq 0}\) is the loss function

**Problem:** We cannot compute \(R(h)\) because \(\mathcal{D}\) is unknown!

### Empirical Risk

Given training set \(S = \{(x_i, y_i)\}_{i=1}^n\) sampled i.i.d. from \(\mathcal{D}\):

```math
\hat{R}_S(h) = \frac{1}{n} \sum_{i=1}^{n} \ell(h(x_i), y_i)

```

This is the average loss over training data—computable!

### ERM Principle

```math
\hat{h}_{\text{ERM}} = \arg\min_{h \in \mathcal{H}} \hat{R}_S(h)

```

Find the hypothesis in class \(\mathcal{H}\) that minimizes training loss.

---

## 📐 Theoretical Foundations

### Law of Large Numbers

**Theorem:** For fixed \(h\), as \(n \to \infty\):

```math
\hat{R}_S(h) \xrightarrow{p} R(h)

```

**Proof:** By LLN, since \(\ell(h(x_i), y_i)\) are i.i.d.:

```math
\frac{1}{n}\sum_{i=1}^n \ell(h(x_i), y_i) \xrightarrow{p} \mathbb{E}[\ell(h(x), y)] = R(h) \quad \blacksquare

```

### Hoeffding's Inequality (Concentration)

**Theorem:** For bounded loss \(\ell \in [0, 1]\), for any fixed \(h\):

```math
\Pr\left[|R(h) - \hat{R}_S(h)| > \epsilon\right] \leq 2\exp(-2n\epsilon^2)

```

**Corollary:** With probability \(\geq 1 - \delta\):

```math
|R(h) - \hat{R}_S(h)| \leq \sqrt{\frac{\ln(2/\delta)}{2n}}

```

### Uniform Convergence

**Theorem (Finite Hypothesis Class):** For \(|\mathcal{H}| < \infty\), with probability \(\geq 1 - \delta\):

```math
\sup_{h \in \mathcal{H}} |R(h) - \hat{R}_S(h)| \leq \sqrt{\frac{\ln(2|\mathcal{H}|/\delta)}{2n}}

```

**Proof:** Apply union bound:

```math
\Pr\left[\exists h: |R(h) - \hat{R}_S(h)| > \epsilon\right] \leq \sum_{h \in \mathcal{H}} \Pr\left[|R(h) - \hat{R}_S(h)| > \epsilon\right] \leq |\mathcal{H}| \cdot 2e^{-2n\epsilon^2}

```

Setting this to \(\delta\) and solving for \(\epsilon\). \(\blacksquare\)

---

## 📐 ERM Generalization Bound

### Main Theorem

**Theorem:** Let \(\mathcal{H}\) be a hypothesis class with VC dimension \(d\). With probability \(\geq 1 - \delta\):

```math
R(\hat{h}_{\text{ERM}}) \leq \hat{R}_S(\hat{h}_{\text{ERM}}) + \sqrt{\frac{8d\ln(en/d) + 8\ln(4/\delta)}{n}}

```

### Decomposition of Excess Risk

```math
R(\hat{h}_{\text{ERM}}) - R(h^*) = \underbrace{R(\hat{h}_{\text{ERM}}) - \hat{R}_S(\hat{h}_{\text{ERM}})}_{\text{generalization gap}} + \underbrace{\hat{R}_S(\hat{h}_{\text{ERM}}) - \hat{R}_S(h^*)}_{\leq 0 \text{ by ERM}} + \underbrace{\hat{R}_S(h^*) - R(h^*)}_{\text{approximation}}

```

where \(h^* = \arg\min_{h \in \mathcal{H}} R(h)\).

---

## 📐 Overfitting Analysis

### The Problem

ERM minimizes \(\hat{R}_S(h)\), not \(R(h)\). The gap:

```math
R(\hat{h}) = \underbrace{\hat{R}_S(\hat{h})}_{\text{training error}} + \underbrace{(R(\hat{h}) - \hat{R}_S(\hat{h}))}_{\text{generalization gap}}

```

### When ERM Overfits

| Condition | Result |
|-----------|--------|
| \(n\) too small | Large generalization gap |
| \(\mathcal{H}\) too complex | Can fit noise |
| Training error ≈ 0 | High variance |

### Solutions

| Method | Modification |
|--------|-------------|
| **Regularized ERM** | \(\min \hat{R}_S(h) + \lambda\Omega(h)\) |
| **Early stopping** | Stop optimization early |
| **Cross-validation** | Estimate true risk |
| **More data** | Tighter bounds |

---

## 💻 Code Implementation

```python
import numpy as np
import torch
import torch.nn as nn

class ERM:
    """
    Empirical Risk Minimization framework.
    
    Principle: h* = argmin_h (1/n) Σᵢ L(h(xᵢ), yᵢ)
    """
    
    @staticmethod
    def true_risk(model, data_distribution, loss_fn, n_samples=10000):
        """
        Approximate true risk via Monte Carlo.
        
        R(h) = E_{(x,y)~D}[L(h(x), y)] ≈ (1/n) Σᵢ L(h(xᵢ), yᵢ)
        
        Note: In practice, we can only approximate this with a test set.
        """
        X, y = data_distribution.sample(n_samples)
        with torch.no_grad():
            predictions = model(X)
            return loss_fn(predictions, y).item()
    
    @staticmethod
    def empirical_risk(model, X, y, loss_fn):
        """
        Compute empirical risk.
        
        R̂(h) = (1/n) Σᵢ L(h(xᵢ), yᵢ)
        """
        with torch.no_grad():
            predictions = model(X)
            return loss_fn(predictions, y).item()
    
    @staticmethod
    def train_erm(model, train_loader, loss_fn, optimizer, epochs):
        """
        Standard ERM training loop.
        
        At each step, we minimize:
        θ_{t+1} = θ_t - η ∇_θ R̂(h_θ)
        """
        model.train()
        history = {'train_loss': []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                predictions = model(X_batch)
                
                # Compute empirical risk on batch
                loss = loss_fn(predictions, y_batch)
                
                # Backward pass (gradient of empirical risk)
                loss.backward()
                
                # Update parameters (minimize empirical risk)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            history['train_loss'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: R̂(h) = {avg_loss:.6f}")
        
        return history

def generalization_bound(n, vc_dim, delta=0.05):
    """
    Compute VC generalization bound.
    
    With probability ≥ 1-δ:
    R(h) ≤ R̂(h) + √((8d·ln(en/d) + 8·ln(4/δ)) / n)
    """
    if n < vc_dim:
        return float('inf')
    
    complexity_term = 8 * vc_dim * np.log(np.e * n / vc_dim)
    confidence_term = 8 * np.log(4 / delta)
    
    return np.sqrt((complexity_term + confidence_term) / n)

def hoeffding_bound(n, delta=0.05):
    """
    Single hypothesis Hoeffding bound.
    
    With probability ≥ 1-δ:
    |R(h) - R̂(h)| ≤ √(ln(2/δ) / (2n))
    """
    return np.sqrt(np.log(2 / delta) / (2 * n))

# Example: Demonstrating ERM and generalization
if __name__ == "__main__":
    print("=== Empirical Risk Minimization ===\n")
    
    # Generate data
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_train = 100
    n_test = 1000
    d = 10
    
    # True model: y = Xw* + noise
    w_true = np.random.randn(d)
    X_train = np.random.randn(n_train, d)
    y_train = X_train @ w_true + 0.1 * np.random.randn(n_train)
    
    X_test = np.random.randn(n_test, d)
    y_test = X_test @ w_true + 0.1 * np.random.randn(n_test)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Linear model (ERM)
    model = nn.Linear(d, 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Training (ERM)
    for epoch in range(100):
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()
    
    # Compute risks
    with torch.no_grad():
        train_risk = loss_fn(model(X_train_t), y_train_t).item()
        test_risk = loss_fn(model(X_test_t), y_test_t).item()
    
    print(f"Empirical Risk R̂(ĥ): {train_risk:.6f}")
    print(f"Test Risk (≈R(ĥ)): {test_risk:.6f}")
    print(f"Generalization Gap: {test_risk - train_risk:.6f}")
    
    # Theoretical bounds
    print(f"\nTheoretical Hoeffding bound: {hoeffding_bound(n_train):.4f}")
    print(f"VC bound (d={d+1}): {generalization_bound(n_train, d+1):.4f}")

```

---

## 📊 Common Loss Functions

| Task | Loss | Formula | Properties |
|------|------|---------|------------|
| **Classification** | 0-1 Loss | \(\mathbb{1}[h(x) \neq y]\) | Non-differentiable |
| **Classification** | Cross-Entropy | \(-\sum_c y_c \log h(x)_c\) | Convex, smooth |
| **Classification** | Hinge Loss | \(\max(0, 1 - y \cdot h(x))\) | Convex, non-smooth |
| **Regression** | MSE | \((h(x) - y)^2\) | Convex, smooth |
| **Regression** | MAE | \(|h(x) - y|\) | Convex, robust |
| **Regression** | Huber | MSE if small, MAE if large | Robust, smooth |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Statistical Learning Theory | [Vapnik](https://link.springer.com/book/10.1007/978-1-4757-3264-1) |
| 📖 | Understanding Machine Learning | [Shalev-Shwartz & Ben-David](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) |
| 📖 | ESL Ch. 7 | [Hastie et al.](https://hastie.su.domains/ElemStatLearn/) |

---

⬅️ [Back: Risk Minimization](../) | ➡️ [Next: PAC Learning](../02_pac_learning/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../">⬅️ Back: Risk Minimization</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../02_pac_learning/">Next: PAC Learning ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
