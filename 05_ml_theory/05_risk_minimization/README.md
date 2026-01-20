<!-- Navigation -->
<p align="center">
  <a href="../04_representation/">⬅️ Prev: Representation</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 ML Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../06_ensemble_methods/">Next: Ensemble Methods ➡️</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Risk%20Minimization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/risk-minimization.svg" width="100%">

*Caption: Risk minimization is the fundamental framework for machine learning: minimize expected loss on unseen data.*

---

## 📂 Overview

**Risk Minimization** provides the theoretical framework for understanding machine learning. The goal is to find a function that minimizes expected loss on unseen data, bridging the gap between training and test performance.

---

## 📐 Mathematical Framework

### True Risk (Population Risk)

The **true risk** is the expected loss over the data distribution:

```math
R(h) = \mathbb{E}_{(x,y) \sim P}[\ell(h(x), y)] = \int \ell(h(x), y) \, dP(x, y)
```

**Problem:** We don't know \(P\)!

### Empirical Risk

The **empirical risk** approximates true risk using training data:

```math
\hat{R}(h) = \frac{1}{n}\sum_{i=1}^n \ell(h(x_i), y_i)
```

**Law of Large Numbers:** As \(n \to \infty\), \(\hat{R}(h) \to R(h)\).

### Generalization Gap

```math
\text{Gap}(h) = R(h) - \hat{R}(h)
```

**Goal:** Control this gap to ensure good generalization.

---

## 📐 Empirical Risk Minimization (ERM)

```math
\hat{h}_{\text{ERM}} = \arg\min_{h \in \mathcal{H}} \hat{R}(h)
```

**Theorem (Uniform Convergence):** With probability \(\geq 1 - \delta\):

```math
\sup_{h \in \mathcal{H}} |R(h) - \hat{R}(h)| \leq \epsilon(n, |\mathcal{H}|, \delta)
```

where \(\epsilon\) decreases with \(n\) and increases with complexity of \(\mathcal{H}\).

---

## 📐 Structural Risk Minimization (SRM)

Add complexity penalty to control overfitting:

```math
\hat{h}_{\text{SRM}} = \arg\min_{h \in \mathcal{H}} \left[\hat{R}(h) + \lambda \Omega(h)\right]
```

| Component | Meaning |
|-----------|---------|
| \(\hat{R}(h)\) | Data fit (training loss) |
| \(\Omega(h)\) | Complexity penalty |
| \(\lambda\) | Trade-off parameter |

---

## 📂 Topics in This Section

| Folder | Topic | Key Concept |
|--------|-------|-------------|
| [01_erm/](./01_erm/) | Empirical Risk | Minimize training loss |
| [02_pac_learning/](./02_pac_learning/) | PAC Learning | Sample complexity bounds |
| [03_structural_risk/](./03_structural_risk/) | SRM | Regularization theory |

---

## 💻 Code Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RiskMinimization:
    """Framework for risk minimization in ML."""
    
    @staticmethod
    def empirical_risk(model, X, y, loss_fn=F.mse_loss):
        """
        Empirical Risk: R̂(h) = (1/n) Σ ℓ(h(xᵢ), yᵢ)
        """
        predictions = model(X)
        return loss_fn(predictions, y, reduction='mean')
    
    @staticmethod
    def structural_risk(model, X, y, reg_lambda=0.01, loss_fn=F.mse_loss):
        """
        Structural Risk: R̂(h) + λ Ω(h)
        """
        emp_risk = RiskMinimization.empirical_risk(model, X, y, loss_fn)
        
        # L2 regularization
        l2_reg = sum(p.pow(2).sum() for p in model.parameters())
        
        return emp_risk + reg_lambda * l2_reg
    
    @staticmethod
    def generalization_bound(train_error, vc_dim, n, delta=0.05):
        """
        VC Generalization Bound:
        R(h) ≤ R̂(h) + √((d(ln(2n/d)+1) + ln(4/δ))/n)
        """
        complexity = vc_dim * (np.log(2*n/vc_dim) + 1) + np.log(4/delta)
        return train_error + np.sqrt(complexity / n)

def train_with_risk_minimization(model, train_loader, val_loader, 
                                  reg_lambda=0.01, epochs=100, lr=0.001):
    """
    Train model using Structural Risk Minimization.
    
    Objective: min R̂(h) + λ||θ||²
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg_lambda)
    
    train_risks = []
    val_risks = []
    
    for epoch in range(epochs):
        # Training (minimize structural risk)
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = F.mse_loss(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        # Validation (estimate true risk)
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)
                total_val_loss += F.mse_loss(pred, y_batch).item()
        
        train_risk = total_train_loss / len(train_loader)
        val_risk = total_val_loss / len(val_loader)
        
        train_risks.append(train_risk)
        val_risks.append(val_risk)
        
        if (epoch + 1) % 20 == 0:
            gap = val_risk - train_risk
            print(f"Epoch {epoch+1}: Train={train_risk:.4f}, Val={val_risk:.4f}, Gap={gap:.4f}")
    
    return train_risks, val_risks

# Theoretical analysis
def analyze_generalization():
    """Analyze generalization bounds for different scenarios."""
    print("Generalization Bounds Analysis")
    print("=" * 50)
    
    train_error = 0.01  # 1% training error
    delta = 0.05  # 95% confidence
    
    print(f"\nFixed train error = {train_error}, δ = {delta}")
    print("-" * 50)
    
    # Effect of sample size
    print("\n1. Effect of Sample Size (VC = 10):")
    for n in [100, 1000, 10000, 100000]:
        bound = RiskMinimization.generalization_bound(train_error, 10, n, delta)
        print(f"   n = {n:6d}: R(h) ≤ {bound:.4f}")
    
    # Effect of VC dimension
    print("\n2. Effect of VC Dimension (n = 1000):")
    for vc in [5, 10, 50, 100]:
        bound = RiskMinimization.generalization_bound(train_error, vc, 1000, delta)
        print(f"   VC = {vc:3d}: R(h) ≤ {bound:.4f}")

if __name__ == "__main__":
    analyze_generalization()
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Statistical Learning Theory | [Vapnik](https://link.springer.com/book/10.1007/978-1-4757-3264-1) |
| 📖 | Understanding ML | [Shalev-Shwartz](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) |
| 📖 | ESL | [Hastie et al.](https://hastie.su.domains/ElemStatLearn/) |

---

⬅️ [Back: Representation](../04_representation/) | ➡️ [Next: Ensemble Methods](../06_ensemble_methods/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../04_representation/">⬅️ Prev: Representation</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 ML Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../06_ensemble_methods/">Next: Ensemble Methods ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
