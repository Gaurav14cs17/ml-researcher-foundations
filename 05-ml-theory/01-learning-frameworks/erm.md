<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Empirical%20Risk%20Minimization%20ER&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# Empirical Risk Minimization (ERM)

> **The fundamental learning principle**

---

## 📐 Definition

```
Population risk (what we want):
R(h) = E_{(x,y)~P}[L(h(x), y)]

Empirical risk (what we compute):
R̂(h) = (1/n) Σᵢ L(h(xᵢ), yᵢ)

ERM: ĥ = argmin_h R̂(h)
```

---

## 🔑 Why It Works

```
Law of Large Numbers:
R̂(h) → R(h) as n → ∞

Uniform convergence (with capacity control):
sup_h |R̂(h) - R(h)| → 0
```

---

## ⚠️ Overfitting

```
Problem: ĥ minimizes R̂, not R

R(ĥ) = R̂(ĥ) + (R(ĥ) - R̂(ĥ))
        +------+  +------------+
         training   generalization
         error         gap

Solutions:
• Regularization: argmin R̂(h) + λΩ(h)
• Validation set: Early stopping
• More data
```

---

## 💻 Code

```python
import torch
import torch.nn.functional as F

def empirical_risk(model, dataloader, loss_fn):
    """Compute empirical risk (average loss)"""
    total_loss = 0
    n_samples = 0
    
    for x, y in dataloader:
        pred = model(x)
        loss = loss_fn(pred, y)
        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)
    
    return total_loss / n_samples

# ERM training loop
for epoch in range(epochs):
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)  # Minimize empirical risk
        loss.backward()
        optimizer.step()
```

---

## 📊 Beyond ERM

| Method | Modification |
|--------|--------------|
| Regularized ERM | + penalty term |
| Distributionally robust | Worst-case over distributions |
| Structural risk min | Capacity-dependent bound |

---

<- [Back](./README.md)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
