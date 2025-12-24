# Risk Minimization

> **The theoretical foundation of machine learning**

---

## 🎯 Visual Overview

<img src="./images/risk-minimization.svg" width="100%">

*Caption: Risk minimization is the goal of ML: minimize expected loss. True risk R(f) is unknown (we can't access the true distribution). We minimize empirical risk R̂(f) on training data and hope it generalizes.*

---

## 📂 Overview

Risk minimization provides the theoretical framework for understanding machine learning. The goal is to find a function that minimizes expected loss on unseen data.

---

## 📐 Mathematical Foundations

### True Risk (Population Risk)
```
R(f) = E_{(x,y)~P}[ℓ(f(x), y)]

Unknown! We can't compute this directly.
```

### Empirical Risk
```
R̂(f) = (1/n) Σᵢ₌₁ⁿ ℓ(f(xᵢ), yᵢ)

This is what we actually minimize (training loss).
```

### Generalization Gap
```
Generalization gap = R(f) - R̂(f)

Goal: Make this small!
Bound: R(f) ≤ R̂(f) + complexity_term
```

### Structural Risk Minimization
```
f* = argmin [R̂(f) + λ Ω(f)]
            ---------  --------
            empirical  complexity
              risk     penalty

Examples:
• Ω(f) = ||w||₂² → L2 regularization
• Ω(f) = ||w||₁ → L1 regularization
```

---

## 🔑 Key Topics

| Topic | Description |
|-------|-------------|
| Fundamentals | Core concepts |
| Applications | ML use cases |
| Implementation | Code examples |

---

## 💻 Example

```python
import torch
import torch.nn.functional as F

# Empirical Risk Minimization
def empirical_risk(model, data, targets, loss_fn=F.cross_entropy):
    """Compute average loss over training data"""
    predictions = model(data)
    return loss_fn(predictions, targets).mean()

# Structural Risk = Empirical Risk + Regularization
def structural_risk(model, data, targets, reg_lambda=0.01):
    emp_risk = empirical_risk(model, data, targets)
    l2_reg = sum(p.pow(2).sum() for p in model.parameters())
    return emp_risk + reg_lambda * l2_reg
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Vapnik: Statistical Learning | [Book](https://link.springer.com/book/10.1007/978-1-4757-3264-1) |
| 📄 | ESL Ch. 7 | [Book](https://hastie.su.domains/ElemStatLearn/) |
| 📖 | Bishop PRML Ch. 1 | [Book](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 🇨🇳 | 经验风险最小化 | [知乎](https://zhuanlan.zhihu.com/p/38853908) |
| 🇨🇳 | 机器学习理论基础 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | 统计学习方法 | [B站](https://www.bilibili.com/video/BV164411b7dx) |

---

<- [Back](../)

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

---

⬅️ [Back: risk-minimization](../)
