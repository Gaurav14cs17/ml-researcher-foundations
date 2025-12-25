<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=6C63FF&height=120&section=header&text=Numerical%20Stability&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-01-6C63FF?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 What is Stability?

```
Stable algorithm: Small input errors → Small output errors
Unstable algorithm: Small input errors → Large output errors

Condition number κ measures sensitivity:
• κ ≈ 1: Well-conditioned (stable)
• κ >> 1: Ill-conditioned (unstable)
```

---

## 🔥 Common Issues in ML

### Softmax Overflow

```python
# BAD: Overflow
def unstable_softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

unstable_softmax([1000, 1001])  # [nan, nan]

# GOOD: Subtract max
def stable_softmax(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

stable_softmax([1000, 1001])  # [0.27, 0.73]
```

### Log-Sum-Exp Underflow

```python
# BAD: Underflow
def unstable_logsumexp(x):
    return np.log(np.sum(np.exp(x)))

# GOOD: Use trick
def stable_logsumexp(x):
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c)))
```

---

## 💻 PyTorch Stable Functions

```python
import torch.nn.functional as F

# Use these instead of manual implementations
F.softmax(x, dim=-1)      # Stable softmax
F.log_softmax(x, dim=-1)  # Stable log-softmax
F.cross_entropy(logits, labels)  # Combines both
torch.logsumexp(x, dim=-1)  # Stable logsumexp
```

---

---

⬅️ [Back: Floating Point](./floating-point.md)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=6C63FF&height=80&section=footer" width="100%"/>
</p>
