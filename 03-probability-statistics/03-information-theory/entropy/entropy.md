<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Entropy&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# Entropy

> **Measuring uncertainty/information**

---

## 📐 Definition

```
Discrete:
H(X) = -Σₓ p(x) log p(x)

Continuous (differential entropy):
h(X) = -∫ p(x) log p(x) dx
```

---

## 🔑 Intuition

```
H(X) = Expected "surprise" of X

High entropy = High uncertainty = More information needed
Low entropy = Low uncertainty = Predictable

Example:
• Fair coin: H = 1 bit
• Biased coin (p=0.99): H ≈ 0.08 bits
```

---

## 📊 Common Entropies

| Distribution | Entropy |
|--------------|---------|
| Bernoulli(p) | -p log p - (1-p) log(1-p) |
| Uniform(n) | log n |
| Gaussian(σ) | ½ log(2πeσ²) |

---

## 💻 Code

```python
import numpy as np

def entropy(p):
    """Discrete entropy"""
    p = np.array(p)
    p = p[p > 0]  # Avoid log(0)
    return -np.sum(p * np.log2(p))

# Fair coin
entropy([0.5, 0.5])  # 1.0 bit

# Biased coin
entropy([0.9, 0.1])  # 0.47 bits
```

---

<- [Back](./README.md)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
