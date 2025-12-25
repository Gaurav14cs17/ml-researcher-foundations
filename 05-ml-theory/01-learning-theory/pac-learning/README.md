<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=PAC%20Learning&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/pac-vc-dimension-complete.svg" width="100%">

---

## 📐 PAC Definition

```
A concept class C is PAC-learnable if there exists:
• Algorithm A
• Polynomial p(1/ε, 1/δ, n, size(c))

Such that for all:
• c ∈ C (target concept)
• D (distribution over X)
• ε > 0 (accuracy parameter)
• δ > 0 (confidence parameter)

With m ≥ p(1/ε, 1/δ, n, size(c)) samples:
  P[error(h) ≤ ε] ≥ 1 - δ

"Probably (1-δ) Approximately (≤ε) Correct"
```

---

## 📐 Sample Complexity

```
Finite hypothesis class |H|:
  m ≥ (1/ε)(ln|H| + ln(1/δ))

General (VC dimension d):
  m = O((d/ε)log(1/ε) + (1/ε)log(1/δ))
```

---

## 💻 Code Example

```python
import numpy as np

def sample_complexity_finite(H_size, epsilon, delta):
    """PAC sample complexity for finite H"""
    return int(np.ceil((1/epsilon) * (np.log(H_size) + np.log(1/delta))))

def sample_complexity_vc(vc_dim, epsilon, delta):
    """PAC sample complexity using VC dimension"""
    return int(np.ceil((4/epsilon) * (vc_dim * np.log(12/epsilon) + np.log(2/delta))))

# Example
m = sample_complexity_vc(vc_dim=10, epsilon=0.1, delta=0.05)
print(f"Need {m} samples for PAC learning")
```

---

⬅️ [Back: Learning Theory](../)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
