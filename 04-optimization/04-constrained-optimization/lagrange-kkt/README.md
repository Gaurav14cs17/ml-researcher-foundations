<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Lagrange & KKT&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# 🎯 Lagrange Multipliers & KKT Conditions

> **Solving constrained optimization problems**

---

## 📐 Lagrangian Method

```
Problem:
  min f(x)
  s.t. g(x) = 0

Lagrangian:
  L(x, λ) = f(x) + λ g(x)

Necessary conditions:
  ∇ₓL = 0  →  ∇f + λ∇g = 0
  ∇λL = 0  →  g(x) = 0
```

---

## 📐 KKT Conditions

```
Problem:
  min f(x)
  s.t. gᵢ(x) ≤ 0  (inequality)
       hⱼ(x) = 0  (equality)

KKT Conditions:
1. Stationarity: ∇f + Σμᵢ∇gᵢ + Σλⱼ∇hⱼ = 0
2. Primal feasibility: gᵢ(x) ≤ 0, hⱼ(x) = 0
3. Dual feasibility: μᵢ ≥ 0
4. Complementary slackness: μᵢgᵢ(x) = 0
```

---

## 💻 Example: SVM Dual

```python
# SVM dual uses KKT conditions
# min 1/2||w||² s.t. yᵢ(w·xᵢ + b) ≥ 1

# Lagrangian:
# L = 1/2||w||² - Σαᵢ[yᵢ(w·xᵢ + b) - 1]

# KKT gives us the dual:
# max Σαᵢ - 1/2 ΣᵢΣⱼ αᵢαⱼyᵢyⱼ(xᵢ·xⱼ)
# s.t. αᵢ ≥ 0, Σαᵢyᵢ = 0
```

---

## 🔗 Applications

| Application | Usage |
|-------------|-------|
| **SVM** | Dual formulation via KKT |
| **Neural Networks** | Constrained training |
| **Portfolio Optimization** | Risk constraints |
| **Physics-Informed ML** | Conservation constraints |

---

⬅️ [Back: Constrained Optimization](../)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

