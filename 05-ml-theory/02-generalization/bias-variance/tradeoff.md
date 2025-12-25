<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=BiasVariance%20Tradeoff&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# Bias-Variance Tradeoff

> **The fundamental tradeoff in machine learning**

---

## 📐 Decomposition

```
E[(y - ŷ)²] = Bias²(ŷ) + Var(ŷ) + σ²

Where:
• Bias² = (E[ŷ] - y_true)²    # Systematic error
• Var = E[(ŷ - E[ŷ])²]        # Sensitivity to training data
• σ² = Var(ε)                  # Irreducible noise
```

---

## 🎯 The Tradeoff

```
Simple model (high bias, low variance):
• Underfits
• Same predictions regardless of training data
• Example: Linear regression on non-linear data

Complex model (low bias, high variance):
• Overfits
• Very different predictions for different training sets
• Example: High-degree polynomial
```

---

## 📊 Visual

```
Error
  |
  |   \                    /
  |    \    Total Error   /
  |     \      ____      /
  |      \    /    \    /
  |       \  /      \  /
  |        \/        \/
  |   Bias²    Variance
  |
  +------------------------> Model Complexity
        ↑
      Optimal
```

---

## 🌍 Modern Deep Learning

```
Classical view: U-shaped test error
Modern reality: Double descent!

Very overparameterized models can have:
• Zero training error
• Low test error
• "Implicit regularization"
```

---

<- [Back](./README.md)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
