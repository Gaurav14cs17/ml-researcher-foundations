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


