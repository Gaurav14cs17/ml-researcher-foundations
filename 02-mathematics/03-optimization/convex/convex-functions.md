# Convex Functions

> **Functions where local minimum = global minimum**

---

## 📐 Definition

```
f is convex if for all x, y and λ ∈ [0,1]:

f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)

"Function lies below line segment"
```

---

## 🔑 Equivalent Conditions

| Condition | Formula |
|-----------|---------|
| First-order | f(y) ≥ f(x) + ∇f(x)ᵀ(y-x) |
| Second-order | ∇²f(x) ⪰ 0 (PSD Hessian) |
| Epigraph | {(x,t): f(x) ≤ t} is convex set |

---

## 📊 Common Convex Functions

| Function | Domain |
|----------|--------|
| Linear: aᵀx + b | ℝⁿ |
| Quadratic: xᵀAx (A ⪰ 0) | ℝⁿ |
| Norms: \|\|x\|\| | ℝⁿ |
| Log-sum-exp: log(Σexp(xᵢ)) | ℝⁿ |
| Negative entropy: Σxᵢlog(xᵢ) | ℝⁿ₊₊ |

---

## 🌍 ML Applications

| Loss | Convex? |
|------|---------|
| MSE | ✓ Yes (in predictions) |
| Cross-entropy | ✓ Yes (in logits) |
| Neural network loss | ✗ No (in weights) |

---

<- [Back](./README.md)


