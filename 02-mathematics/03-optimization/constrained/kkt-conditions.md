# KKT Conditions

> **Optimality conditions for constrained problems**

---

## 📐 Problem

```
minimize f(x)
subject to: gᵢ(x) ≤ 0, i = 1,...,m  (inequality)
           hⱼ(x) = 0, j = 1,...,p  (equality)
```

---

## 📐 KKT Conditions

```
1. Stationarity: ∇f(x*) + Σμᵢ∇gᵢ(x*) + Σλⱼ∇hⱼ(x*) = 0

2. Primal feasibility: gᵢ(x*) ≤ 0, hⱼ(x*) = 0

3. Dual feasibility: μᵢ ≥ 0

4. Complementary slackness: μᵢgᵢ(x*) = 0
```

---

## 🔑 Complementary Slackness

```
μᵢ · gᵢ(x*) = 0 means:

Either μᵢ = 0 (constraint inactive, slack)
Or gᵢ(x*) = 0 (constraint active, binding)

"Multiplier positive only for active constraints"
```

---

## 💻 Example

```python
# minimize f(x) = x²
# subject to: x ≥ 1 (i.e., 1 - x ≤ 0)

# KKT conditions:
# 1. 2x - μ = 0
# 2. 1 - x ≤ 0
# 3. μ ≥ 0
# 4. μ(1 - x) = 0

# Case 1: μ = 0 (inactive)
#   2x = 0 → x = 0, but 1 - 0 = 1 > 0 violates (2)

# Case 2: 1 - x = 0 (active)
#   x = 1, μ = 2·1 = 2 ≥ 0 ✓

# Solution: x* = 1, μ* = 2
```

---

## 🌍 In ML

| Application | KKT Use |
|-------------|---------|
| SVM | Derive dual problem |
| Constrained RL | Lagrangian relaxation |
| Trust region | Active set methods |

---

---

➡️ [Next: Lagrange Multipliers](./lagrange-multipliers.md)
