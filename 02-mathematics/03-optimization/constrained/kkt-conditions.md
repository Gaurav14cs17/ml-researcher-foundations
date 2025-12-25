<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=120&section=header&text=KKT%20Conditions&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-02-00C853?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

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

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=80&section=footer" width="100%"/>
</p>
