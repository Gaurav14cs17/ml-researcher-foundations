<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=120&section=header&text=Convex%20Functions&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-02-00C853?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

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

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=80&section=footer" width="100%"/>
</p>
