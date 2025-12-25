<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=120&section=header&text=Convex%20Optimization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-02-00C853?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/convex-vs-nonconvex.svg" width="100%">

*Caption: Comparison of convex (left) and non-convex (right) functions. Convex functions have a single global minimum that gradient descent will always find. Non-convex functions (like neural network loss landscapes) have multiple local minima and saddle points.*

---

## 📐 Convex Set

```
A set C is convex if:
∀x, y ∈ C, ∀λ ∈ [0,1]: λx + (1-λ)y ∈ C

The line segment between any two points stays in C.
```

---

## 📐 Convex Function

```
f is convex if:
f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)

Equivalently (for smooth f):
• f(y) ≥ f(x) + ∇f(x)ᵀ(y-x)  (first-order)
• H = ∇²f ⪰ 0 (second-order)
```

---

## 🔥 Why Convexity Matters

```
Convex function:
• Every local minimum is global!
• Gradient descent converges
• Duality theory applies

Non-convex (deep learning):
• Many local minima/saddle points
• No global guarantees
• But often works anyway!
```

---

## 📊 Common Convex Functions

| Function | Domain |
|----------|--------|
| Linear | ℝⁿ |
| Quadratic (PSD) | ℝⁿ |
| Norms | ℝⁿ |
| Log-sum-exp | ℝⁿ |
| Cross-entropy | Δⁿ |


## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |


## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Textbook | See parent folder |
| 🎥 | Video Lectures | YouTube/Coursera |
| 🇨🇳 | 中文资源 | 知乎/B站 |

---

⬅️ [Back: Optimization](../)

---

⬅️ [Back: Constrained](../constrained/) | ➡️ [Next: Duality](../duality/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=80&section=footer" width="100%"/>
</p>
