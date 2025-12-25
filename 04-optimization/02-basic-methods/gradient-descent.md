<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Gradient%20Descent&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# Gradient Descent

> First-order iterative optimization algorithm

## 📊 Visual Intuition

```
     Loss Surface              Gradient Descent Path
    +---------------+         +---------------+
    | ╱╲    ╱╲      |         | •→→→╲         |
    |╱  ╲  ╱  ╲     |         |      ╲→→→    |
    |    ╲╱    ╲    |  --->   |          ╲→  |
    |          ╲╱   |         |           •  |
    |    minimum    |         |    found!    |
    +---------------+         +---------------+
```

## 📐 Key Formula

```
Update Rule:
+-------------------------------------+
|  x_{k+1} = x_k - α ∇f(x_k)         |
+-------------------------------------+

where:
  α = learning rate (step size)
  ∇f = gradient (direction of steepest ascent)
  -∇f = direction of steepest descent
```

## ⚙️ Algorithm

```
1. Initialize x₀ randomly
2. while not converged:
3.     g = ∇f(x)           # compute gradient
4.     x = x - α * g       # take step
5.     check convergence   # ||g|| < ε ?
```

## 📈 Convergence Rates

| Function Type | Rate | Meaning |
|---------------|------|---------|
| Convex | O(1/k) | Sublinear |
| Strongly Convex | O(ρᵏ), ρ<1 | Linear |
| Non-convex | May not converge | Local minimum |

## 🎯 Key Concepts

- **Learning Rate α** — Too large: diverge, too small: slow
- **Line Search** — Armijo, Wolfe conditions
- **Momentum** — Accelerate through flat regions
- **Batch vs Stochastic** — Full gradient vs mini-batch

## ⚠️ Common Pitfalls

```
α too large:        α too small:        Good α:
    •                   •               •
   ╱ ╲                  |              ╲
  •   •                 •               •
 ╱     ╲                |                ╲
•       •               •                 •
 Oscillate!          Too slow!         Converge ✓
```

## 📚 Resources

| Type | Resource | Link |
|------|----------|------|
| 📄 Paper | Ruder: Overview of GD | [arXiv](https://arxiv.org/abs/1609.04747) |
| 📖 Book | Nocedal & Wright Ch.3 | [Springer](https://link.springer.com/book/10.1007/978-0-387-40065-5) |
| 🎥 Video | 3Blue1Brown | [YouTube](https://www.youtube.com/watch?v=IHZwWFHWa-w) |
| 🇨🇳 博客 | CSDN梯度下降详解 | [CSDN](https://blog.csdn.net/google19890102/article/details/69942970) |
| 🇨🇳 知乎 | 梯度下降直观理解 | [知乎](https://zhuanlan.zhihu.com/p/22252270) |

---

---

➡️ [Next: Newton](./newton.md)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
