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

> **The fundamental optimization algorithm**

---

## 📐 Algorithm

```
Initialize x₀
For k = 0, 1, 2, ...:
    x_{k+1} = x_k - α∇f(x_k)

Where α = learning rate / step size
```

---

## 📊 Convergence Rates

| Setting | Rate | Steps to ε error |
|---------|------|------------------|
| Convex, L-smooth | O(1/k) | O(1/ε) |
| Strongly convex | O((1-μ/L)^k) | O(κ log(1/ε)) |
| Non-convex | O(1/√k) | O(1/ε²) |

Where κ = L/μ is condition number.

---

## 🔑 Learning Rate Selection

```
Too small: Slow convergence
Too large: Divergence / oscillation
Just right: α ≤ 1/L for L-smooth functions

Practical: Start with 0.01, tune
```

---

## 💻 Code

```python
def gradient_descent(grad_f, x0, lr=0.01, n_steps=1000, tol=1e-6):
    x = x0.copy()
    for _ in range(n_steps):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        x = x - lr * g
    return x

# Example: minimize f(x) = x²
grad_f = lambda x: 2 * x
x_opt = gradient_descent(grad_f, x0=np.array([5.0]))
```

---

<- [Back](./README.md)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
