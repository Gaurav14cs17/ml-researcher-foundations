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


