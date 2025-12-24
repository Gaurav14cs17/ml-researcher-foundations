# First-Order Methods

> **Optimization using only gradients**

---

## 🎯 Visual Overview

<img src="./images/gradient-descent-visualization.svg" width="100%">

*Caption: Gradient descent iteratively moves toward the minimum by following the negative gradient. The path shows how the algorithm navigates the loss landscape, with each step proportional to the gradient magnitude and learning rate.*

---

## 📐 Gradient Descent

```
x_{k+1} = x_k - α∇f(x_k)

Where:
• α = learning rate / step size
• ∇f(x_k) = gradient at current point
```

---

## 📊 Convergence

| Setting | Rate | Assumption |
|---------|------|------------|
| Convex | O(1/k) | L-smooth |
| Strongly convex | O((1-μ/L)^k) | μ-strongly convex |
| Non-convex | O(1/√k) | To stationary point |

---

## 🔑 Momentum

```
v_{k+1} = βv_k + ∇f(x_k)
x_{k+1} = x_k - αv_k

β = 0.9 typically

Accelerates through ravines, reduces oscillation
```

---

## 💻 Code

```python
def gradient_descent(f, grad_f, x0, lr=0.01, n_steps=100):
    x = x0
    for _ in range(n_steps):
        x = x - lr * grad_f(x)
    return x

def momentum_gd(f, grad_f, x0, lr=0.01, beta=0.9, n_steps=100):
    x, v = x0, 0
    for _ in range(n_steps):
        v = beta * v + grad_f(x)
        x = x - lr * v
    return x
```


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

⬅️ [Back: Duality](../duality/) | ➡️ [Next: Second Order](../second-order/)
