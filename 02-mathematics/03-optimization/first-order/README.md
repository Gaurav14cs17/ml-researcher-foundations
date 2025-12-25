<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=120&section=header&text=First-Order%20Methods&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-02-00C853?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

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

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=80&section=footer" width="100%"/>
</p>
