# 📐 Calculus for Optimization

> **Understanding how functions change**

## 🎯 Visual Overview

<img src="./images/calculus-foundations.svg" width="100%">

*Caption: Calculus for optimization: derivatives (rate of change), gradients (direction of steepest ascent in n-D), Hessian (curvature for second-order methods), and chain rule (enables backpropagation).*

---

## 📂 Topics in This Folder

| File | Topic | Key Concept |
|------|-------|-------------|
| [gradients.md](./gradients.md) | Gradients & Partial Derivatives | ∇f points uphill |
| [hessian.md](./hessian.md) | Hessian Matrix | Curvature information |

---

## 🎯 The Big Picture

```
Function f(x)
     |
     v
+-----------------------------------------------------+
|                                                     |
|   First Derivative        Second Derivative         |
|   ∇f (Gradient)           H (Hessian)              |
|                                                     |
|   "Which direction        "How curved is           |
|    to move?"              the surface?"            |
|                                                     |
|        |                        |                   |
|        v                        v                   |
|   Gradient Descent         Newton's Method          |
|   (first-order)            (second-order)           |
|                                                     |
+-----------------------------------------------------+
```

---

## 📐 Key Formulas

### Gradient (First Derivative)
```
For f: ℝⁿ → ℝ

∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ

• Points in direction of steepest ASCENT
• Negative gradient → steepest DESCENT
```

### Hessian (Second Derivative)
```
H(f) = [∂²f/∂xᵢ∂xⱼ]

      +                              +
      | ∂²f/∂x₁²    ∂²f/∂x₁∂x₂  ... |
H =   | ∂²f/∂x₂∂x₁  ∂²f/∂x₂²    ... |
      | ...         ...          ... |
      +                              +

• Symmetric matrix (nice properties!)
• Eigenvalues tell us about curvature
```

---

## 🌍 Real-World Applications

| Application | Calculus Concept | How |
|-------------|------------------|-----|
| **Neural Network Training** | Chain rule | Backpropagation |
| **GPT/LLM** | Gradient | Loss.backward() |
| **Physics Simulation** | Hessian | Force = -∇V |
| **Robotics** | Jacobian | Inverse kinematics |

---

## 💻 Python Example

```python
import torch

# Define function: f(x,y) = x² + y²
def f(x, y):
    return x**2 + y**2

# Compute gradient using autograd
x = torch.tensor([3.0], requires_grad=True)
y = torch.tensor([4.0], requires_grad=True)

z = f(x, y)
z.backward()

print(f"∇f at (3,4) = [{x.grad.item()}, {y.grad.item()}]")
# Output: ∇f at (3,4) = [6.0, 8.0]
```

---

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 📖 | Khan Academy Multivariable | [Link](https://www.khanacademy.org/math/multivariable-calculus) |
| 🎥 | 3Blue1Brown Calculus | [YouTube](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) |
| 🇨🇳 | 知乎梯度详解 | [知乎](https://zhuanlan.zhihu.com/p/25202034) |

---

<- [Back](../README.md) | ➡️ [Next: Gradients](./gradients.md)

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---
