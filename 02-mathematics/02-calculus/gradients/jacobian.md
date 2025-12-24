# Jacobian Matrix

> **The derivative of vector-valued functions**

---

## 📐 Definition

```
For f: ℝⁿ → ℝᵐ

J = ∂f/∂x ∈ ℝᵐˣⁿ

J[i,j] = ∂fᵢ/∂xⱼ

"Row i = gradient of fᵢ"
```

---

## 🎯 Intuition

```
For small Δx:
f(x + Δx) ≈ f(x) + J·Δx

"Jacobian is the best linear approximation"
```

---

## 📊 Special Cases

| Type | Input | Output | Derivative |
|------|-------|--------|------------|
| Scalar function | 1 | 1 | Derivative (1×1) |
| Gradient | n | 1 | Gradient (1×n) |
| Jacobian | n | m | Jacobian (m×n) |

---

## 💻 Code

```python
import torch

# Function f: R³ → R²
def f(x):
    return torch.stack([
        x[0] * x[1],
        x[1] * x[2]
    ])

x = torch.tensor([1., 2., 3.], requires_grad=True)

# Compute Jacobian
jacobian = torch.autograd.functional.jacobian(f, x)
print(jacobian.shape)  # (2, 3)
# [[∂f₁/∂x₁, ∂f₁/∂x₂, ∂f₁/∂x₃],
#  [∂f₂/∂x₁, ∂f₂/∂x₂, ∂f₂/∂x₃]]
```

---

## 🌍 In Deep Learning

| Context | Jacobian |
|---------|----------|
| Layer output | ∂output/∂input |
| Change of variables | For flow-based models |
| Lipschitz constant | Spectral norm of Jacobian |

---

---

⬅️ [Back: Hessian](./hessian.md)
