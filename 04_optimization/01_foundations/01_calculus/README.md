<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=Calculus%20for%20Optimization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

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

# Part 1: Gradients & Partial Derivatives

## 📖 What is a Gradient?

The gradient is a vector of all partial derivatives. It points in the direction of **steepest ascent**.

```
+---------------------------------------------------------+
|                                                         |
|   For f(x₁, x₂, ..., xₙ):                              |
|                                                         |
|                + ∂f/∂x₁ +                               |
|                | ∂f/∂x₂ |                               |
|   ∇f(x) =      |   ⋮    |                               |
|                | ∂f/∂xₙ |                               |
|                +        +                               |
|                                                         |
|   Size: n × 1 vector                                    |
|                                                         |
+---------------------------------------------------------+

```

---

## 🎯 Visual Intuition

```
         Mountain Surface (Loss Landscape)
         
              ↗ ∇f (gradient points UP)
             /
            •  You are here
           /|\
          / | \
         /  |  \
        ----+----  Valley (minimum)
        
   To minimize: Go OPPOSITE to gradient!
   
   x_new = x_old - α∇f(x_old)
           -----------------
           Gradient Descent!

```

---

## 📐 Mathematical Definition and Properties

### Formal Definition

For a scalar-valued function \( f: \mathbb{R}^n \to \mathbb{R} \), the gradient is defined as:

\[
\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}
\]

### Directional Derivative

The directional derivative of \( f \) at \( \mathbf{x} \) in direction \( \mathbf{d} \) (unit vector):

\[
D_\mathbf{d} f(\mathbf{x}) = \nabla f(\mathbf{x})^\top \mathbf{d} = \|\nabla f(\mathbf{x})\| \cos\theta
\]

where \( \theta \) is the angle between \( \nabla f \) and \( \mathbf{d} \).

**Key Insight:** Maximum increase occurs when \( \theta = 0 \), i.e., moving in the direction of the gradient.

### Gradient Properties

| Property | Formula | Meaning |
|----------|---------|---------|
| **Linearity** | ∇(af + bg) = a∇f + b∇g | Gradients add |
| **Product Rule** | ∇(fg) = f∇g + g∇f | Chain rule for products |
| **Chain Rule** | ∇(f∘g) = (∇f)·(∇g) | Backpropagation! |
| **Zero at extrema** | ∇f(x*) = 0 | How we find optima |

---

## 📐 Step-by-Step Example

### Function: f(x, y) = x² + 2y²

**Step 1: Partial Derivatives**

```
∂f/∂x = 2x    (derivative treating y as constant)
∂f/∂y = 4y    (derivative treating x as constant)

```

**Step 2: Gradient Vector**

```
∇f(x,y) = [2x, 4y]ᵀ

```

**Step 3: Evaluate at Point (3, 2)**

```
∇f(3,2) = [2(3), 4(2)]ᵀ = [6, 8]ᵀ

```

**Step 4: Gradient Descent Step**

```
α = 0.1  (learning rate)

[x_new]   [3]         [6]   [3 - 0.6]   [2.4]
[y_new] = [2] - 0.1 × [8] = [2 - 0.8] = [1.2]

```

---

## 📐 Proof: Gradient Points in Direction of Steepest Ascent

**Theorem:** For a differentiable function \( f \), the gradient \( \nabla f(\mathbf{x}) \) points in the direction of steepest increase at \( \mathbf{x} \).

**Proof:**

```
Step 1: Consider the directional derivative in direction d (||d|| = 1):
  D_d f(x) = ∇f(x)ᵀ · d = ||∇f(x)|| · ||d|| · cos(θ) = ||∇f(x)|| · cos(θ)

Step 2: Maximize the directional derivative
  max_d D_d f(x) = max_θ ||∇f(x)|| · cos(θ)

Step 3: Since cos(θ) ≤ 1 with equality at θ = 0:
  Maximum occurs when d is parallel to ∇f(x)
  
Step 4: Maximum value is:
  ||∇f(x)|| · cos(0) = ||∇f(x)||

Therefore: ∇f points in direction of steepest ascent with magnitude ||∇f||. ∎

```

---

## 🌍 Where Gradients Are Used

| Application | How | Example |
|-------------|-----|---------|
| **Neural Networks** | Backpropagation computes ∇L | Training GPT |
| **Diffusion Models** | Score ∇log p(x) | Stable Diffusion |
| **Physics** | Force = -∇V (potential) | Molecular dynamics |
| **Economics** | Marginal utility = ∂U/∂x | Optimization |
| **Computer Graphics** | Surface normals | Shading |

---

## 💻 Code Examples

### PyTorch (Autograd)

```python
import torch

# Define parameters
x = torch.tensor([3.0, 2.0], requires_grad=True)

# Define function
def f(x):
    return x[0]**2 + 2*x[1]**2

# Compute gradient
loss = f(x)
loss.backward()

print(f"∇f = {x.grad}")  # tensor([6., 8.])

```

### NumPy (Manual)

```python
import numpy as np

def gradient_f(x, y):
    """Gradient of f(x,y) = x² + 2y²"""
    df_dx = 2 * x
    df_dy = 4 * y
    return np.array([df_dx, df_dy])

grad = gradient_f(3, 2)
print(f"∇f(3,2) = {grad}")  # [6, 8]

```

### JAX (Automatic)

```python
import jax
import jax.numpy as jnp

def f(x):
    return x[0]**2 + 2*x[1]**2

# Auto-compute gradient function
grad_f = jax.grad(f)

x = jnp.array([3.0, 2.0])
print(f"∇f = {grad_f(x)}")  # [6., 8.]

```

---

## ⚠️ Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| Confusing ∇f direction | Gradient is ASCENT, not descent | Use **-**∇f |
| Wrong partial derivative | Forgot to treat others as constant | Check each variable |
| Not normalizing | Gradient can be huge | Clip or normalize |
| Ignoring numerical issues | Gradient vanishing/exploding | Use techniques like BatchNorm |

---

# Part 2: Hessian Matrix

## 📖 What is the Hessian?

The Hessian is a matrix of **second partial derivatives**. It tells us about the **curvature** of the function.

```
+---------------------------------------------------------+
|                                                         |
|   For f(x₁, x₂, ..., xₙ):                              |
|                                                         |
|         + ∂²f/∂x₁²    ∂²f/∂x₁∂x₂  ...  ∂²f/∂x₁∂xₙ +   |
|         | ∂²f/∂x₂∂x₁  ∂²f/∂x₂²    ...  ∂²f/∂x₂∂xₙ |   |
|   H =   |     ⋮           ⋮        ⋱       ⋮      |   |
|         | ∂²f/∂xₙ∂x₁  ∂²f/∂xₙ∂x₂  ...  ∂²f/∂xₙ²   |   |
|         +                                          +   |
|                                                         |
|   Size: n × n matrix (symmetric!)                       |
|                                                         |
+---------------------------------------------------------+

```

---

## 🎯 Visual Intuition

```
Hessian tells us the SHAPE of the bowl:

    Positive Definite         Negative Definite        Indefinite
    (H ≻ 0, all λ > 0)        (H ≺ 0, all λ < 0)       (mixed λ)
    
         ╲___╱                    ╱‾‾‾╲                  ╲__╱‾╲
          \•/                     \•/                      •
        MINIMUM                 MAXIMUM               SADDLE POINT
        
    "Bowl up"                "Bowl down"             "Saddle"

```

---

## 📐 Mathematical Definition

For \( f: \mathbb{R}^n \to \mathbb{R} \), the Hessian matrix is:

\[
H(f) = \nabla^2 f = \begin{bmatrix} 
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}
\]

**Schwarz's Theorem:** If \( f \) has continuous second derivatives, then \( H \) is symmetric:
\[
\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}
\]

---

## 📐 Example: f(x, y) = x² + 3y²

**Step 1: First Derivatives**

```
∂f/∂x = 2x
∂f/∂y = 6y

```

**Step 2: Second Derivatives**

```
∂²f/∂x² = 2       (how ∂f/∂x changes with x)
∂²f/∂y² = 6       (how ∂f/∂y changes with y)
∂²f/∂x∂y = 0      (how ∂f/∂x changes with y)
∂²f/∂y∂x = 0      (how ∂f/∂y changes with x)

```

**Step 3: Hessian Matrix**

```
      +     +
H =   | 2  0 |
      | 0  6 |
      +     +

```

**Step 4: Analyze Eigenvalues**

```
λ₁ = 2 > 0
λ₂ = 6 > 0

Both positive → MINIMUM at (0,0) ✓

```

---

## 📐 Classifying Critical Points Using the Hessian

### Second Derivative Test (Proof)

**Theorem:** Let \( \mathbf{x}^* \) be a critical point where \( \nabla f(\mathbf{x}^*) = \mathbf{0} \). Then:
- If \( H(\mathbf{x}^*) \succ 0 \) (positive definite), \( \mathbf{x}^* \) is a local minimum
- If \( H(\mathbf{x}^*) \prec 0 \) (negative definite), \( \mathbf{x}^* \) is a local maximum  
- If \( H(\mathbf{x}^*) \) has both positive and negative eigenvalues, \( \mathbf{x}^* \) is a saddle point

**Proof:**

```
Step 1: Taylor expansion around x*
  f(x* + Δx) = f(x*) + ∇f(x*)ᵀΔx + ½ΔxᵀH(x*)Δx + O(||Δx||³)

Step 2: Since x* is critical, ∇f(x*) = 0
  f(x* + Δx) = f(x*) + ½ΔxᵀH(x*)Δx + O(||Δx||³)

Step 3: For small Δx, the quadratic term dominates
  f(x* + Δx) - f(x*) ≈ ½ΔxᵀH(x*)Δx

Step 4: Positive definite H means ΔxᵀHΔx > 0 for all Δx ≠ 0
  Therefore f(x* + Δx) > f(x*) for all small perturbations
  → x* is a local minimum ∎

```

| Hessian Eigenvalues | Type | Example |
|---------------------|------|---------|
| All λ > 0 | Local minimum | Bottom of bowl |
| All λ < 0 | Local maximum | Top of hill |
| Mixed signs | Saddle point | Horse saddle |
| Some λ = 0 | Degenerate | Needs more analysis |

---

## 🔗 Connection to Optimization: Taylor Expansion

```
Taylor Expansion (2nd order):

f(x + Δx) ≈ f(x) + ∇f(x)ᵀΔx + ½ΔxᵀHΔx
            -----  ----------  --------
            value   linear      quadratic
                    term        term (curvature)

Newton's method minimizes this quadratic approximation!

```

### Newton's Method Derivation

```
Step 1: Approximate f with Taylor expansion
  f(x + Δx) ≈ f(x) + ∇f(x)ᵀΔx + ½ΔxᵀHΔx

Step 2: Minimize the approximation over Δx
  ∂/∂Δx [f(x) + ∇f(x)ᵀΔx + ½ΔxᵀHΔx] = 0
  ∇f(x) + HΔx = 0

Step 3: Solve for optimal Δx
  Δx* = -H⁻¹∇f(x)

Step 4: Newton update
  x_{k+1} = x_k - H(x_k)⁻¹∇f(x_k)

```

---

## 🌍 Where Hessian Is Used

| Application | How | Why |
|-------------|-----|-----|
| **Newton's Method** | x_{k+1} = x_k - H⁻¹∇f | Faster convergence |
| **Loss Landscape Analysis** | Eigenvalues of H | Sharp vs flat minima |
| **Fisher Information** | Expected Hessian | Natural gradient |
| **Laplacian** | Trace of H | Image processing |
| **Mode Connectivity** | Hessian along path | Understanding DL |

---

## 💻 Computing Hessian in Code

### PyTorch

```python
import torch
from torch.autograd.functional import hessian

def f(x):
    return x[0]**2 + 3*x[1]**2

x = torch.tensor([1.0, 1.0])
H = hessian(f, x)
print(f"Hessian:\n{H}")
# [[2., 0.],
#  [0., 6.]]

```

### JAX

```python
import jax
import jax.numpy as jnp

def f(x):
    return x[0]**2 + 3*x[1]**2

hess_f = jax.hessian(f)
x = jnp.array([1.0, 1.0])
print(f"Hessian:\n{hess_f(x)}")

```

---

## ⚠️ Why We Often Avoid Hessian

| Problem | Details | Solution |
|---------|---------|----------|
| **Storage** | O(n²) memory | Too big for neural nets |
| **Computation** | O(n²) to compute | Too slow |
| **Inversion** | O(n³) to invert | Even slower |

**Solution: Quasi-Newton methods (BFGS, L-BFGS)** approximate H using only gradient info!

---

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 📖 | Khan Academy Multivariable | [Link](https://www.khanacademy.org/math/multivariable-calculus) |
| 🎥 | 3Blue1Brown Calculus | [YouTube](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) |
| 📖 | Numerical Optimization Ch.2 | [Springer](https://link.springer.com/book/10.1007/978-0-387-40065-5) |
| 🎥 | Hessian Visualized | [YouTube](https://www.youtube.com/watch?v=LbBcuZukCAw) |
| 🇨🇳 | 知乎梯度详解 | [知乎](https://zhuanlan.zhihu.com/p/25202034) |
| 🇨🇳 | 知乎 - Hessian矩阵 | [知乎](https://zhuanlan.zhihu.com/p/37688632) |

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

⬅️ [Back: Foundations](../) | ➡️ [Next: Linear Algebra](../02_linear_algebra/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
