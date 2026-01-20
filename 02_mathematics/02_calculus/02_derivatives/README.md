<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Derivatives&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=32&desc=The%20Foundation%20of%20Calculus%20and%20Deep%20Learning&descAlignY=52&descSize=16" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/📚_Section-02.02_Derivatives-00C853?style=for-the-badge" alt="Section"/>
  <img src="https://img.shields.io/badge/📊_Topics-Limits_Rules_Activations-blue?style=for-the-badge" alt="Topics"/>
  <img src="https://img.shields.io/badge/✍️_Author-Gaurav_Goswami-purple?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/📅_Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## ⚡ TL;DR

> **Derivatives measure instantaneous rate of change.** Every neural network parameter update uses derivatives computed via the chain rule (backpropagation).

- 📐 **Definition**: $f'(x) = \lim\_{h \to 0} \frac{f(x+h) - f(x)}{h}$
- 📊 **Interpretation**: Slope of tangent line, sensitivity of output to input
- 🔗 **Chain Rule**: $(f \circ g)' = f'(g(x)) \cdot g'(x)$ — the backbone of backprop
- 🧠 **In DL**: Every `loss.backward()` computes derivatives of loss w.r.t. all parameters

---

## 📑 Table of Contents

1. [Definition and Intuition](#1-definition-and-intuition)
2. [Differentiation Rules](#2-differentiation-rules)
3. [Common Derivatives](#3-common-derivatives)
4. [Activation Function Derivatives](#4-activation-function-derivatives)
5. [Partial Derivatives](#5-partial-derivatives)
6. [Code Implementation](#6-code-implementation)
7. [Resources](#-resources)

---

## 🎨 Visual Overview

<img src="./images/derivatives.svg" width="100%">

```
+-----------------------------------------------------------------------------+
|                    DERIVATIVE = INSTANTANEOUS SLOPE                          |
+-----------------------------------------------------------------------------+
|                                                                              |
|   y = f(x)                                                                   |
|   |                                                                          |
|   |          ╱ tangent line (slope = f'(x))                                 |
|   |        ╱                                                                 |
|   |      ╱      ╱ f(x)                                                      |
|   |    ╱    __╱                                                              |
|   |  ╱   __╱                                                                 |
|   |╱  __╱                                                                    |
|   +-╱------------------------------ x                                       |
|                                                                              |
|   INTERPRETATION:                                                            |
|   ---------------                                                            |
|   f'(x) > 0  →  f is increasing at x                                        |
|   f'(x) < 0  →  f is decreasing at x                                        |
|   f'(x) = 0  →  x is a critical point (min/max/saddle)                      |
|                                                                              |
|   IN DEEP LEARNING:                                                          |
|   -----------------                                                          |
|   ∂L/∂θ = "how much does the loss change when I wiggle θ?"                  |
|                                                                              |
|   If ∂L/∂θ > 0: Decrease θ to decrease loss                                 |
|   If ∂L/∂θ < 0: Increase θ to decrease loss                                 |
|                                                                              |
+-----------------------------------------------------------------------------+
```

---

## 1. Definition and Intuition

### 📌 Formal Definition

```math
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
```

Alternative notation: $\frac{df}{dx}$, $\frac{dy}{dx}$, $Df$, $\dot{f}$

### 💡 Intuition

```
Derivative = Instantaneous rate of change

Physical examples:
  • Position → Velocity (derivative of position)
  • Velocity → Acceleration (derivative of velocity)
  • Loss → Gradient (derivative of loss w.r.t. parameters)

Key insight:
  The derivative tells you how sensitive the output is to small changes in input.
  
  If f'(x) = 5, then increasing x by 0.01 increases f(x) by approximately 0.05
```

### 🔍 Proof: Derivative of $x^2$

```
f(x) = x²

f'(x) = lim[h→0] (f(x+h) - f(x)) / h
      = lim[h→0] ((x+h)² - x²) / h
      = lim[h→0] (x² + 2xh + h² - x²) / h
      = lim[h→0] (2xh + h²) / h
      = lim[h→0] (2x + h)
      = 2x  ∎
```

---

## 2. Differentiation Rules

### 📊 Basic Rules

| Rule | Formula | Example |
|------|---------|---------|
| **Constant** | $\frac{d}{dx}[c] = 0$ | $\frac{d}{dx}[5] = 0$ |
| **Power** | $\frac{d}{dx}[x^n] = nx^{n-1}$ | $\frac{d}{dx}[x^3] = 3x^2$ |
| **Sum** | $(f + g)' = f' + g'$ | $(x^2 + x)' = 2x + 1$ |
| **Product** | $(fg)' = f'g + fg'$ | $(x \cdot e^x)' = e^x + xe^x$ |
| **Quotient** | $(\frac{f}{g})' = \frac{f'g - fg'}{g^2}$ | $(\frac{x}{x+1})' = \frac{1}{(x+1)^2}$ |
| **Chain** | $(f \circ g)' = f'(g(x)) \cdot g'(x)$ | $(\sin(x^2))' = \cos(x^2) \cdot 2x$ |

### 🔍 Proof: Product Rule

```
(fg)'(x) = lim[h→0] (f(x+h)g(x+h) - f(x)g(x)) / h

Add and subtract f(x+h)g(x):
= lim[h→0] (f(x+h)g(x+h) - f(x+h)g(x) + f(x+h)g(x) - f(x)g(x)) / h

= lim[h→0] [f(x+h) · (g(x+h) - g(x))/h + g(x) · (f(x+h) - f(x))/h]

= f(x) · g'(x) + g(x) · f'(x)  ∎
```

---

## 3. Common Derivatives

### 📊 Standard Functions

| Function | Derivative | Domain |
|----------|------------|--------|
| $x^n$ | $nx^{n-1}$ | $\mathbb{R}$ |
| $e^x$ | $e^x$ | $\mathbb{R}$ |
| $\ln(x)$ | $1/x$ | $x > 0$ |
| $\sin(x)$ | $\cos(x)$ | $\mathbb{R}$ |
| $\cos(x)$ | $-\sin(x)$ | $\mathbb{R}$ |
| $\tan(x)$ | $\sec^2(x)$ | $x \neq \frac{\pi}{2} + k\pi$ |
| $a^x$ | $a^x \ln(a)$ | $\mathbb{R}$ |
| $\log\_a(x)$ | $\frac{1}{x \ln(a)}$ | $x > 0$ |

### 💡 Examples

**Example 1**: Polynomial
```
f(x) = 3x⁴ - 2x³ + x - 7
f'(x) = 12x³ - 6x² + 1
```

**Example 2**: Exponential
```
f(x) = e^(2x + 1)
f'(x) = e^(2x+1) · 2 = 2e^(2x+1)  (chain rule)
```

**Example 3**: Logarithm
```
f(x) = ln(x² + 1)
f'(x) = (1/(x²+1)) · 2x = 2x/(x²+1)  (chain rule)
```

---

## 4. Activation Function Derivatives

### 📊 Critical for Backpropagation

| Activation | Formula | Derivative | Notes |
|------------|---------|------------|-------|
| **ReLU** | $\max(0, x)$ | $\begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases}$ | Undefined at 0, use subgradient |
| **Leaky ReLU** | $\max(\alpha x, x)$ | $\begin{cases} 1 & x > 0 \\ \alpha & x < 0 \end{cases}$ | $\alpha \approx 0.01$ |
| **Sigmoid** | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1 - \sigma(x))$ | Max at $x=0$: 0.25 |
| **Tanh** | $\tanh(x)$ | $1 - \tanh^2(x)$ | Centered at 0 |
| **Softmax** | $\frac{e^{x\_i}}{\sum\_j e^{x\_j}}$ | $p\_i(\delta\_{ij} - p\_j)$ | Jacobian matrix |
| **GELU** | $x \cdot \Phi(x)$ | $\Phi(x) + x\phi(x)$ | Used in Transformers |

### 🔍 Proof: Sigmoid Derivative

```
σ(x) = 1/(1 + e^(-x))

Let u = 1 + e^(-x), so σ = 1/u

dσ/dx = d(u^(-1))/dx
      = -u^(-2) · du/dx
      = -1/(1 + e^(-x))² · (-e^(-x))
      = e^(-x)/(1 + e^(-x))²

Now simplify:
= e^(-x)/(1 + e^(-x))² 
= [1/(1 + e^(-x))] · [e^(-x)/(1 + e^(-x))]
= σ(x) · [e^(-x)/(1 + e^(-x))]

Note: e^(-x)/(1 + e^(-x)) = 1 - 1/(1 + e^(-x)) = 1 - σ(x)

Therefore: σ'(x) = σ(x)(1 - σ(x))  ∎
```

### 📐 Vanishing Gradient Problem

```
Sigmoid derivative: σ'(x) = σ(x)(1-σ(x))

Maximum value: σ'(0) = 0.5 × 0.5 = 0.25

In deep networks with n layers:
  Gradient ∝ (0.25)^n

  n = 10 layers: 0.25^10 ≈ 10^(-6)  (gradient vanishes!)

This is why ReLU works better:
  ReLU'(x) = 1 for x > 0 (no vanishing!)
```

---

## 5. Partial Derivatives

### 📌 Definition

For $f(x\_1, x\_2, \ldots, x\_n)$:

```math
\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_n)}{h}
```

*Differentiate with respect to one variable, treating others as constants.*

### 💡 Example

```
f(x, y) = x²y + 3xy² + y³

∂f/∂x = 2xy + 3y²  (treat y as constant)

∂f/∂y = x² + 6xy + 3y²  (treat x as constant)
```

### 📐 Gradient

The **gradient** collects all partial derivatives:

```math
\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}
```

---

## 6. Code Implementation

```python
import torch
import numpy as np

# ============================================================
# AUTOMATIC DIFFERENTIATION
# ============================================================

# PyTorch autograd
x = torch.tensor(2.0, requires_grad=True)
y = x**3 + 2*x**2 - x + 1  # f(x) = x³ + 2x² - x + 1

y.backward()
print(f"f(2) = {y.item()}")
print(f"f'(2) = {x.grad.item()}")  # 3x² + 4x - 1 at x=2 = 12 + 8 - 1 = 19

# ============================================================
# NUMERICAL DIFFERENTIATION
# ============================================================

def numerical_derivative(f, x, h=1e-5):
    """Central difference approximation."""
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_second_derivative(f, x, h=1e-5):
    """Second derivative approximation."""
    return (f(x + h) - 2*f(x) + f(x - h)) / h**2

f = lambda x: x**3 + 2*x**2 - x + 1
print(f"Numerical f'(2): {numerical_derivative(f, 2.0):.6f}")  # ≈ 19
print(f"Numerical f''(2): {numerical_second_derivative(f, 2.0):.6f}")  # 6x + 4 = 16

# ============================================================
# ACTIVATION FUNCTION DERIVATIVES
# ============================================================

class Activations:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x):
        s = Activations.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x)**2
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

# Verify derivatives numerically
x = np.linspace(-2, 2, 100)
act = Activations()

# Sigmoid verification
numerical_grad = numerical_derivative(act.sigmoid, 0.5)
analytical_grad = act.sigmoid_derivative(0.5)
print(f"Sigmoid derivative at 0.5:")
print(f"  Numerical: {numerical_grad:.6f}")
print(f"  Analytical: {analytical_grad:.6f}")

# ============================================================
# PARTIAL DERIVATIVES
# ============================================================

def compute_partial_derivatives():
    """Compute partial derivatives using PyTorch."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    
    # f(x, y) = x²y + xy²
    f = x[0]**2 * x[1] + x[0] * x[1]**2
    f.backward()
    
    print(f"f(1, 2) = {f.item()}")
    print(f"∂f/∂x at (1,2) = {x.grad[0].item()}")  # 2xy + y² = 4 + 4 = 8
    print(f"∂f/∂y at (1,2) = {x.grad[1].item()}")  # x² + 2xy = 1 + 4 = 5

compute_partial_derivatives()

# ============================================================
# HIGHER-ORDER DERIVATIVES
# ============================================================

def higher_order_example():
    """Compute second derivative using autograd."""
    x = torch.tensor(2.0, requires_grad=True)
    
    # f(x) = x⁴
    y = x**4
    
    # First derivative: f'(x) = 4x³
    grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
    print(f"f'(2) = {grad1.item()}")  # 4*8 = 32
    
    # Second derivative: f''(x) = 12x²
    grad2 = torch.autograd.grad(grad1, x)[0]
    print(f"f''(2) = {grad2.item()}")  # 12*4 = 48

higher_order_example()
```

---

## 📚 Resources

| Type | Resource | Description |
|------|----------|-------------|
| 🎥 | [3Blue1Brown: Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) | Beautiful visual explanations |
| 📖 | [MIT 18.01 Single Variable Calculus](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/) | Complete course |
| 📖 | [Khan Academy Calculus](https://www.khanacademy.org/math/calculus-1) | Free lessons |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [Chain Rule](../01_chain_rule/README.md) | [Calculus](../README.md) | [Gradients](../03_gradients/README.md) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer&animation=twinkling" width="100%"/>
</p>
