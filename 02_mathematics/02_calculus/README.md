<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=120&section=header&text=Calculus&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-02-00C853?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/gradient-descent.svg" width="100%">

*Caption: Gradient descent follows the negative gradient to find minima. The gradient ∇f points in the direction of steepest ascent, so we move in the opposite direction. This is the core algorithm behind all neural network training.*

---

## 📐 Mathematical Foundations

### Derivatives

```
Single variable:
df/dx = lim_{h→0} (f(x+h) - f(x)) / h

Partial derivative:
∂f/∂xᵢ = lim_{h→0} (f(x + heᵢ) - f(x)) / h

```

### Gradient Vector

```
∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ

Properties:
• Points in direction of steepest ascent
• |∇f| = rate of steepest increase
• ∇f ⊥ level sets

```

### Jacobian and Hessian

```
For f: ℝⁿ → ℝᵐ:
J = [∂fᵢ/∂xⱼ]  (m × n matrix)

For f: ℝⁿ → ℝ:
H = [∂²f/∂xᵢ∂xⱼ]  (n × n matrix)

Positive definite H ⟹ local minimum

```

### Chain Rule (Multivariate)

```
If y = f(u) and u = g(x):
∂y/∂x = (∂y/∂u)(∂u/∂x) = Jacobians multiply!

Backpropagation:
∂L/∂W = ∂L/∂y · ∂y/∂z · ∂z/∂W

```

---

## 📐 DETAILED CALCULUS MATHEMATICS

### 1. Multivariable Chain Rule: Complete Derivation

**Single Variable Chain Rule:**

```
If y = f(u) and u = g(x), then:
  dy/dx = dy/du · du/dx

Proof:
  dy/dx = lim_{h→0} (f(g(x+h)) - f(g(x))) / h
  
  Let Δu = g(x+h) - g(x)
  
  = lim_{h→0} (f(g(x)+Δu) - f(g(x))) / h
  = lim_{h→0} [(f(g(x)+Δu) - f(g(x))) / Δu] · [Δu / h]
  = lim_{h→0} (f(g(x)+Δu) - f(g(x))) / Δu · lim_{h→0} Δu/h
  = f'(g(x)) · g'(x)
  = dy/du · du/dx  ✓

```

**Vector Chain Rule (Key for Backpropagation):**

```
Case 1: Scalar function of vector
  y = f(u), u = g(x)
  y ∈ ℝ, u ∈ ℝᵐ, x ∈ ℝⁿ

Chain rule:
  ∂y/∂xⱼ = Σᵢ (∂y/∂uᵢ)(∂uᵢ/∂xⱼ)

Matrix form:
  ∇ₓy = (∂u/∂x)ᵀ ∇ᵤy
      = Jᵤ,ₓᵀ · ∇ᵤy

```

**Proof of Vector Chain Rule:**

```
Step 1: Taylor expansion
  y(x + Δx) ≈ y(x) + ∇ₓy · Δx

Step 2: Express through intermediate variable
  u(x + Δx) ≈ u(x) + Jᵤ,ₓ · Δx
  y(u + Δu) ≈ y(u) + ∇ᵤy · Δu

Step 3: Substitute
  y(x + Δx) ≈ y(x) + ∇ᵤy · (Jᵤ,ₓ · Δx)
              = y(x) + (∇ᵤy)ᵀJᵤ,ₓ · Δx
              = y(x) + (Jᵤ,ₓᵀ∇ᵤy) · Δx

Step 4: Compare with Taylor expansion
  ∇ₓy = Jᵤ,ₓᵀ · ∇ᵤy  ✓

This is the foundation of backpropagation!

```

---

### 2. Backpropagation as Repeated Chain Rule

**Neural Network:**

```
Input:  x ∈ ℝᵈ⁰
Layer 1: z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾ ∈ ℝᵈ¹
         a⁽¹⁾ = σ(z⁽¹⁾)
Layer 2: z⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾ ∈ ℝᵈ²
         a⁽²⁾ = σ(z⁽²⁾)
...
Layer L: z⁽ᴸ⁾ = W⁽ᴸ⁾a⁽ᴸ⁻¹⁾ + b⁽ᴸ⁾
         ŷ = a⁽ᴸ⁾ = σ(z⁽ᴸ⁾)

Loss: L = loss(ŷ, y)

```

**Goal: Compute ∂L/∂W⁽ˡ⁾ for all layers l**

**Step-by-Step Chain Rule:**

```
Layer L (output):
  ∂L/∂z⁽ᴸ⁾ = ∂L/∂ŷ · ∂ŷ/∂z⁽ᴸ⁾
           = ∂L/∂ŷ ⊙ σ'(z⁽ᴸ⁾)

Layer l < L (hidden):
  ∂L/∂z⁽ˡ⁾ = ∂L/∂z⁽ˡ⁺¹⁾ · ∂z⁽ˡ⁺¹⁾/∂a⁽ˡ⁾ · ∂a⁽ˡ⁾/∂z⁽ˡ⁾
           = [(W⁽ˡ⁺¹⁾)ᵀ · ∂L/∂z⁽ˡ⁺¹⁾] ⊙ σ'(z⁽ˡ⁾)

Weight gradient:
  ∂L/∂W⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾ · ∂z⁽ˡ⁾/∂W⁽ˡ⁾
           = ∂L/∂z⁽ˡ⁾ · (a⁽ˡ⁻¹⁾)ᵀ  [outer product]

```

**Formal Proof for Layer l:**

```
Let L(W⁽ˡ⁾) be the loss

Change W⁽ˡ⁾ by δW⁽ˡ⁾:
  δz⁽ˡ⁾ = δW⁽ˡ⁾ · a⁽ˡ⁻¹⁾
  δa⁽ˡ⁾ = σ'(z⁽ˡ⁾) ⊙ δz⁽ˡ⁾
  δz⁽ˡ⁺¹⁾ = W⁽ˡ⁺¹⁾ · δa⁽ˡ⁾
  ...
  δL = (∂L/∂ŷ)ᵀ · δŷ

Working backwards (chain rule):
  δL = (∂L/∂ŷ)ᵀ · δŷ
     = (∂L/∂ŷ)ᵀ · (∂ŷ/∂z⁽ᴸ⁾) · δz⁽ᴸ⁾
     = ...  [continue backwards]
     = [(∂L/∂z⁽ˡ⁾)ᵀ · δW⁽ˡ⁾] · a⁽ˡ⁻¹⁾

Therefore:
  ∂L/∂W⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾ · (a⁽ˡ⁻¹⁾)ᵀ  ✓

```

---

### 3. Jacobian Matrix: Theory and Computation

**Definition:**

```
For f: ℝⁿ → ℝᵐ, the Jacobian is:

J = [∂fᵢ/∂xⱼ] ∈ ℝᵐˣⁿ

     +                           +
     | ∂f₁/∂x₁  ∂f₁/∂x₂  ...    |
J =  | ∂f₂/∂x₁  ∂f₂/∂x₂  ...    |
     |    ⋮        ⋮      ⋱      |
     | ∂fₘ/∂x₁  ∂fₘ/∂x₂  ...    |
     +                           +

Interpretation: How each output changes with each input

```

**Chain Rule with Jacobians:**

```
If z = f(y) and y = g(x):
  z: ℝᵖ → ℝ
  y: ℝⁿ → ℝᵐ  
  x: ℝⁿ

Jacobian chain rule:
  J_{z,x} = J_{z,y} · J_{y,x}
  
Dimensions: (p×n) = (p×m) · (m×n)  ✓

```

**Example: Batch Normalization:**

```
Input: x ∈ ℝⁿ
Output: y = (x - μ) / √(σ² + ε)

where μ = mean(x), σ² = var(x)

Jacobian: ∂yᵢ/∂xⱼ = ?

Step 1: ∂μ/∂xⱼ = 1/n

Step 2: ∂σ²/∂xⱼ = 2(xⱼ - μ)/n

Step 3: Apply chain rule
  ∂yᵢ/∂xⱼ = ∂yᵢ/∂xᵢ · ∂xᵢ/∂xⱼ  [direct]
           + ∂yᵢ/∂μ · ∂μ/∂xⱼ    [through mean]
           + ∂yᵢ/∂σ² · ∂σ²/∂xⱼ  [through variance]

Full calculation:
  J = (1/√(σ²+ε)) · [I - (1/n)11ᵀ - (1/n)(x-μ)(x-μ)ᵀ/(σ²+ε)]

where 1 is vector of ones

```

---

### 4. Hessian Matrix: Second-Order Information

**Definition:**

```
For f: ℝⁿ → ℝ, the Hessian is:

H = ∇²f = [∂²f/∂xᵢ∂xⱼ] ∈ ℝⁿˣⁿ

     +                                    +
     | ∂²f/∂x₁²    ∂²f/∂x₁∂x₂   ...      |
H =  | ∂²f/∂x₂∂x₁  ∂²f/∂x₂²     ...      |
     |     ⋮            ⋮         ⋱       |
     +                                    +

If f is C²: ∂²f/∂xᵢ∂xⱼ = ∂²f/∂xⱼ∂xᵢ  [Schwarz's theorem]
Therefore: H is symmetric

```

**Taylor Series (Second-Order):**

```
f(x + h) ≈ f(x) + ∇f(x)ᵀh + ½hᵀH(x)h

This is quadratic approximation!

At critical point (∇f(x) = 0):
  f(x + h) ≈ f(x) + ½hᵀH(x)h

Classification:
  H ≻ 0 (positive definite) → local minimum
  H ≺ 0 (negative definite) → local maximum
  H indefinite              → saddle point

```

**Newton's Method:**

```
Use second-order Taylor approximation:
  f(x + h) ≈ f(x) + ∇f(x)ᵀh + ½hᵀH(x)h

Minimize w.r.t. h:
  ∇ₕ[f(x) + ∇f(x)ᵀh + ½hᵀH(x)h] = 0
  ∇f(x) + H(x)h = 0
  h = -H(x)⁻¹∇f(x)

Newton's update:
  xₙₑw = xₒₗd - H⁻¹∇f

Convergence: Quadratic near minimum
  ||xₙₑw - x*|| = O(||xₒₗd - x*||²)

Problem: O(n³) to compute H⁻¹

```

---

### 5. Directional Derivatives and Gradient

**Directional Derivative:**

```
Rate of change of f in direction v:
  D_v f(x) = lim_{t→0} (f(x + tv) - f(x)) / t

Theorem: If f differentiable, then:
  D_v f(x) = ∇f(x) · v

Proof:
  f(x + tv) ≈ f(x) + ∇f(x)ᵀ(tv)
  (f(x + tv) - f(x)) / t ≈ ∇f(x)ᵀv  ✓

```

**Gradient as Direction of Steepest Ascent:**

```
Theorem: ∇f(x) points in direction of maximum increase

Proof:
  D_v f = ∇f · v = ||∇f|| · ||v|| · cos(θ)

  Maximum when cos(θ) = 1, i.e., v ∥ ∇f

  max_||v||=1 D_v f = ||∇f||

Direction: v* = ∇f / ||∇f||  ✓

This is why gradient descent works!
  Move in -∇f direction to minimize

```

**Gradient Perpendicular to Level Sets:**

```
Theorem: ∇f(x) is perpendicular to level set

Proof:
Let γ(t) be curve on level set: f(γ(t)) = c

Differentiate:
  d/dt f(γ(t)) = ∇f(γ(t)) · γ'(t) = 0

Since γ'(t) is tangent to level set and ∇f · γ' = 0:
  ∇f ⊥ level set  ✓

Visualization:
  Level curves of f(x,y) = c
  Gradient field points perpendicular

```

---

### 6. Matrix Calculus Identities

**Scalar-by-Vector:**

```
∂/∂x (aᵀx) = a

∂/∂x (xᵀAx) = (A + Aᵀ)x
            = 2Ax  [if A symmetric]

∂/∂x ||x||² = 2x

∂/∂x ||Ax - b||² = 2Aᵀ(Ax - b)

```

**Proof (∂/∂x xᵀAx):**

```
Let f(x) = xᵀAx = Σᵢⱼ xᵢAᵢⱼxⱼ

∂f/∂xₖ = ∂/∂xₖ Σᵢⱼ xᵢAᵢⱼxⱼ
       = Σⱼ Aₖⱼxⱼ + Σᵢ xᵢAᵢₖ  [product rule]
       = (Ax)ₖ + (Aᵀx)ₖ
       = [(A + Aᵀ)x]ₖ

Therefore: ∇ₓ(xᵀAx) = (A + Aᵀ)x  ✓

If A symmetric: = 2Ax

```

**Vector-by-Matrix:**

```
∂/∂W (Wx) = xᵀ ⊗ I = xIᵀ
           (results in correct shape)

∂/∂W tr(WX) = Xᵀ

∂/∂W tr(WᵀAW) = A + Aᵀ)W
                = 2AW  [if A symmetric]

∂/∂W ||WX - Y||²_F = 2(WX - Y)Xᵀ

```

**Matrix-by-Matrix (Trace Formulation):**

```
For loss L involving matrix W:

∂L/∂W computed using:
  dL = tr((∂L/∂W)ᵀ dW)

Example: L = tr(WᵀAW)
  dL = tr((A + Aᵀ)W)ᵀ dW)
  
  Therefore: ∂L/∂W = (A + Aᵀ)W

```

---

### 7. Automatic Differentiation: Forward vs Reverse Mode

**Forward Mode (Tangent Propagation):**

```
Compute derivatives along with values

For y = f(g(x)):
  Seed: ẋ = 1 (derivative w.r.t. x)
  
  Forward:
    u = g(x),  u̇ = g'(x)ẋ
    y = f(u),  ẏ = f'(u)u̇
  
  Result: ẏ = dy/dx

Complexity: O(n) operations per input variable
Good for: Few inputs, many outputs (Jacobian rows)

```

**Reverse Mode (Backpropagation):**

```
Compute derivatives backward from output

For y = f(g(x)):
  Forward: Compute and store values
    u = g(x)
    y = f(u)
  
  Seed: ȳ = 1 (derivative of y w.r.t. itself)
  
  Backward:
    ū = ȳ · f'(u)
    x̄ = ū · g'(x)
  
  Result: x̄ = dy/dx

Complexity: O(1) operations per output variable
Good for: Many inputs, few outputs (gradient!)
  
This is why backprop is efficient for neural networks!
  Millions of parameters, single loss

```

**Comparison:**

```
Function: f: ℝⁿ → ℝᵐ

Forward Mode:
  One pass per input variable
  Total: n passes
  Computes: One column of Jacobian per pass
  Best: n << m

Reverse Mode:
  One pass per output variable
  Total: m passes
  Computes: One row of Jacobian per pass
  Best: m << n (like neural networks: m=1, n=millions)

```

**Example: Neural Network**

```
f: ℝ¹'⁰⁰⁰'⁰⁰⁰ → ℝ¹  (million params, scalar loss)

Forward mode: Need 1,000,000 passes  ✗
Reverse mode: Need 1 pass          ✓

This is the power of backpropagation!

```

---

### 8. Implicit Function Theorem

**Theorem:**

```
Let F(x, y) = 0 define y implicitly as function of x

If:
  1. F is C¹
  2. F(x₀, y₀) = 0
  3. ∂F/∂y(x₀, y₀) ≠ 0

Then near (x₀, y₀), ∃ function y = g(x) such that:
  F(x, g(x)) = 0

Derivative:
  dy/dx = -(∂F/∂x) / (∂F/∂y)

```

**Proof:**

```
Differentiate F(x, y(x)) = 0 w.r.t. x:
  ∂F/∂x + (∂F/∂y)(dy/dx) = 0

Solve for dy/dx:
  dy/dx = -(∂F/∂x) / (∂F/∂y)  ✓

```

**Application: Constrained Optimization**

```
Minimize f(x) subject to g(x) = 0

At optimum, ∇f parallel to ∇g:
  ∇f = λ∇g  [Lagrange multipliers]

Implicit function theorem explains why:
  Constraint defines manifold
  Gradient perpendicular to feasible directions

```

---

### 9. Mean Value Theorem and Lipschitz Continuity

**Mean Value Theorem:**

```
If f: [a,b] → ℝ is continuous on [a,b] and differentiable on (a,b):
  ∃c ∈ (a,b) such that:
    f'(c) = (f(b) - f(a)) / (b - a)

Geometric: ∃ point where tangent parallel to chord

```

**Multivariable Version:**

```
For f: ℝⁿ → ℝ:
  f(x + h) - f(x) = ∇f(x + th) · h  for some t ∈ [0,1]

Used extensively in convergence proofs!

```

**Lipschitz Continuity:**

```
f is L-Lipschitz if:
  ||f(x) - f(y)|| ≤ L||x - y||  for all x, y

L-smooth (differentiable version):
  ||∇f(x) - ∇f(y)|| ≤ L||x - y||

Implications:
  • Bounded gradient: ||∇f(x)|| ≤ L||x|| + C
  • Predictable behavior
  • Convergence guarantees for optimization

```

**Application to Neural Networks:**

```
Activation functions:
  ReLU: 1-Lipschitz (gradient ∈ {0,1})
  Sigmoid: 0.25-Lipschitz (max gradient = 1/4)
  Tanh: 1-Lipschitz (max gradient = 1)

Network: f = f_L ∘ ... ∘ f_1

Lipschitz constant:
  L(f) ≤ L(f_L) · ... · L(f_1)
       ≤ ||W_L|| · ... · ||W_1||  [for linear layers]

This explains exploding/vanishing gradients!
  If ||W|| > 1: gradients explode
  If ||W|| < 1: gradients vanish

```

---

## 📂 Topics in This Folder

| Folder | Topics | ML Application |
|--------|--------|----------------|
| [limits-continuity/](./limits-continuity/) | Limits, continuity | Convergence analysis |
| [derivatives/](./derivatives/) | Single, partial, higher-order | Gradient computation |
| [gradients/](./gradients/) | ∇f, Jacobian, Hessian | 🔥 Core of ML training! |
| [chain-rule/](./chain-rule/) | Composition, backprop | 🔥 How neural nets learn! |
| [taylor/](./taylor/) | Series, approximations | Convergence rates |
| [integration/](./integration/) | Single, multiple, Monte Carlo | Expectation, sampling |

---

## 🎯 Why Calculus is Essential for ML

```
Training a Neural Network:

1. Forward pass: compute ŷ = f(x; θ)
2. Loss: L = loss(ŷ, y)
3. Backward pass: ∂L/∂θ = ???    ← Need calculus!
4. Update: θ ← θ - α · ∂L/∂θ    ← Gradient descent!

The entire deep learning revolution is built on:
• Chain rule → Backpropagation
• Gradients → Optimization
• Taylor → Convergence analysis

```

---

## 🌍 ML Applications

| Concept | Application | Example |
|---------|-------------|---------|
| Gradient ∇f | Parameter updates | SGD, Adam |
| Jacobian | Multi-output models | Normalizing flows |
| Hessian | Curvature | Newton's method, Fisher |
| Chain rule | Backpropagation | All deep learning |
| Taylor | Local approximation | Convergence proofs |
| Monte Carlo | Expected gradients | Variational inference |

---

## 🔗 Dependency Graph

```
limits-continuity/
        |
        v
    derivatives/
        |
        +--> gradients/     --> All ML optimization
        |         |
        |                                                               ▼
        +--> chain-rule/    --> Backpropagation
                  |
                  v
             taylor/        --> Convergence analysis

```

---

## 🔑 Key Formulas

```
Gradient:      ∇f = (∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ)ᵀ

Jacobian:      J = [∂fᵢ/∂xⱼ]  (m × n matrix)

Hessian:       H = [∂²f/∂xᵢ∂xⱼ]  (n × n matrix)

Chain rule:    d/dx f(g(x)) = f'(g(x)) · g'(x)

Taylor (1st):  f(x + h) ≈ f(x) + ∇f(x)ᵀh

Taylor (2nd):  f(x + h) ≈ f(x) + ∇f(x)ᵀh + ½hᵀH(x)h

```

---

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 📖 | Calculus on Manifolds | Spivak |
| 🎥 | Essence of Calculus | [3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) |
| 📖 | Matrix Calculus for DL | [Paper](https://arxiv.org/abs/1802.01528) |

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

⬅️ [Back: 01-Linear Algebra](../01_linear_algebra/) | ➡️ [Next: 03-Optimization](../03_optimization/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=80&section=footer" width="100%"/>
</p>
