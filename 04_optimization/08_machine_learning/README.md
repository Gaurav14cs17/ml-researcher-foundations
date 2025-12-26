<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=Machine%20Learning%20Optimization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📂 Subtopics

| Folder | Topic | Used In |
|--------|-------|---------|
| [01_adam/](./01_adam/) | Adam Optimizer | GPT, Stable Diffusion |
| [02_sgd/](./02_sgd/) | SGD & Variants | ResNet, BERT |

---

## 🎯 Why ML Optimization is Special

```
+---------------------------------------------------------+
|                                                         |
|   Classical Optimization:                               |
|   • Full gradient ∇f(x) available                       |
|   • Single objective                                    |
|   • Compute-bound                                       |
|                                                         |
|   Machine Learning:                                     |
|   • Stochastic gradients (mini-batch)                   |
|   • Non-convex (neural networks)                        |
|   • Billions of parameters                              |
|   • Memory-bound                                        |
|                                                         |
+---------------------------------------------------------+
```

---

# Part 1: Stochastic Gradient Descent (SGD)

## 📐 Mathematical Formulation

### The Problem

```
minimize  f(θ) = (1/n) Σᵢ fᵢ(θ)

where fᵢ = loss on data point i
      n = dataset size (millions/billions)
      θ = model parameters (millions/billions)

Computing full gradient is expensive: O(n)
```

### SGD Update Rule

```
θₜ₊₁ = θₜ - η ∇fᵢₜ(θₜ)

where iₜ is randomly sampled from {1,...,n}

Key insight: E[∇fᵢ(θ)] = ∇f(θ)  (unbiased!)
```

### Mini-batch SGD

```
θₜ₊₁ = θₜ - η · (1/|B|) Σᵢ∈B ∇fᵢ(θₜ)

where B = mini-batch of size b

Properties:
• Variance ∝ 1/b
• Parallelizable (GPU-friendly)
• Typical b: 32, 64, 128, 256, 512
```

---

## 📐 Convergence Analysis

### Assumptions

```
1. L-smoothness: ||∇f(x) - ∇f(y)|| ≤ L||x - y||
2. Bounded variance: E[||∇fᵢ(θ) - ∇f(θ)||²] ≤ σ²
3. (Optional) μ-strong convexity: f(y) ≥ f(x) + ∇f(x)ᵀ(y-x) + (μ/2)||y-x||²
```

### Convex Case: Convergence Theorem

**Theorem:** For L-smooth convex f with bounded variance σ², SGD with η = 1/(L√T) achieves:

```
E[f(θ̄ₜ)] - f* ≤ O(||θ₀ - θ*||²L/T + σ||θ₀ - θ*||/√T)

= O(1/√T)
```

**Proof:**

```
Step 1: Smoothness bound
f(θₜ₊₁) ≤ f(θₜ) + ∇f(θₜ)ᵀ(θₜ₊₁ - θₜ) + (L/2)||θₜ₊₁ - θₜ||²
       = f(θₜ) - η∇f(θₜ)ᵀgₜ + (Lη²/2)||gₜ||²

where gₜ = ∇fᵢₜ(θₜ) is stochastic gradient

Step 2: Take expectation
E[f(θₜ₊₁)] ≤ E[f(θₜ)] - ηE[∇f(θₜ)ᵀgₜ] + (Lη²/2)E[||gₜ||²]

Since E[gₜ|θₜ] = ∇f(θₜ):
E[∇f(θₜ)ᵀgₜ] = E[||∇f(θₜ)||²]

And E[||gₜ||²] = ||∇f(θₜ)||² + σ² (variance decomposition)

Step 3: Simplify
E[f(θₜ₊₁)] ≤ E[f(θₜ)] - η(1 - Lη/2)E[||∇f(θₜ)||²] + (Lη²σ²/2)

Step 4: With η ≤ 1/L and convexity
||∇f(θₜ)||² ≥ 2μ(f(θₜ) - f*)  (PL condition for strongly convex)

Step 5: Telescope sum over T iterations
For η = 1/(L√T), summing and using convexity of f(θ̄):

E[f(θ̄)] - f* ≤ O(1/√T)  ∎
```

### Strongly Convex Case

```
With μ-strong convexity and η = 1/(μt):

E[f(θₜ)] - f* ≤ O(σ²/(μT))

Linear convergence to neighborhood of optimum!
```

### Non-Convex Case

```
For L-smooth non-convex f:

(1/T) Σₜ E[||∇f(θₜ)||²] ≤ O((f(θ₀) - f*)/(ηT) + Lησ²)

SGD finds approximate stationary point!
(but may be saddle or local min)
```

---

## 📐 SGD with Momentum

### Update Rule

```
vₜ = γvₜ₋₁ + ∇f(θₜ)
θₜ₊₁ = θₜ - ηvₜ

Or equivalently:
vₜ = γvₜ₋₁ + η∇f(θₜ)
θₜ₊₁ = θₜ - vₜ

where γ ∈ [0.9, 0.99] typically
```

### Why Momentum Helps

```
Physical intuition: Ball rolling down hill

Without momentum:          With momentum:
        ↓                       ↓↓↓
       ╱ ╲                     ╱   ╲
      ╱   ╲                   ╱     ╲
     ╱  ↓  ╲                 ╱ ↓↓↓   ╲
    ╱       ╲               ╱         ╲
   
   Oscillates in            Accelerates in
   narrow valleys           consistent direction
```

### Convergence Improvement

```
For quadratic f(x) = (1/2)xᵀAx with eigenvalues λ_min ≤ ... ≤ λ_max:

Without momentum: κ = λ_max/λ_min iterations
With momentum:    √κ iterations

Optimal γ = (√κ - 1)/(√κ + 1) ≈ 1 - 2/√κ
```

---

## 💻 SGD Implementation

```python
import numpy as np

class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(p) for p in params]
    
    def step(self, grads):
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Weight decay (L2 regularization)
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            
            # Momentum
            self.velocity[i] = self.momentum * self.velocity[i] + grad
            
            # Update
            param -= self.lr * self.velocity[i]

# PyTorch equivalent
import torch

optimizer = torch.optim.SGD(
    model.parameters(), 
    lr=0.01, 
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True  # Nesterov momentum
)

# Training loop
for x, y in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

### Nesterov Momentum

```
"Look ahead" before computing gradient:

θ_lookahead = θₜ - γvₜ₋₁
vₜ = γvₜ₋₁ + η∇f(θ_lookahead)
θₜ₊₁ = θₜ - vₜ

Intuition: Correct momentum direction using future position
Achieves optimal O(1/T²) rate for convex problems!
```

---

# Part 2: Adam Optimizer

## 📐 The Adam Algorithm

```
Adam = Adaptive Moment Estimation

Combines:
• Momentum (first moment)
• RMSprop (second moment)
• Bias correction

Update rule:
mₜ = β₁mₜ₋₁ + (1-β₁)gₜ           (first moment estimate)
vₜ = β₂vₜ₋₁ + (1-β₂)gₜ²          (second moment estimate)
m̂ₜ = mₜ/(1-β₁ᵗ)                   (bias-corrected first)
v̂ₜ = vₜ/(1-β₂ᵗ)                   (bias-corrected second)
θₜ₊₁ = θₜ - η m̂ₜ/(√v̂ₜ + ε)

Default values:
β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸
```

---

## 📐 Why Bias Correction?

```
Problem: Initial moments are biased toward zero

m₀ = 0, v₀ = 0

After t steps:
E[mₜ] = (1-β₁ᵗ) E[g]  ≠ E[g]  (biased!)
E[vₜ] = (1-β₂ᵗ) E[g²] ≠ E[g²] (biased!)

Bias correction fixes this:
E[m̂ₜ] = E[mₜ]/(1-β₁ᵗ) = E[g]    ✓
E[v̂ₜ] = E[vₜ]/(1-β₂ᵗ) = E[g²]   ✓
```

### Derivation of Bias Correction

```
mₜ = (1-β₁) Σᵢ₌₁ᵗ β₁ᵗ⁻ⁱ gᵢ

E[mₜ] = (1-β₁) Σᵢ₌₁ᵗ β₁ᵗ⁻ⁱ E[g]
      = E[g] (1-β₁) (1-β₁ᵗ)/(1-β₁)
      = E[g] (1-β₁ᵗ)

Therefore:
E[mₜ/(1-β₁ᵗ)] = E[g]  ✓
```

---

## 📐 Adam Convergence

**Theorem (Kingma & Ba, 2015):** For convex f with bounded gradients, Adam achieves:

```
Regret ≤ O(d√T)

where d = dimension, T = iterations

Equivalent to O(1/√T) convergence rate
```

**Proof Sketch:**

```
Step 1: Define regret
Rₜ = Σₛ₌₁ᵗ (f(θₛ) - f(θ*))

Step 2: Per-step bound (using online learning analysis)
f(θₜ) - f(θ*) ≤ ⟨gₜ, θₜ - θ*⟩

Step 3: Adaptive learning rate helps
With v̂ₜ tracking gradient magnitudes:
• Large gradients → smaller effective lr → stability
• Small gradients → larger effective lr → faster progress

Step 4: Bound via potential function
Careful analysis of ||θₜ - θ*||²_diag(√v̂ₜ)
yields O(d√T) regret ∎
```

---

## 📐 Adam vs SGD: The Great Debate

```
+---------------------------------------------------------+
|                                                         |
|   Adam wins:                     SGD wins:              |
|   ----------                     ---------              |
|   • Faster initial progress      • Better final acc     |
|   • Less lr tuning               • Simpler analysis     |
|   • Sparse gradients             • Better generalization|
|   • NLP, transformers            • Vision (sometimes)   |
|                                                         |
+---------------------------------------------------------+

Why SGD generalizes better (conjecture):
• Adam finds "sharp" minima (poor generalization)
• SGD's noise helps find "flat" minima (good generalization)
```

---

## 📐 AdamW: Weight Decay Done Right

```
Problem with Adam + L2 regularization:

Standard Adam:
g' = g + λθ
mₜ = β₁mₜ₋₁ + (1-β₁)g'

The regularization gets scaled by √v̂ₜ, weakening it!

AdamW (decoupled weight decay):
mₜ = β₁mₜ₋₁ + (1-β₁)gₜ  (no λθ here)
θₜ₊₁ = θₜ - η(m̂ₜ/(√v̂ₜ + ε) + λθₜ)  (add separately)

This is the standard for modern transformers!
```

---

## 💻 Adam Implementation

```python
import numpy as np

class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), 
                 eps=1e-8, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
    
    def step(self, grads):
        self.t += 1
        
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Update biased first moment
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second moment
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            # Update with AdamW-style weight decay
            param -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + 
                               self.weight_decay * param)

# PyTorch
import torch

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01
)

# With learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs
)

for epoch in range(num_epochs):
    for x, y in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    scheduler.step()
```

---

## 📐 Other Adaptive Methods

### RMSprop

```
vₜ = γvₜ₋₁ + (1-γ)gₜ²
θₜ₊₁ = θₜ - η gₜ/√(vₜ + ε)

Adam's predecessor, no momentum or bias correction
```

### AdaGrad

```
vₜ = vₜ₋₁ + gₜ²
θₜ₊₁ = θₜ - η gₜ/√(vₜ + ε)

Problem: Learning rate decays to zero
Good for sparse gradients
```

### Comparison

| Optimizer | Momentum | Adaptive LR | Bias Correct | Best For |
|-----------|----------|-------------|--------------|----------|
| SGD | Optional | No | N/A | Vision |
| AdaGrad | No | Yes | No | Sparse |
| RMSprop | No | Yes | No | RNNs |
| Adam | Yes | Yes | Yes | Transformers |
| AdamW | Yes | Yes | Yes | Modern DL |

---

## 📐 Learning Rate Schedules

```
Warmup + Decay (standard for transformers):

η(t) = { η_max · t/T_warmup           if t < T_warmup
       { η_max · cos(π(t-T_warmup)/(2T_total))  otherwise

Why warmup?
• Adam's m, v need time to initialize
• Large initial gradients can destabilize

Why decay?
• Helps convergence to better minima
• Reduces final oscillation
```

---

## 📊 Convergence Bounds Summary

| Algorithm | Convex | Strongly Convex | Non-Convex |
|-----------|--------|-----------------|------------|
| **GD** | O(1/T) | O(exp(-T)) | O(1/T) |
| **SGD** | O(1/√T) | O(1/T) | O(1/√T) |
| **SGD+Momentum** | O(1/T²) | O(exp(-√T)) | O(1/√T) |
| **Adam** | O(1/√T) | O(1/√T) | O(1/√T) |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Adam Paper | [arXiv](https://arxiv.org/abs/1412.6980) |
| 📄 | AdamW Paper | [arXiv](https://arxiv.org/abs/1711.05101) |
| 📄 | On the Convergence of Adam | [arXiv](https://arxiv.org/abs/1904.09237) |
| 📖 | Deep Learning Book Ch 8 | [Book](https://www.deeplearningbook.org/) |
| 🎥 | Stanford CS231n | [Optimization Lecture](http://cs231n.stanford.edu/) |
| 🇨🇳 | 优化器详解 | [知乎](https://zhuanlan.zhihu.com/p/32230623) |
| 🇨🇳 | Adam原理与实现 | [知乎](https://zhuanlan.zhihu.com/p/32626442) |

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Computer Vision** | ResNet, ViT training |
| **NLP** | BERT, GPT, LLaMA |
| **Generative AI** | Diffusion models, GANs |
| **Reinforcement Learning** | Policy optimization |

---

⬅️ [Back: Integer Programming](../07_integer_programming/) | ➡️ [Next: Metaheuristics](../09_metaheuristics/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
