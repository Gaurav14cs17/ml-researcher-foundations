<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=Trust%20Region%20Policy%20Optimization&fontSize=28&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🔗 Navigation

⬅️ [Back: PPO](../03_ppo/) | ➡️ [Next: Exploration](../../04_exploration/)

---

## 🎯 Visual Overview

<img src="./images/trpo.svg" width="100%">

*Caption: TRPO prevents policy collapse by constraining updates to a trust region defined by KL divergence. This ensures monotonic improvement.*

---

## 📂 Overview

TRPO is a policy gradient method that guarantees monotonic improvement by restricting how much the policy can change at each update. It provides the theoretical foundation for PPO.

---

## 📐 Mathematical Foundation

### The Policy Optimization Problem

The goal is to find a policy that maximizes expected return:

```
max_π J(π) = E_{τ~π}[Σ_t γ^t r_t]

```

The challenge: How do we update the policy without making it worse?

### Conservative Policy Iteration Bound

**Theorem (Kakade & Langford 2002):**

```
For any two policies π and π':

J(π') ≥ J(π) + Σ_s ρ_π(s) Σ_a π'(a|s) A_π(s,a) - C · max_s KL(π'(·|s) || π(·|s))

Where:
• ρ_π(s) = Σ_t γ^t P(s_t = s | π) is the discounted state visitation
• A_π(s,a) = Q_π(s,a) - V_π(s) is the advantage
• C = 2γε_max / (1-γ)² where ε_max = max_s KL(π' || π)

```

This bound guarantees improvement if we control the KL divergence!

---

## 📐 TRPO Objective

### Surrogate Objective

Instead of maximizing J(π) directly, TRPO maximizes a surrogate:

```
L(θ) = E_{s~ρ_{θ_old}, a~π_{θ_old}} [π_θ(a|s) / π_{θ_old}(a|s) · A_{θ_old}(s,a)]
     = E [ρ(θ) · A]

Where ρ(θ) = π_θ(a|s) / π_{θ_old}(a|s) is the probability ratio.

```

### Trust Region Constraint

```
TRPO Optimization Problem:

max_θ  L(θ) = E[ρ(θ) · A]

subject to:  E_s[KL(π_{θ_old}(·|s) || π_θ(·|s))] ≤ δ

Where δ is the trust region size (typically 0.01).

```

---

## 📐 Derivation of TRPO Update

### Step 1: Linear Approximation

Near θ_old, the objective is approximately linear:

```
L(θ) ≈ L(θ_old) + g^T(θ - θ_old)

Where g = ∇_θ L(θ)|_{θ=θ_old}

```

### Step 2: Quadratic Approximation of Constraint

The KL divergence is approximately quadratic:

```
KL(π_{θ_old} || π_θ) ≈ ½(θ - θ_old)^T F (θ - θ_old)

Where F = E[∇_θ log π · (∇_θ log π)^T] is the Fisher Information Matrix.

```

### Step 3: Constrained Optimization

The problem becomes:

```
max_{θ}  g^T(θ - θ_old)
s.t.     ½(θ - θ_old)^T F (θ - θ_old) ≤ δ

Using Lagrangian:
L = g^T d - λ(½ d^T F d - δ)

Taking derivative: g - λFd = 0  →  d = λ^{-1} F^{-1} g

```

### Step 4: Natural Gradient Direction

```
The optimal step direction is:

d* = F^{-1} g  (Natural gradient!)

The step size α is chosen to satisfy the constraint:
½ α² d^T F d = δ  →  α = √(2δ / d^T F d)

Final update:
θ ← θ_old + α · F^{-1} g
θ ← θ_old + √(2δ / g^T F^{-1} g) · F^{-1} g

```

---

## 📐 Fisher Information Matrix

### Definition

```
F = E_{s~ρ, a~π}[∇_θ log π_θ(a|s) · (∇_θ log π_θ(a|s))^T]

This is the expected outer product of the score function.

```

### Properties

```
1. F is positive semi-definite

2. F measures the curvature of the KL divergence

3. F^{-1} transforms gradients to natural gradient space

4. Natural gradients are invariant to parameterization

```

### Fisher-Vector Product (Efficient Computation)

Computing F^{-1}g directly is expensive. Instead, use conjugate gradient:

```
Solve: Fx = g  for x using conjugate gradient

The Fisher-vector product Fv can be computed efficiently:

Fv = ∇_θ [∇_θ L(θ)^T v]  (Hessian-vector product on KL)

```

---

## 📐 Conjugate Gradient Algorithm

```
To solve Fx = g without forming F explicitly:

Initialize: x_0 = 0, r_0 = g, p_0 = r_0

For k = 0, 1, 2, ..., until convergence:
    α_k = r_k^T r_k / (p_k^T F p_k)
    x_{k+1} = x_k + α_k p_k
    r_{k+1} = r_k - α_k F p_k
    β_k = r_{k+1}^T r_{k+1} / (r_k^T r_k)
    p_{k+1} = r_{k+1} + β_k p_k

The key is that F p_k (Fisher-vector product) can be computed efficiently!

```

---

## 📐 Line Search for Constraint Satisfaction

After finding the search direction, perform line search to ensure:

1. The constraint is satisfied

2. The objective actually improves

```
For step_size in [1, 0.5, 0.25, 0.125, ...]:
    θ_new = θ_old + step_size · d
    
    if KL(π_{θ_old} || π_{θ_new}) ≤ δ  and  L(θ_new) > L(θ_old):
        Accept θ_new
        break

```

---

## 💻 Complete Implementation

```python
import torch
import torch.nn as nn
import numpy as np

class TRPOAgent:
    """Trust Region Policy Optimization"""
    
    def __init__(self, policy, value_fn, delta=0.01, damping=0.1,
                 cg_iters=10, backtrack_iters=10, backtrack_coef=0.5):
        self.policy = policy
        self.value_fn = value_fn
        self.delta = delta
        self.damping = damping
        self.cg_iters = cg_iters
        self.backtrack_iters = backtrack_iters
        self.backtrack_coef = backtrack_coef
        
        self.value_optimizer = torch.optim.Adam(value_fn.parameters(), lr=1e-3)
    
    def compute_advantages(self, states, rewards, dones, gamma=0.99, lam=0.95):
        """Compute GAE advantages"""
        values = self.value_fn(states).detach().squeeze()
        T = len(rewards)
        advantages = torch.zeros(T)
        gae = 0
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def surrogate_loss(self, states, actions, advantages, old_log_probs):
        """Compute surrogate objective L(θ)"""
        log_probs = self.policy.log_prob(states, actions)
        ratio = torch.exp(log_probs - old_log_probs)
        return (ratio * advantages).mean()
    
    def kl_divergence(self, states, old_policy_params):
        """Compute mean KL divergence"""
        # Save current params
        current_params = self.get_flat_params()
        
        # Get old distribution
        self.set_flat_params(old_policy_params)
        old_dist = self.policy.get_distribution(states)
        
        # Restore current params
        self.set_flat_params(current_params)
        new_dist = self.policy.get_distribution(states)
        
        kl = torch.distributions.kl_divergence(old_dist, new_dist).mean()
        return kl
    
    def hessian_vector_product(self, states, vector, old_params):
        """Compute F @ vector efficiently"""
        self.set_flat_params(old_params)
        kl = self.kl_divergence(states, old_params)
        
        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_grads = torch.cat([g.view(-1) for g in grads])
        
        grad_vector_product = (flat_grads * vector).sum()
        hvp = torch.autograd.grad(grad_vector_product, self.policy.parameters())
        flat_hvp = torch.cat([g.view(-1) for g in hvp])
        
        return flat_hvp + self.damping * vector
    
    def conjugate_gradient(self, states, b, old_params):
        """Solve Fx = b using conjugate gradient"""
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        r_dot_r = torch.dot(r, r)
        
        for _ in range(self.cg_iters):
            Ap = self.hessian_vector_product(states, p, old_params)
            alpha = r_dot_r / (torch.dot(p, Ap) + 1e-8)
            x = x + alpha * p
            r = r - alpha * Ap
            new_r_dot_r = torch.dot(r, r)
            beta = new_r_dot_r / (r_dot_r + 1e-8)
            p = r + beta * p
            r_dot_r = new_r_dot_r
            
            if r_dot_r < 1e-10:
                break
        
        return x
    
    def get_flat_params(self):
        """Flatten all policy parameters"""
        return torch.cat([p.view(-1) for p in self.policy.parameters()])
    
    def set_flat_params(self, flat_params):
        """Set policy parameters from flat vector"""
        idx = 0
        for p in self.policy.parameters():
            p.data.copy_(flat_params[idx:idx + p.numel()].view(p.shape))
            idx += p.numel()
    
    def update(self, states, actions, rewards, dones):
        """TRPO update step"""
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        # Compute advantages
        advantages, returns = self.compute_advantages(states, rewards, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get old log probs
        with torch.no_grad():
            old_log_probs = self.policy.log_prob(states, actions)
        
        old_params = self.get_flat_params().detach()
        
        # Compute policy gradient
        loss = self.surrogate_loss(states, actions, advantages, old_log_probs)
        grads = torch.autograd.grad(loss, self.policy.parameters())
        flat_grads = torch.cat([g.view(-1) for g in grads])
        
        # Compute natural gradient using conjugate gradient
        step_dir = self.conjugate_gradient(states, flat_grads, old_params)
        
        # Compute step size
        sHs = torch.dot(step_dir, self.hessian_vector_product(states, step_dir, old_params))
        max_step_size = torch.sqrt(2 * self.delta / (sHs + 1e-8))
        
        # Line search
        for i in range(self.backtrack_iters):
            step_size = max_step_size * (self.backtrack_coef ** i)
            new_params = old_params + step_size * step_dir
            self.set_flat_params(new_params)
            
            new_loss = self.surrogate_loss(states, actions, advantages, old_log_probs)
            kl = self.kl_divergence(states, old_params)
            
            if kl <= self.delta and new_loss > loss:
                break
        else:
            # Revert to old params if line search failed
            self.set_flat_params(old_params)
        
        # Update value function
        for _ in range(5):
            values = self.value_fn(states).squeeze()
            value_loss = ((values - returns) ** 2).mean()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        
        return {
            'policy_loss': loss.item(),
            'value_loss': value_loss.item(),
            'kl': kl.item() if 'kl' in dir() else 0
        }

```

---

## 📊 TRPO vs PPO Comparison

| Aspect | TRPO | PPO |
|--------|------|-----|
| **Constraint** | Hard KL constraint | Soft clipping |
| **Optimization** | Conjugate gradient | Standard SGD |
| **Computation** | Expensive (Hessian) | Cheap |
| **Stability** | Very stable | Stable |
| **Implementation** | Complex | Simple |
| **Performance** | Similar | Similar |

### Why PPO Replaced TRPO

```
PPO approximates TRPO's trust region with clipping:

L_PPO(θ) = E[min(ρ(θ)A, clip(ρ(θ), 1-ε, 1+ε)A)]

Benefits:

1. No conjugate gradient needed

2. No Fisher information matrix

3. Works with standard SGD

4. Easier to implement and tune

5. Comparable or better performance

```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | TRPO Paper | [arXiv](https://arxiv.org/abs/1502.05477) |
| 📄 | Natural Policy Gradient | [Paper](https://papers.nips.cc/paper/2002/hash/5c04925674920eb58467fb52ce4ef728-Abstract.html) |
| 📖 | OpenAI Spinning Up | [Docs](https://spinningup.openai.com/en/latest/algorithms/trpo.html) |
| 🇨🇳 | TRPO详解 | [知乎](https://zhuanlan.zhihu.com/p/26308073) |
| 🇨🇳 | 信赖域方法原理 | [CSDN](https://blog.csdn.net/qq_30615903/article/details/81275638) |
| 🇨🇳 | TRPO与PPO对比 | [B站](https://www.bilibili.com/video/BV1cP4y1Y7DN) |

## 🔗 Where This Topic Is Used

| Application | TRPO |
|-------------|-----|
| **Robotics** | Safe policy updates |
| **PPO Predecessor** | Led to simpler PPO |
| **Continuous Control** | MuJoCo benchmarks |
| **Safe RL** | Trust region constraints |

---

⬅️ [Back: PPO](../03_ppo/) | ➡️ [Next: Exploration](../../04_exploration/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
