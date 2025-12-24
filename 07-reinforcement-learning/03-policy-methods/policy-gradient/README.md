# Policy Gradient Methods

> **Directly optimize the policy for maximum reward**

---

## 🎯 Visual Overview

<img src="./images/policy-gradient.svg" width="100%">

*Caption: Policy gradient methods parameterize the policy π_θ(a|s) with a neural network and optimize it directly using gradient ascent on expected return. The REINFORCE algorithm increases probabilities of actions that led to high returns.*

---

## 📂 Overview

Policy gradient methods directly optimize the policy parameters θ to maximize expected cumulative reward, without explicitly learning value functions.

---

## 📐 Mathematical Foundation

### Objective Function

```
J(θ) = E_τ~π_θ [R(τ)] = E_τ~π_θ [Σₜ γᵗ rₜ]

Where:
    τ = (s₀, a₀, r₀, s₁, a₁, r₁, ...) is a trajectory
    π_θ(a|s) is the parameterized policy
    R(τ) is the total return of trajectory τ
```

### Policy Gradient Theorem

```
∇_θ J(θ) = E_τ~π_θ [Σₜ ∇_θ log π_θ(aₜ|sₜ) · Gₜ]

Where:
    Gₜ = Σₖ₌ₜ^T γᵏ⁻ᵗ rₖ  (return from time t)

Key insight: We can estimate gradients without knowing P(s'|s,a)!

Proof sketch:
    ∇_θ J = ∇_θ E_τ[R(τ)]
          = ∇_θ ∫ P(τ|θ) R(τ) dτ
          = ∫ P(τ|θ) ∇_θ log P(τ|θ) R(τ) dτ  (log-derivative trick)
          = E_τ [∇_θ log P(τ|θ) · R(τ)]
          
    And: log P(τ|θ) = Σₜ log π_θ(aₜ|sₜ) + terms not depending on θ
```

### REINFORCE Algorithm

```
1. Sample trajectory τ = (s₀, a₀, r₀, ..., s_T) using π_θ
2. Compute returns Gₜ = Σₖ₌ₜ^T γᵏ⁻ᵗ rₖ
3. Update: θ ← θ + α Σₜ ∇_θ log π_θ(aₜ|sₜ) · Gₜ
4. Repeat
```

---

## 🔑 Key Concepts

| Concept | Description |
|---------|-------------|
| **REINFORCE** | Vanilla policy gradient using Monte Carlo returns |
| **Baseline** | Subtract b(s) to reduce variance: ∇J ∝ (G - b) |
| **Actor-Critic** | Use learned value function V(s) as baseline |
| **Advantage** | A(s,a) = Q(s,a) - V(s), even lower variance |
| **GAE** | Generalized Advantage Estimation (λ-weighted) |

### Variance Reduction

```
High Variance Problem:
    ∇_θ J = E[∇_θ log π(a|s) · G]
    
    G varies a lot → noisy gradients → slow learning

Solutions:
1. Baseline: Use V(s) as baseline (doesn't change expectation)
    ∇_θ J = E[∇_θ log π(a|s) · (G - V(s))]
    
2. Advantage: A(s,a) = Q(s,a) - V(s)
    ∇_θ J = E[∇_θ log π(a|s) · A(s,a)]
    
3. GAE: λ-weighted combination of n-step advantages
    Â_t^GAE = Σₗ₌₀^∞ (γλ)ˡ δₜ₊ₗ
    where δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)
```

---

## 💻 Code Examples

### REINFORCE

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)
    
    def get_action(self, state):
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


def reinforce(env, policy, optimizer, num_episodes, gamma=0.99):
    """REINFORCE algorithm"""
    
    for episode in range(num_episodes):
        log_probs = []
        rewards = []
        
        state = env.reset()
        done = False
        
        # Collect trajectory
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob = policy.get_action(state_tensor)
            next_state, reward, done, _ = env.step(action)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
        
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        
        # Normalize returns (optional, helps training)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient update
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G  # Negative for gradient ascent
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return policy
```

### REINFORCE with Baseline

```python
class PolicyWithBaseline(nn.Module):
    """Actor-Critic style with shared layers"""
    
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)
    
    def forward(self, x):
        features = self.shared(x)
        probs = F.softmax(self.policy_head(features), dim=-1)
        value = self.value_head(features)
        return probs, value


def reinforce_with_baseline(env, model, optimizer, num_episodes, gamma=0.99):
    """REINFORCE with learned baseline (value function)"""
    
    for episode in range(num_episodes):
        log_probs = []
        values = []
        rewards = []
        
        state = env.reset()
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs, value = model(state_tensor)
            
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            log_probs.append(dist.log_prob(action))
            values.append(value.squeeze())
            
            state, reward, done, _ = env.step(action.item())
            rewards.append(reward)
        
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        values = torch.stack(values)
        
        # Advantages
        advantages = returns - values.detach()
        
        # Policy loss
        policy_loss = -(torch.stack(log_probs) * advantages).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Generalized Advantage Estimation (GAE)

```python
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation
    
    GAE(γ, λ) = Σₗ₌₀^∞ (γλ)ˡ δₜ₊ₗ
    where δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)
    """
    advantages = []
    gae = 0
    
    # values should have one extra element (V(s_T+1) = 0 if terminal)
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0  # Terminal state
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    return torch.FloatTensor(advantages)
```

---

## 📊 Comparison of Policy Gradient Methods

| Method | Baseline | On/Off-Policy | Variance | Sample Efficiency |
|--------|----------|---------------|----------|-------------------|
| **REINFORCE** | None | On | High | Low |
| **REINFORCE + Baseline** | V(s) | On | Medium | Low |
| **A2C** | V(s) | On | Medium | Low |
| **PPO** | V(s), clipped | On | Low | Medium |
| **TRPO** | V(s), KL constraint | On | Low | Medium |

---

## 🔗 Connection to Other Methods

```
Policy Gradient
    |
    +-- REINFORCE (Monte Carlo)
    |       +-- + Baseline → Actor-Critic (A2C)
    |                           +-- + Trust Region → TRPO
    |                                                   +-- + Clipping → PPO
    |
    +-- Connection to Value Methods
            Q(s,a) ≈ advantage estimation
            V(s) as baseline reduces variance
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Actor-Critic | [../actor-critic/](../actor-critic/) |
| 📖 | PPO | [../ppo/](../ppo/) |
| 📖 | TRPO | [../trpo/](../trpo/) |
| 📄 | Policy Gradient Theorem | [Paper](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) |
| 📄 | GAE Paper | [arXiv](https://arxiv.org/abs/1506.02438) |
| 🇨🇳 | 策略梯度算法详解 | [知乎](https://zhuanlan.zhihu.com/p/26174099) |
| 🇨🇳 | Policy Gradient推导 | [CSDN](https://blog.csdn.net/qq_30615903/article/details/81275638) |
| 🇨🇳 | 强化学习-策略梯度 | [B站](https://www.bilibili.com/video/BV1sd4y167NS) |
| 🇨🇳 | 策略梯度方法综述 | [机器之心](https://www.jiqizhixin.com/articles/2018-02-13-4) |
| 🇨🇳 | GAE详解 | [PaperWeekly](https://www.paperweekly.site/papers/notes/1468)


## 🔗 Where This Topic Is Used

| Application | Policy Gradient |
|-------------|----------------|
| **LLM Training** | REINFORCE for RLHF |
| **Robotics** | Direct policy learning |
| **NAS** | Architecture search |
| **Continuous Control** | Gaussian policies |

---

⬅️ [Back: Policy Methods](../)

---

⬅️ [Back: Actor Critic](../actor-critic/) | ➡️ [Next: Ppo](../ppo/)
