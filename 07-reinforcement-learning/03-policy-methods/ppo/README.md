<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Ppo&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 🔥 Proximal Policy Optimization (PPO)

> **The most widely used RL algorithm - powering ChatGPT and RLHF**

---

## 🎯 Visual Overview

<img src="../images/ppo-clipping.svg" width="100%">

*Caption: PPO's clipped objective prevents large policy updates that could destabilize training. This is why PPO is the algorithm of choice for training ChatGPT, Claude, and most RLHF systems.*

---

## 📐 Mathematical Formulation

### Policy Gradient Foundation

```
Policy Gradient Theorem:
∇J(θ) = E_π[∇log πθ(a|s) · Qπ(s,a)]
      = E_π[∇log πθ(a|s) · Aπ(s,a)]  (with baseline)

Where:
• J(θ) = E[Σₜ γᵗ rₜ] (expected return)
• Aπ(s,a) = Qπ(s,a) - Vπ(s) (advantage)
```

### TRPO Objective (Precursor)

```
maximize L(θ) = E[πθ(a|s)/πθ_old(a|s) · A(s,a)]
  subject to: KL(πθ_old || πθ) ≤ δ

Problem: Constraint is expensive to compute
Solution: PPO approximates the constraint via clipping
```

### PPO Clipped Objective

```
L^CLIP(θ) = E[min(rₜ(θ)Aₜ, clip(rₜ(θ), 1-ε, 1+ε)Aₜ)]

Where:
• rₜ(θ) = πθ(aₜ|sₜ) / πθ_old(aₜ|sₜ)  (probability ratio)
• ε ∈ [0.1, 0.2]  (clipping parameter)
• Aₜ = advantage estimate (typically GAE)

How clipping works:
If Aₜ > 0 (good action): clip ratio at 1+ε (limit improvement)
If Aₜ < 0 (bad action):  clip ratio at 1-ε (limit damage)

This keeps πθ close to πθ_old without explicit KL constraint!
```

### Generalized Advantage Estimation (GAE)

```
Âₜ^GAE(λ) = Σₗ₌₀^∞ (γλ)ˡ δₜ₊ₗ

Where:
• δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)  (TD error)
• λ ∈ [0.9, 0.99]  (bias-variance tradeoff)

λ = 0: Low variance, high bias (1-step TD)
λ = 1: High variance, low bias (Monte Carlo)
λ = 0.95: Common practical choice
```

### Full PPO Objective

```
L(θ) = E[L^CLIP(θ) - c₁ L^VF(θ) + c₂ S[πθ](s)]

Where:
• L^VF = (Vθ(s) - Vₜarget)²  (value function loss)
• S[π] = -Σₐ π(a|s) log π(a|s)  (entropy bonus)
• c₁ ≈ 0.5, c₂ ≈ 0.01  (coefficients)

Entropy bonus encourages exploration!
```

---

## 📐 PPO for RLHF

```
RLHF Objective:
maximize E[R(x, y)] - β · KL(πθ || πref)
         ---------   -----------------
         reward      stay close to SFT policy

Where:
• R(x, y) = learned reward model
• πref = SFT policy (frozen reference)
• β ∈ [0.01, 0.1] (KL penalty coefficient)

PPO update with KL penalty:
L(θ) = E[min(rₜAₜ, clip(rₜ)Aₜ) - β log(πθ/πref)]
```

---

## 🌍 Applications

| Application | Model | Details |
|-------------|-------|---------|
| **ChatGPT** | GPT-3.5/4 | RLHF with PPO |
| **Claude** | Anthropic | Constitutional AI + PPO |
| **LLaMA-2 Chat** | Meta | PPO fine-tuning |
| **OpenAI Five** | Dota 2 | Distributed PPO |
| **Robotics** | Various | Sim-to-real with PPO |

---

## 💻 Code Examples

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PPO:
    def __init__(self, policy, value_fn, lr=3e-4, clip_eps=0.2, 
                 gamma=0.99, gae_lambda=0.95, epochs=10):
        self.policy = policy
        self.value_fn = value_fn
        self.optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value_fn.parameters()), lr=lr
        )
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epochs = epochs
    
    def compute_gae(self, rewards, values, dones):
        """Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, states, actions, old_log_probs, rewards, dones):
        """PPO update step"""
        # Compute values and advantages
        values = self.value_fn(states).squeeze()
        advantages, returns = self.compute_gae(rewards, values.detach(), dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.epochs):
            # Current policy log probs
            log_probs = self.policy.log_prob(states, actions)
            
            # Probability ratio
            ratio = torch.exp(log_probs - old_log_probs.detach())
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value function loss
            values = self.value_fn(states).squeeze()
            value_loss = F.mse_loss(values, returns)
            
            # Entropy bonus
            entropy = self.policy.entropy(states).mean()
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

# RLHF-specific PPO
def rlhf_ppo_loss(policy, ref_policy, states, actions, rewards, beta=0.1):
    """PPO with KL penalty for RLHF"""
    log_probs = policy.log_prob(states, actions)
    ref_log_probs = ref_policy.log_prob(states, actions).detach()
    
    # KL penalty
    kl = log_probs - ref_log_probs
    
    # Reward with KL penalty
    modified_rewards = rewards - beta * kl
    
    # Standard PPO loss with modified rewards
    # ... (compute advantages with modified_rewards)
    
    return policy_loss
```

---

## 📊 Hyperparameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| clip_eps (ε) | 0.1-0.2 | Trust region size |
| gamma (γ) | 0.99 | Discount factor |
| gae_lambda (λ) | 0.95 | Bias-variance tradeoff |
| epochs | 3-10 | Updates per batch |
| batch_size | 64-256 | Samples per update |
| lr | 3e-4 | Learning rate |
| entropy_coef | 0.01 | Exploration bonus |
| value_coef | 0.5 | Value loss weight |
| kl_coef (β) | 0.01-0.1 | RLHF KL penalty |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Proximal Policy Optimization | [arXiv](https://arxiv.org/abs/1707.06347) |
| 📄 | GAE | [arXiv](https://arxiv.org/abs/1506.02438) |
| 📄 | InstructGPT (RLHF) | [arXiv](https://arxiv.org/abs/2203.02155) |
| 📖 | OpenAI Spinning Up | [Docs](https://spinningup.openai.com/en/latest/algorithms/ppo.html) |
| 📖 | Hugging Face RLHF | [Blog](https://huggingface.co/blog/rlhf) |
| 💻 | CleanRL PPO | [GitHub](https://github.com/vwxyzjn/cleanrl) |
| 🇨🇳 | PPO算法详解 | [知乎](https://zhuanlan.zhihu.com/p/512327050) |
| 🇨🇳 | 强化学习PPO教程 | [B站](https://www.bilibili.com/video/BV1cP4y1Y7DN) |
| 🇨🇳 | RLHF实战 | [CSDN](https://blog.csdn.net/qq_37006625/article/details/129405866) |
| 🇨🇳 | PPO原理与实现 | [机器之心](https://www.jiqizhixin.com/articles/2018-06-19-8)

---

## 🔗 Where PPO Is Used

| Application | How PPO Is Applied |
|-------------|-------------------|
| **RLHF/ChatGPT** | Fine-tune LLMs to follow human preferences |
| **InstructGPT** | Original RLHF paper using PPO |
| **Claude/Anthropic** | Constitutional AI uses PPO variants |
| **Robotics** | Sim-to-real transfer for motor control |
| **Game Playing** | OpenAI Five (Dota 2), AlphaStar |
| **Autonomous Driving** | Waymo uses RL for planning |

---


⬅️ [Back: Policy Methods](../)

---

⬅️ [Back: Policy Gradient](../policy-gradient/) | ➡️ [Next: Trpo](../trpo/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
