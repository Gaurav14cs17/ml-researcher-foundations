<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=RLHF%20-%20Reinforcement%20Learning%20from%20Human%20Feedback&fontSize=28&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🔗 Navigation

⬅️ [Back: Games](../01_games/) | ➡️ [Next: Robotics](../03_robotics/)

---

## 🎯 Visual Overview

<img src="./images/rlhf-pipeline.svg" width="100%">

*Caption: The RLHF pipeline showing three stages: (1) Supervised Fine-Tuning (SFT) on demonstrations, (2) Reward Model training from human preferences, and (3) RL optimization using PPO to maximize reward while staying close to the SFT policy. This is how ChatGPT and Claude are aligned.*

---

## 📂 Overview

Reinforcement Learning from Human Feedback (RLHF) is the technique that transforms base language models into helpful, harmless, and honest AI assistants. It captures human preferences that are difficult to specify programmatically.

---

## 📐 Mathematical Framework

### The RLHF Optimization Problem

The goal is to find a policy π that maximizes human preference while staying close to a reference policy:

```
max_π  E_{x~D, y~π(·|x)} [r(x, y)]  -  β · KL(π || π_ref)
       ────────────────────────────    ─────────────────────
         Maximize learned reward        Stay close to SFT policy

Where:
• x: Input prompt
• y: Generated response
• r(x, y): Learned reward model
• π_ref: Reference policy (typically SFT model)
• β: KL penalty coefficient (typically 0.01-0.1)
```

---

## 📐 Stage 1: Supervised Fine-Tuning (SFT)

### Objective

Train on high-quality demonstrations to create a base policy:

```
L_SFT(θ) = -E_{(x,y)~D_demo} [log π_θ(y|x)]
         = -E [Σ_t log π_θ(y_t | x, y_{<t})]

This is standard language modeling on curated data.
```

### Purpose

```
Pretrained LLM (knows language, not task)
         ↓ SFT
SFT Model (understands instruction format, basic helpfulness)
         ↓ RLHF
Aligned Model (optimized for human preferences)
```

---

## 📐 Stage 2: Reward Model Training

### The Bradley-Terry Model

Human preferences are modeled as pairwise comparisons:

```
P(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))
                 = exp(r(x, y_w)) / (exp(r(x, y_w)) + exp(r(x, y_l)))

Where:
• y_w: Preferred (winning) response
• y_l: Dispreferred (losing) response
• σ: Sigmoid function
• r(x, y): Scalar reward for response y given prompt x
```

### Maximum Likelihood Estimation

Given preference dataset D = {(x^i, y_w^i, y_l^i)}:

```
L_RM(φ) = -E_{(x,y_w,y_l)~D} [log σ(r_φ(x, y_w) - r_φ(x, y_l))]

Gradient:
∇_φ L_RM = -E[(1 - P(y_w ≻ y_l)) · (∇_φ r(x, y_w) - ∇_φ r(x, y_l))]
```

### Proof of Bradley-Terry MLE

```
The likelihood of observing preference y_w ≻ y_l:
L = P(y_w ≻ y_l) = σ(r(y_w) - r(y_l))

Log-likelihood:
log L = log σ(r(y_w) - r(y_l))
      = r(y_w) - r(y_l) - log(1 + exp(r(y_w) - r(y_l)))

This is the negative cross-entropy loss for binary classification.
```

### Reward Model Architecture

```
Reward Model = LLM backbone + Linear head

r(x, y) = W · h_final(x, y) + b

Where:
• h_final: Last token's hidden state from transformer
• W ∈ R^{1×d}: Learned projection
• b ∈ R: Bias term
```

---

## 📐 Stage 3: RL Fine-Tuning with PPO

### RLHF Objective with KL Penalty

```
J(θ) = E_{x~D, y~π_θ} [r(x, y) - β · log(π_θ(y|x) / π_ref(y|x))]

Equivalent form:
J(θ) = E[r(x, y)] - β · KL(π_θ(·|x) || π_ref(·|x))

The KL term prevents:
• Reward hacking
• Catastrophic forgetting
• Collapse to degenerate solutions
```

### Per-Token KL Penalty

In practice, KL is computed per-token:

```
KL(π_θ || π_ref) = Σ_t E[log π_θ(y_t|x,y_{<t}) - log π_ref(y_t|x,y_{<t})]

Modified reward at each step:
r'_t = r_RM(x, y) · 𝟙[t = T] - β · (log π_θ(y_t) - log π_ref(y_t))
                  └─ reward at end ─┘   └───── KL penalty at each step ─────┘
```

### PPO Clipped Objective for RLHF

```
L^CLIP(θ) = E_t [min(ρ_t · A_t, clip(ρ_t, 1-ε, 1+ε) · A_t)]

Where:
• ρ_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (probability ratio)
• A_t = GAE advantage estimate
• ε = 0.2 (typical clipping parameter)
```

### Advantage Estimation for LLMs

```
A_t^GAE = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}

Where:
δ_t = r'_t + γ · V(s_{t+1}) - V(s_t)
    = [r_RM · 𝟙[t=T] - β·KL_t] + γ · V(s_{t+1}) - V(s_t)
```

---

## 📐 Direct Preference Optimization (DPO)

### Key Insight: Closed-Form Optimal Policy

Under the RLHF objective, the optimal policy has a closed form:

```
Theorem: The optimal policy for the KL-regularized reward maximization is:

π*(y|x) = π_ref(y|x) · exp(r(x,y)/β) / Z(x)

Where Z(x) = Σ_y π_ref(y|x) · exp(r(x,y)/β) is the partition function.

Proof:
The objective is: max_π E_y~π[r(x,y)] - β·KL(π||π_ref)

Taking the functional derivative and setting to zero:
∂/∂π(y) [Σ_y π(y)(r(y) - β log(π(y)/π_ref(y))) + λ(Σ_y π(y) - 1)] = 0

r(y) - β log(π(y)/π_ref(y)) - β + λ = 0
π(y) = π_ref(y) · exp((r(y) + λ - β)/β)

Normalizing: π*(y) = π_ref(y) · exp(r(y)/β) / Z
```

### Rearranging for Implicit Reward

```
From π*(y|x) = π_ref(y|x) · exp(r(x,y)/β) / Z(x):

r(x, y) = β · log(π*(y|x) / π_ref(y|x)) + β · log Z(x)

Key insight: The reward is implicitly defined by the policy!
```

### DPO Loss Function

Substituting the implicit reward into Bradley-Terry:

```
P(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))
                 = σ(β log(π*(y_w|x)/π_ref(y_w|x)) - β log(π*(y_l|x)/π_ref(y_l|x)))

The log Z(x) terms cancel!

DPO Loss:
L_DPO(θ) = -E_{(x,y_w,y_l)~D} [log σ(β · (log(π_θ(y_w|x)/π_ref(y_w|x)) 
                                        - log(π_θ(y_l|x)/π_ref(y_l|x))))]
```

### Gradient of DPO

```
∇_θ L_DPO = -β · E[(1 - σ(β·Δ)) · (∇_θ log π_θ(y_w|x) - ∇_θ log π_θ(y_l|x))]

Where Δ = log(π_θ(y_w)/π_ref(y_w)) - log(π_θ(y_l)/π_ref(y_l))

Intuition:
• Increase probability of preferred response y_w
• Decrease probability of dispreferred response y_l
• Weighted by how wrong the current model is (1 - σ(β·Δ))
```

---

## 💻 Complete Implementation

### Reward Model Training

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class RewardModel(nn.Module):
    """Reward model built on LLM backbone"""
    
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(model_name)
        self.reward_head = nn.Linear(self.backbone.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        # Get last hidden state
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden = outputs.hidden_states[-1]
        
        # Use last token (or EOS) representation
        last_token_idx = attention_mask.sum(dim=1) - 1
        last_hidden = hidden[torch.arange(hidden.size(0)), last_token_idx]
        
        # Scalar reward
        reward = self.reward_head(last_hidden).squeeze(-1)
        return reward

def train_reward_model(model, dataloader, optimizer, epochs=3):
    """Train reward model on preference pairs"""
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            # Get rewards for chosen and rejected
            r_chosen = model(batch['chosen_ids'], batch['chosen_mask'])
            r_rejected = model(batch['rejected_ids'], batch['rejected_mask'])
            
            # Bradley-Terry loss
            loss = -F.logsigmoid(r_chosen - r_rejected).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (r_chosen > r_rejected).sum().item()
            total += len(r_chosen)
        
        accuracy = correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}, Acc={accuracy:.4f}")
```

### PPO for RLHF

```python
class RLHFTrainer:
    """PPO trainer for RLHF"""
    
    def __init__(self, policy, ref_policy, reward_model, 
                 beta=0.1, clip_eps=0.2, gamma=1.0, gae_lambda=0.95):
        self.policy = policy
        self.ref_policy = ref_policy  # Frozen
        self.reward_model = reward_model  # Frozen
        
        self.beta = beta
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-5)
    
    def compute_rewards(self, prompts, responses):
        """Compute reward with KL penalty"""
        # Get reward model score
        with torch.no_grad():
            reward_scores = self.reward_model(prompts + responses)
        
        # Compute per-token KL
        with torch.no_grad():
            ref_logprobs = self.ref_policy.log_prob(responses, prompts)
        policy_logprobs = self.policy.log_prob(responses, prompts)
        
        kl = policy_logprobs - ref_logprobs
        
        # Per-token rewards: KL penalty + final reward
        rewards = -self.beta * kl
        rewards[:, -1] += reward_scores  # Add RM reward at end
        
        return rewards, kl.mean()
    
    def compute_advantages(self, rewards, values):
        """GAE advantage computation"""
        T = rewards.shape[1]
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0
            else:
                next_value = values[:, t + 1]
            
            delta = rewards[:, t] + self.gamma * next_value - values[:, t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[:, t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def ppo_step(self, prompts, responses, old_logprobs, old_values):
        """Single PPO update step"""
        # Get current policy outputs
        logprobs = self.policy.log_prob(responses, prompts)
        values = self.policy.value(prompts, responses)
        
        # Compute rewards and advantages
        rewards, kl = self.compute_rewards(prompts, responses)
        advantages, returns = self.compute_advantages(rewards, old_values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy ratio
        ratio = torch.exp(logprobs - old_logprobs.detach())
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns.detach())
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'kl': kl.item(),
            'reward': rewards.mean().item()
        }
```

### DPO Implementation

```python
class DPOTrainer:
    """Direct Preference Optimization"""
    
    def __init__(self, model, ref_model, beta=0.1, lr=1e-6):
        self.model = model
        self.ref_model = ref_model  # Frozen
        self.beta = beta
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    def compute_log_probs(self, model, input_ids, attention_mask, labels):
        """Compute log probabilities for sequences"""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # Shift for next token prediction
        labels = labels[:, 1:]  # Shift labels
        
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        
        # Mask padding
        mask = (labels != -100).float()
        return (selected_log_probs * mask).sum(-1) / mask.sum(-1)
    
    def dpo_loss(self, chosen_ids, chosen_mask, rejected_ids, rejected_mask):
        """Compute DPO loss"""
        # Policy log probs
        policy_chosen_logp = self.compute_log_probs(
            self.model, chosen_ids, chosen_mask, chosen_ids
        )
        policy_rejected_logp = self.compute_log_probs(
            self.model, rejected_ids, rejected_mask, rejected_ids
        )
        
        # Reference log probs (frozen)
        with torch.no_grad():
            ref_chosen_logp = self.compute_log_probs(
                self.ref_model, chosen_ids, chosen_mask, chosen_ids
            )
            ref_rejected_logp = self.compute_log_probs(
                self.ref_model, rejected_ids, rejected_mask, rejected_ids
            )
        
        # Log ratios
        chosen_ratio = policy_chosen_logp - ref_chosen_logp
        rejected_ratio = policy_rejected_logp - ref_rejected_logp
        
        # DPO loss
        logits = self.beta * (chosen_ratio - rejected_ratio)
        loss = -F.logsigmoid(logits).mean()
        
        return loss, {
            'chosen_reward': chosen_ratio.mean().item(),
            'rejected_reward': rejected_ratio.mean().item(),
            'accuracy': (logits > 0).float().mean().item()
        }
    
    def train_step(self, batch):
        """Single DPO training step"""
        loss, metrics = self.dpo_loss(
            batch['chosen_ids'], batch['chosen_mask'],
            batch['rejected_ids'], batch['rejected_mask']
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        metrics['loss'] = loss.item()
        return metrics
```

---

## 📊 RLHF vs DPO Comparison

| Aspect | RLHF (PPO) | DPO |
|--------|------------|-----|
| **Reward Model** | Required (separate training) | Not needed |
| **RL Loop** | Yes (PPO optimization) | No (supervised learning) |
| **Sampling** | Online (during training) | Offline (precomputed) |
| **Stability** | Can be unstable | Very stable |
| **Memory** | High (value network, ref model) | Lower |
| **Compute** | High (sampling + optimization) | Lower |
| **Flexibility** | More flexible | Less flexible |

---

## 📊 Modern Variants

| Method | Key Innovation | Advantage |
|--------|----------------|-----------|
| **DPO** | No reward model needed | Simpler, more stable |
| **IPO** | Identity mapping | Better calibration |
| **KTO** | Only positive examples | Works with binary feedback |
| **ORPO** | Odds ratio | No reference model needed |
| **SimPO** | Length-normalized | Better for varying lengths |
| **RLAIF** | AI feedback | Scales without humans |

---

## 🌍 Where RLHF/DPO is Used

| Model | Year | Method | Notes |
|-------|------|--------|-------|
| **ChatGPT** | 2022 | RLHF (PPO) | Original RLHF for chat |
| **GPT-4** | 2023 | Advanced RLHF | Improved pipeline |
| **Claude** | 2023 | Constitutional AI | RLHF + AI feedback |
| **LLaMA-2** | 2023 | RLHF + DPO | Open-source RLHF |
| **Gemini** | 2023 | RLHF variants | Google's approach |
| **Mistral** | 2024 | DPO | Efficient alignment |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | InstructGPT (RLHF) | [arXiv](https://arxiv.org/abs/2203.02155) |
| 📄 | DPO Paper | [arXiv](https://arxiv.org/abs/2305.18290) |
| 📄 | Constitutional AI | [arXiv](https://arxiv.org/abs/2212.08073) |
| 📄 | PPO Paper | [arXiv](https://arxiv.org/abs/1707.06347) |
| 🎥 | RLHF Explained | [HuggingFace Blog](https://huggingface.co/blog/rlhf) |
| 🇨🇳 | RLHF详解 | [知乎](https://zhuanlan.zhihu.com/p/595579042) |
| 🇨🇳 | DPO算法详解 | [CSDN](https://blog.csdn.net/qq_37006625/article/details/134567890) |
| 🇨🇳 | 大模型对齐技术 | [B站](https://www.bilibili.com/video/BV1cP4y1Y7DN) |

---

## 🔗 Training Pipeline Position

```
Pretraining → SFT → RLHF/DPO
(language)   (task) (preference)
                        ↑
                   You are here!
```

---

⬅️ [Back: Games](../01_games/) | ➡️ [Next: Robotics](../03_robotics/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
