# Dreamer

> **Learning to dream for sample-efficient RL**

---

## 🎯 Visual Overview

<img src="./images/dreamer.svg" width="100%">

*Caption: Dreamer learns a world model from real experience, then trains an actor-critic policy purely from imagined trajectories. The RSSM provides a compact latent state representation.*

---

## 📂 Overview

Dreamer is a state-of-the-art model-based RL algorithm that achieves near model-free performance with ~100x less data by learning from imagined experience.

---

## 📐 Mathematical Framework

### World Model Components

```
World Model = (Encoder, RSSM, Reward, Decoder)

Encoder:   q(z_t | h_t, x_t)      Image → latent
RSSM:      p(z_t | h_t)           Dynamics prediction
           h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
Reward:    p(r_t | h_t, z_t)      Latent → reward
Decoder:   p(x_t | h_t, z_t)      Latent → image
```

### Training Objective (ELBO)

```
max E[Σ_t (log p(x_t | z_t, h_t) + log p(r_t | z_t, h_t) 
        - β KL(q(z_t | h_t, x_t) || p(z_t | h_t)))]
        
Terms:
    log p(x_t | ...) : Reconstruction (image prediction)
    log p(r_t | ...) : Reward prediction
    KL(...) : Regularization (match prior)
```

---

## 📐 Key Components

| Component | Function |
|-----------|----------|
| **RSSM** | Recurrent State-Space Model for dynamics |
| **Encoder** | Image → latent state |
| **Decoder** | Latent → reconstructed image |
| **Actor** | π(a\|z) policy in latent space |
| **Critic** | V(z) value function |

---

## 🔑 RSSM (Recurrent State-Space Model)

```
State = (h, z) where:
- h ∈ ℝ^d: Deterministic recurrent state (GRU hidden)
- z ∈ ℝ^k: Stochastic latent state (sampled)

Transitions:
h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})    # Deterministic
z_t ~ p(z_t | h_t)                       # Stochastic prior
z_t ~ q(z_t | h_t, x_t)                  # Posterior (training)

Why both?
- Deterministic: Captures long-term dependencies
- Stochastic: Models environment uncertainty
```

### RSSM Architecture

```
                    x_t (observation)
                      |
                      v
    +--------------------------------+
    |           Encoder              |
    |    q(z_t | h_t, x_t)          |
    +--------------------------------+
                      |
                      v
    +---------------------------------------------+
    |                RSSM Core                     |
    |  +---------+      +-----------------+       |
    |  |   GRU   |------|  Stochastic z   |       |
    |  |  (h_t)  |      |  p(z_t | h_t)   |       |
    |  +---------+      +-----------------+       |
    |       ^                   |                 |
    |       |      ▼      |
    |  [h_{t-1}, z_{t-1}, a_{t-1}]   State (h_t, z_t)  |
    +---------------------------------------------+
```

---

## 🚀 Training Loop

```
1. Collect experience in real environment (small amount)
2. Train world model on real data
3. Imagine trajectories starting from real states
4. Train actor-critic on imagined trajectories
5. Repeat

Key insight: Backprop through differentiable world model!
```

### Imagination Process

```
Given initial state (h_0, z_0) from real data:

For t = 1 to H (imagination horizon):
    a_t = π(h_t, z_t)           # Actor samples action
    h_{t+1} = GRU(h_t, z_t, a_t) # Deterministic transition
    z_{t+1} ~ p(z | h_{t+1})     # Stochastic transition
    r_t = reward_model(h_t, z_t) # Predicted reward

Gradients flow through the entire imagined trajectory!
```

---

## 📐 Policy Learning (Actor-Critic)

### Value Function (λ-returns)

```
V_λ(z_t) = (1-λ) Σ_{n=1}^{H-t} λ^{n-1} V_n(z_t) + λ^{H-t} V_H(z_t)

Where V_n is n-step return:
V_n(z_t) = E[r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n V(z_{t+n})]
```

### Actor Objective

```
max E[Σ_t (V_λ(z_t) - η H[π(·|z_t)])]

With entropy regularization for exploration
```

---

## 💻 Code Implementation

### World Model

```python
import torch
import torch.nn as nn
import torch.distributions as D

class RSSM(nn.Module):
    """Recurrent State-Space Model"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=200, stoch_dim=30):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stoch_dim = stoch_dim
        
        # Encoder: observation → posterior
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 200),
            nn.ELU(),
            nn.Linear(200, 2 * stoch_dim)  # mean, std
        )
        
        # Recurrent model
        self.gru = nn.GRUCell(stoch_dim + action_dim, hidden_dim)
        
        # Prior: h → z distribution
        self.prior = nn.Sequential(
            nn.Linear(hidden_dim, 200),
            nn.ELU(),
            nn.Linear(200, 2 * stoch_dim)
        )
        
        # Decoder, reward predictor
        self.decoder = nn.Linear(hidden_dim + stoch_dim, obs_dim)
        self.reward_head = nn.Linear(hidden_dim + stoch_dim, 1)
    
    def get_dist(self, stats):
        mean, std = stats.chunk(2, dim=-1)
        std = nn.functional.softplus(std) + 0.1
        return D.Normal(mean, std)
    
    def observe(self, obs, action, h_prev, z_prev):
        """One step with observation (training)"""
        # Deterministic transition
        h = self.gru(torch.cat([z_prev, action], -1), h_prev)
        
        # Posterior (uses observation)
        posterior_stats = self.encoder(obs) + self.prior(h)
        posterior = self.get_dist(posterior_stats)
        z = posterior.rsample()
        
        # Prior (for KL loss)
        prior_stats = self.prior(h)
        prior = self.get_dist(prior_stats)
        
        return h, z, prior, posterior
    
    def imagine(self, action, h_prev, z_prev):
        """One step without observation (imagination)"""
        h = self.gru(torch.cat([z_prev, action], -1), h_prev)
        prior = self.get_dist(self.prior(h))
        z = prior.rsample()
        return h, z
    
    def reward(self, h, z):
        return self.reward_head(torch.cat([h, z], -1))


class DreamerAgent:
    def __init__(self, obs_dim, action_dim):
        self.world_model = RSSM(obs_dim, action_dim)
        self.actor = Actor(obs_dim=230, action_dim=action_dim)
        self.critic = Critic(obs_dim=230)
        
        self.world_opt = torch.optim.Adam(self.world_model.parameters(), lr=6e-4)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=8e-5)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=8e-5)
    
    def imagine_trajectories(self, initial_h, initial_z, horizon=15):
        """Dream about the future"""
        h, z = initial_h, initial_z
        imagined = []
        
        for t in range(horizon):
            state = torch.cat([h, z], -1)
            action = self.actor(state).sample()
            h, z = self.world_model.imagine(action, h, z)
            reward = self.world_model.reward(h, z)
            imagined.append((h, z, action, reward))
        
        return imagined
    
    def update_world_model(self, batch):
        """Train world model on real experience"""
        obs, actions, rewards = batch
        T, B = obs.shape[:2]
        
        h = torch.zeros(B, self.world_model.hidden_dim)
        z = torch.zeros(B, self.world_model.stoch_dim)
        
        total_loss = 0
        for t in range(T-1):
            h, z, prior, posterior = self.world_model.observe(
                obs[t], actions[t], h, z
            )
            
            # Reconstruction loss
            recon = self.world_model.decoder(torch.cat([h, z], -1))
            recon_loss = ((recon - obs[t+1]) ** 2).mean()
            
            # Reward loss
            pred_reward = self.world_model.reward(h, z)
            reward_loss = ((pred_reward - rewards[t]) ** 2).mean()
            
            # KL loss
            kl_loss = D.kl_divergence(posterior, prior).mean()
            
            total_loss += recon_loss + reward_loss + 0.1 * kl_loss
        
        self.world_opt.zero_grad()
        total_loss.backward()
        self.world_opt.step()
        
        return total_loss.item()
```

---

## 📊 Dreamer Versions

| Version | Key Improvement | Paper |
|---------|----------------|-------|
| **Dreamer v1** | Basic RSSM + λ-returns | Hafner 2019 |
| **Dreamer v2** | Discrete latents, actor-critic fixes | Hafner 2020 |
| **Dreamer v3** | Symlog predictions, world model scaling | Hafner 2023 |

---

## 🔗 Connection to Other Methods

```
Model-Based RL
    |
    +-- Dyna (simple model + planning)
    +-- World Models (VAE + RNN)
    +-- MuZero (learned model + MCTS)
    +-- Dreamer (RSSM + imagination)
            |
            +-- DreamerV2 (discrete latents)
            +-- DreamerV3 (scaling + symlog)
```

---

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 📄 | DreamerV1 | [arXiv](https://arxiv.org/abs/1912.01603) |
| 📄 | DreamerV2 | [arXiv](https://arxiv.org/abs/2010.02193) |
| 📄 | DreamerV3 | [arXiv](https://arxiv.org/abs/2301.04104) |
| 💻 | Official Code | [GitHub](https://github.com/danijar/dreamer) |
| 🇨🇳 | Dreamer系列详解 | [知乎](https://zhuanlan.zhihu.com/p/563656219) |
| 🇨🇳 | 世界模型与Dreamer | [CSDN](https://blog.csdn.net/qq_37006625/article/details/123629543) |
| 🇨🇳 | 模型基RL-Dreamer | [B站](https://www.bilibili.com/video/BV1C34y1H7Eq) |
| 🇨🇳 | DreamerV3解读 | [机器之心](https://www.jiqizhixin.com/articles/2023-01-13-5) |
| 🇨🇳 | 基于想象的强化学习 | [PaperWeekly](https://www.paperweekly.site/papers/notes/2568)


## 🔗 Where This Topic Is Used

| Application | Dreamer |
|-------------|--------|
| **Sample Efficiency** | Learn from imagination |
| **Visual RL** | Latent dynamics |
| **Continuous Control** | DMC benchmarks |
| **World Models** | State-of-the-art |

---

⬅️ [Back: Model-Based](../)

---

➡️ [Next: Mcts](../mcts/)
