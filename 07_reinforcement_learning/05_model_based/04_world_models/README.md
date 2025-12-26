<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=World%20Models&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🔗 Navigation

⬅️ [Back: Planning](../03_planning/) | ➡️ [Next: Applications](../../06_applications/)

---

## 🎯 Visual Overview

<img src="../images/world-models.svg" width="100%">

*Caption: World models learn to predict future states and rewards given actions. The agent can then "dream" - simulate trajectories in imagination to plan without real environment interaction.*

---

## 📂 Overview

World models are learned representations of environment dynamics. They enable sample-efficient RL by allowing agents to learn from imagined experience.

---

## 🔑 Key Components

| Component | Function |
|-----------|----------|
| **Encoder** | State → latent representation z |
| **Dynamics Model** | Predict next latent: z' = f(z, a) |
| **Reward Model** | Predict reward: r̂ = g(z, a) |
| **Decoder** | Latent → reconstructed state (optional) |

---

## 📐 Mathematical Framework

### World Model Definition

```
A world model consists of:

1. Dynamics Model: p_θ(s_{t+1} | s_t, a_t)
   or deterministic: s_{t+1} = f_θ(s_t, a_t)

2. Reward Model: r_t = R_φ(s_t, a_t)

3. (Optional) Encoder: z_t = E_ψ(s_t)
   Maps observations to latent states

4. (Optional) Decoder: ŝ_t = D_ω(z_t)
   Reconstructs observations
```

### Latent Space Dynamics

```
For high-dimensional observations (images):

Encode: z_t = E_ψ(s_t)
Dynamics: z_{t+1} = f_θ(z_t, a_t)  (in latent space!)
Reward: r_t = R_φ(z_t, a_t)
Decode: ŝ_t = D_ω(z_t)  (for visualization)

Benefits:
  - Lower-dimensional (faster planning)
  - Captures relevant features
  - Ignores irrelevant details
```

---

## 📐 Variational World Model (VAE-based)

### ELBO Objective

```
Maximize Evidence Lower Bound:

L(θ,ψ,φ) = E_{z~q_ψ(z|s)} [log p_ω(s|z)]     (reconstruction)
         - β D_KL(q_ψ(z|s) || p(z))           (regularization)
         + E_{z~q_ψ} [log p_φ(r|z,a)]         (reward prediction)
         + E_{z~q_ψ} [log p_θ(z'|z,a)]        (dynamics)

Where:
  q_ψ(z|s) = encoder (posterior)
  p(z) = prior (typically N(0,I))
  p_θ(z'|z,a) = dynamics model
```

### Reparameterization Trick

```
For Gaussian latents:
  q_ψ(z|s) = N(μ_ψ(s), σ_ψ(s)²)

Sample via:
  z = μ_ψ(s) + σ_ψ(s) ⊙ ε,  where ε ~ N(0, I)

This allows gradient flow: ∂L/∂ψ well-defined
```

---

## 📐 Recurrent World Models

### RNN/GRU Dynamics

```
Hidden state captures history:
  h_t = RNN(h_{t-1}, z_{t-1}, a_{t-1})

Stochastic latent:
  z_t ~ p_θ(z|h_t)

Combined state: (h_t, z_t)
  - h_t: Deterministic, long-term memory
  - z_t: Stochastic, captures uncertainty
```

### RSSM Loss (Dreamer-style)

```
L = E_t [ -log p(x_t|z_t,h_t)         (reconstruction)
        - log p(r_t|z_t,h_t)          (reward)
        + β D_KL(q(z_t|h_t,x_t) || p(z_t|h_t))  (KL)
       ]

Where:
  p(z_t|h_t) = prior (prediction before observation)
  q(z_t|h_t,x_t) = posterior (after seeing observation)
```

---

## 📐 Model Uncertainty

### Epistemic vs Aleatoric Uncertainty

```
Total uncertainty = Epistemic + Aleatoric

Epistemic (model uncertainty):
  - Due to limited data
  - Reducible with more data
  - Model with ensemble: Var[f_1(s,a), ..., f_M(s,a)]

Aleatoric (environment stochasticity):
  - Inherent randomness
  - Irreducible
  - Model with distribution: p(s'|s,a)
```

### Ensemble World Models

```
Train M models: {f_θ₁, ..., f_θ_M}

Prediction:
  Mean: μ(s,a) = (1/M) Σᵢ f_θᵢ(s,a)
  Variance: σ²(s,a) = (1/M) Σᵢ (f_θᵢ(s,a) - μ)²

Use uncertainty for:
  - Exploration bonus: r_i ∝ σ(s,a)
  - Conservative planning: penalize high variance
```

---

## 📐 Model-Based Policy Optimization

### Dyna-style Learning

```
Algorithm:
  1. Act in real environment, collect (s,a,r,s')
  2. Add to replay buffer D
  3. Train world model on D
  4. Generate synthetic experience with model
  5. Train policy on real + synthetic data
  6. Repeat

Ratio of real:synthetic typically 1:10 or higher
```

### Analytic Policy Gradient (Dreamer)

```
For differentiable world model:

∇J(π) = E_τ~model [Σ_t ∇_π log π(a_t|s_t) · ∂V/∂a_t]

Can backprop through imagined trajectory:
  s_{t+1} = f_θ(s_t, a_t)  ← differentiable
  ∂s_{t+1}/∂a_t = ∂f_θ/∂a_t

This is more efficient than REINFORCE!
```

---

## 🌍 Applications

| Model | Key Innovation |
|-------|----------------|
| **World Models (Ha 2018)** | VAE + RNN for racing |
| **SimPLe** | Model-based Atari |
| **Dreamer** | RSSM, backprop through model |
| **MuZero** | Value prediction, no reconstruction |

---

## 💻 Code

```python
class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # mean + logvar
        )
        self.dynamics = nn.GRU(latent_dim + action_dim, 256)
        self.reward_head = nn.Linear(256, 1)
        
    def imagine(self, z, actions, horizon):
        """Rollout in imagination"""
        imagined = []
        h = self.dynamics.init_hidden(z.size(0))
        for t in range(horizon):
            z_a = torch.cat([z, actions[:, t]], dim=-1)
            h = self.dynamics(z_a, h)
            z = self.transition(h)
            r = self.reward_head(h)
            imagined.append((z, r))
        return imagined
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | World Models Paper | [arXiv](https://arxiv.org/abs/1803.10122) |
| 📄 | Dreamer Paper | [arXiv](https://arxiv.org/abs/1912.01603) |
| 📄 | MuZero Paper | [Nature](https://www.nature.com/articles/s41586-020-03051-4) |
| 🇨🇳 | 世界模型详解 | [知乎](https://zhuanlan.zhihu.com/p/563656219) |
| 🇨🇳 | Dreamer系列 | [CSDN](https://blog.csdn.net/qq_37006625/article/details/123629543) |
| 🇨🇳 | 模型基RL | [B站](https://www.bilibili.com/video/BV1C34y1H7Eq) |

## 🔗 Where This Topic Is Used

| Application | World Models |
|-------------|-------------|
| **Dreamer** | Learning in imagination |
| **MuZero** | Learned dynamics model |
| **Planning** | Model predictive control |
| **Sim-to-Real** | Domain randomization |

---

⬅️ [Back: Planning](../03_planning/) | ➡️ [Next: Applications](../../06_applications/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
