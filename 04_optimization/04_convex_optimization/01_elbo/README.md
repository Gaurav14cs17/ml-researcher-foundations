<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=ELBO%20%26%20Diffusion%20Models&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

# Part 1: ELBO (Evidence Lower Bound)

## 🎯 What is ELBO?

```
Problem: We want to maximize log p(x) (log-likelihood)
         But it's intractable!

Solution: Maximize a lower bound instead = ELBO

+-----------------------------------------------------+

|                                                     |
|   log p(x) = ELBO + KL(q || p)                     |
|                                                     |
|   Since KL ≥ 0, we have:                           |
|                                                     |
|   log p(x) ≥ ELBO                                  |
|                                                     |
|   Maximizing ELBO ≈ Maximizing log p(x)            |
|                                                     |
+-----------------------------------------------------+

```

---

## 📐 Formula

```
ELBO = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))
       ---------------------   ------------------
       Reconstruction term      Regularization term
       
       "How well can we         "Stay close to 
        reconstruct x?"          the prior p(z)"

```

---

## 📐 Complete ELBO Derivation

```
Goal: Maximize log p(x) = log ∫ p(x,z) dz  (intractable!)

Step 1: Introduce variational distribution q(z|x)
  log p(x) = log ∫ p(x,z) dz
           = log ∫ q(z|x) [p(x,z)/q(z|x)] dz
           = log E_q[p(x,z)/q(z|x)]

Step 2: Apply Jensen's inequality (log is concave)
  log E_q[p(x,z)/q(z|x)] ≥ E_q[log(p(x,z)/q(z|x))]
  
  Therefore: log p(x) ≥ E_q[log p(x,z) - log q(z|x)] = ELBO

Step 3: Decompose ELBO
  ELBO = E_q[log p(x,z)] - E_q[log q(z|x)]
       = E_q[log p(x|z) + log p(z)] - E_q[log q(z|x)]
       = E_q[log p(x|z)] + E_q[log p(z)] - E_q[log q(z|x)]
       = E_q[log p(x|z)] - E_q[log q(z|x) - log p(z)]
       = E_q[log p(x|z)] - KL(q(z|x) || p(z))

Step 4: Show the gap is a KL divergence
  log p(x) - ELBO = log p(x) - E_q[log p(x,z)] + E_q[log q(z|x)]
                  = E_q[log p(x) - log p(x,z) + log q(z|x)]
                  = E_q[log q(z|x) - log p(z|x)]
                  = KL(q(z|x) || p(z|x)) ≥ 0

Therefore: log p(x) = ELBO + KL(q(z|x) || p(z|x)) ✓

```

---

## 📊 Three Ways to Write ELBO

```
1. ELBO = E_q[log p(x,z)] - E_q[log q(z)]

2. ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))

3. ELBO = log p(x) - KL(q(z|x) || p(z|x))

```

---

## 🌍 Where ELBO is Used

| Model | How ELBO is Used | Paper |
|-------|------------------|-------|
| **VAE** | Main training objective | [Kingma 2013](https://arxiv.org/abs/1312.6114) |
| **Diffusion Models** | Variational bound on likelihood | [DDPM 2020](https://arxiv.org/abs/2006.11239) |
| **Bayesian NN** | Approximate posterior | [Weight Uncertainty](https://arxiv.org/abs/1505.05424) |
| **LLM Fine-tuning** | RLHF uses variational methods | [InstructGPT](https://arxiv.org/abs/2203.02155) |
| **Normalizing Flows** | Tighter ELBO with flows | [Rezende 2015](https://arxiv.org/abs/1505.05770) |

---

## 🔗 Connection to Optimization

```
ELBO is convex in certain parameterizations!

For exponential family:
• ELBO is concave in natural parameters
• Can use convex optimization tools
• Coordinate ascent works well

For neural networks:
• Non-convex in weights
• Use SGD/Adam
• Local optima issues

```

---

# Part 2: ELBO in Diffusion Models

## 🎨 Diffusion Models Overview

```
Forward Process (Add Noise):
x_0 --> x_1 --> x_2 --> ... --> x_T
 |       |       |              |
 v       v       v              v
Clean   Noisy   Noisier    Pure Noise

Reverse Process (Denoise):
x_T --> x_{T-1} --> ... --> x_1 --> x_0
 |         |               |       |
 v         v               v       v
Noise   Less Noisy      Cleaner  Clean!

```

---

## 📐 ELBO for Diffusion

```
+---------------------------------------------------------+

|                                                         |
|  log p(x_0) ≥ ELBO = E_q[ log p(x_T)                   |
|                          + Σ log p(x_{t-1}|x_t)        |
|                          - Σ log q(x_t|x_{t-1}) ]      |
|                                                         |
+---------------------------------------------------------+

Simplified Training Objective (DDPM):

L_simple = E_{t,x_0,ε}[ ||ε - ε_θ(x_t, t)||² ]

• t = random timestep
• ε = noise added at step t  
• ε_θ = neural network predicting noise

```

---

## 📐 Full ELBO Decomposition for Diffusion

```
Step 1: Forward process (fixed)
  q(x_{1:T}|x_0) = Π_{t=1}^T q(x_t|x_{t-1})
  
  where q(x_t|x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)

Step 2: Reverse process (learned)
  p_θ(x_{0:T}) = p(x_T) Π_{t=1}^T p_θ(x_{t-1}|x_t)

Step 3: ELBO decomposition
  log p(x_0) ≥ E_q[ log p(x_T)/q(x_T|x_0) 
                   + Σ_{t=2}^T log p_θ(x_{t-1}|x_t)/q(x_{t-1}|x_t,x_0)
                   + log p_θ(x_0|x_1) ]
             
             = -L_T - Σ_{t=2}^T L_{t-1} - L_0

  where:
    L_T = KL(q(x_T|x_0) || p(x_T))           (constant, no learning)
    L_{t-1} = KL(q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t))
    L_0 = -log p_θ(x_0|x_1)                  (reconstruction)

Step 4: Key insight - q(x_{t-1}|x_t,x_0) is Gaussian!
  q(x_{t-1}|x_t,x_0) = N(x_{t-1}; μ̃_t(x_t,x_0), β̃_t I)
  
  where:
    μ̃_t = (√ᾱ_{t-1} β_t)/(1-ᾱ_t) x_0 + (√α_t(1-ᾱ_{t-1}))/(1-ᾱ_t) x_t
    β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) β_t

Step 5: Parameterize p_θ to match
  p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), σ_t² I)
  
  Model predicts μ_θ, which depends on predicting x_0 or ε

Step 6: Simplify to noise prediction
  Since x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
  We have x_0 = (x_t - √(1-ᾱ_t) ε) / √ᾱ_t
  
  Substituting: L_{t-1} ∝ ||ε - ε_θ(x_t, t)||²

```

---

## 🔬 Why This Works

```
Mathematical Chain:

1. Want: max log p(x_0)
2. Use: ELBO as lower bound
3. Decompose: into KL terms per timestep
4. Each KL: between Gaussians
5. Simplify: to MSE loss on noise
6. Train: with SGD/Adam

The "score" ε_θ approximates:
∇_x log p(x_t) ≈ -ε_θ(x_t,t) / √(1-ᾱ_t)

```

---

## 💻 Training Code (Simplified)

```python
# Diffusion Model Training with ELBO-based Loss
import torch
import torch.nn as nn

def train_step(model, x_0, noise_schedule, T=1000):
    """
    One training step for diffusion model
    
    Args:
        model: UNet that predicts noise ε_θ(x_t, t)
        x_0: Clean images [batch, channels, H, W]
        noise_schedule: Contains α, β, ᾱ values
        T: Total timesteps
    """
    batch_size = x_0.shape[0]
    
    # Sample random timestep
    t = torch.randint(0, T, (batch_size,), device=x_0.device)
    
    # Sample noise
    epsilon = torch.randn_like(x_0)
    
    # Create noisy image: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
    alpha_bar = noise_schedule.alpha_bar[t]
    alpha_bar = alpha_bar.view(-1, 1, 1, 1)  # Broadcast
    
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * epsilon
    
    # Predict noise
    epsilon_pred = model(x_t, t)
    
    # ELBO-derived loss: ||ε - ε_θ(x_t, t)||²
    loss = nn.MSELoss()(epsilon_pred, epsilon)
    
    return loss

@torch.no_grad()
def sample(model, noise_schedule, shape, T=1000, device='cuda'):
    """
    Sample from diffusion model using reverse process
    """
    # Start from pure noise
    x = torch.randn(shape, device=device)
    
    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, device=device)
        
        # Predict noise
        epsilon_pred = model(x, t_batch)
        
        # Reverse step
        alpha = noise_schedule.alpha[t]
        alpha_bar = noise_schedule.alpha_bar[t]
        beta = noise_schedule.beta[t]
        
        # Mean of p_θ(x_{t-1}|x_t)
        mean = (1 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1 - alpha_bar)) * epsilon_pred
        )
        
        # Add noise (except at t=0)
        if t > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(beta)
            x = mean + sigma * noise
        else:
            x = mean
    
    return x

```

---

## 🌍 Real-World Applications

| Model | What it Does | Uses ELBO? |
|-------|--------------|------------|
| **Stable Diffusion** | Text → Image | ✅ Yes (simplified) |
| **DALL-E 2** | Text → Image | ✅ Yes (prior) |
| **Midjourney** | Text → Image | ✅ Likely |
| **Sora** (OpenAI) | Text → Video | ✅ Yes |
| **AudioLM** | Audio generation | ✅ Yes |

---

## 🔗 Related Optimization Concepts

| Concept | Role in Diffusion |
|---------|-------------------|
| **ELBO** | Training objective |
| **KL Divergence** | Each timestep loss |
| **SGD/Adam** | Optimizer for training |
| **Score Matching** | Alternative view |
| **Langevin Dynamics** | Sampling process |

---

## 📚 Key Papers

| Paper | Contribution | Link |
|-------|--------------|------|
| DDPM (Ho 2020) | Modern diffusion framework | [arXiv](https://arxiv.org/abs/2006.11239) |
| Score Matching (Song 2019) | Score-based view | [arXiv](https://arxiv.org/abs/1907.05600) |
| Improved DDPM (Nichol 2021) | Better sampling | [arXiv](https://arxiv.org/abs/2102.09672) |
| Stable Diffusion | Latent diffusion | [arXiv](https://arxiv.org/abs/2112.10752) |

---

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 📄 Paper | VAE Original | [arXiv:1312.6114](https://arxiv.org/abs/1312.6114) |
| 📄 Paper | DDPM (Diffusion) | [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) |
| 📄 Tutorial | VI Tutorial | [arXiv:1601.00670](https://arxiv.org/abs/1601.00670) |
| 🇨🇳 知乎 | ELBO推导详解 | [知乎](https://zhuanlan.zhihu.com/p/22464760) |
| 🇨🇳 知乎 | Diffusion模型原理 | [知乎](https://zhuanlan.zhihu.com/p/525106459) |
| 🇨🇳 B站 | DDPM详解视频 | [B站](https://www.bilibili.com/video/BV1b541197HX) |

---

⬅️ [Back: Convex Optimization](../) | ➡️ [Next: Constrained Optimization](../../05_constrained_optimization/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
