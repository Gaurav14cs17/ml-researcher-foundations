<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=ELBO%20in%20Diffusion%20Models&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# ELBO in Diffusion Models

> How optimization makes Stable Diffusion work

---

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

## 🔬 Connection to ELBO

```
Full ELBO decomposition:

L = L_0 + L_1 + ... + L_{T-1} + L_T

where each L_t is a KL divergence:

L_t = KL( q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t) )

Key insight:
• q(x_{t-1}|x_t,x_0) is Gaussian (tractable!)
• p_θ(x_{t-1}|x_t) is also Gaussian
• KL between Gaussians has closed form
• Reduces to ||ε - ε_θ||² loss!
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

## 💻 Training Code (Simplified)

```python
# Diffusion Model Training with ELBO-based Loss
import torch
import torch.nn as nn

def train_step(model, x_0, noise_schedule):
    # Sample random timestep
    t = torch.randint(0, T, (batch_size,))
    
    # Sample noise
    epsilon = torch.randn_like(x_0)
    
    # Create noisy image: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
    alpha_bar = noise_schedule.alpha_bar[t]
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * epsilon
    
    # Predict noise
    epsilon_pred = model(x_t, t)
    
    # ELBO-derived loss: ||ε - ε_θ(x_t, t)||²
    loss = nn.MSELoss()(epsilon_pred, epsilon)
    
    return loss
```

---

## 📊 Why This Works

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

## 🇨🇳 Chinese Resources

| 平台 | 内容 | 链接 |
|------|------|------|
| 知乎 | Diffusion模型原理 | [知乎](https://zhuanlan.zhihu.com/p/525106459) |
| B站 | DDPM详解视频 | [B站](https://www.bilibili.com/video/BV1b541197HX) |
| CSDN | 扩散模型数学推导 | [CSDN](https://blog.csdn.net/qq_39388410/article/details/126684954) |

---

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
