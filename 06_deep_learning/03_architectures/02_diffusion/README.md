<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Diffusion%20Models&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/diffusion-process.svg" width="100%">

*Caption: The diffusion process showing forward diffusion (adding noise progressively) and reverse diffusion (learning to denoise). The model learns to predict the noise at each step, enabling generation of new images from pure noise. This powers Stable Diffusion, DALL-E, and Midjourney.*

---

## 📐 Mathematical Foundations

### Forward Diffusion Process

**Definition:** Gradually add Gaussian noise over $T$ timesteps.

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)
$$

Where $\beta\_t$ is the noise schedule (variance at step $t$).

**Closed-form for any timestep:**

Define $\alpha\_t = 1 - \beta\_t$ and $\bar{\alpha}\_t = \prod\_{s=1}^{t} \alpha\_s$

$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)
$$

**Reparameterization:**

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

### Reverse Diffusion Process

**Goal:** Learn to reverse the forward process.

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

**Posterior (tractable when conditioned on $x\_0$):**

$$
q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)
$$

Where:

$$
\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t
\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t
$$

---

## 🔬 Training Objective

### Variational Lower Bound (ELBO)

$$
\log p(x_0) \geq \mathbb{E}_q\left[\log \frac{p(x_{0:T})}{q(x_{1:T}|x_0)}\right] = -L_{VLB}
$$

**Simplified Loss (DDPM):**

Instead of predicting $\mu$, predict the noise $\epsilon$:

$$
L_{simple} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]
$$

**Proof of equivalence:**

From $x\_t = \sqrt{\bar{\alpha}\_t} x\_0 + \sqrt{1-\bar{\alpha}\_t} \epsilon$:

$$
x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon}{\sqrt{\bar{\alpha}_t}}
$$

Substituting into $\tilde{\mu}\_t$:

$$
\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon\right)
$$

So predicting $\epsilon$ is equivalent to predicting $\mu$!

### Training Algorithm

```python
def train_step(model, x0):

    # 1. Sample timestep
    t = torch.randint(0, T, (batch_size,))
    
    # 2. Sample noise
    epsilon = torch.randn_like(x0)
    
    # 3. Create noisy image
    alpha_bar_t = alpha_bar[t]
    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon
    
    # 4. Predict noise
    epsilon_pred = model(x_t, t)
    
    # 5. Compute loss
    loss = F.mse_loss(epsilon_pred, epsilon)
    
    return loss
```

---

## 🔬 Sampling (Generation)

### DDPM Sampling

Starting from $x\_T \sim \mathcal{N}(0, I)$:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z
$$

Where $z \sim \mathcal{N}(0, I)$ and $\sigma\_t = \sqrt{\beta\_t}$ or $\sigma\_t = \sqrt{\tilde{\beta}\_t}$.

### DDIM Sampling (Deterministic)

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left(\frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{predicted } x_0} + \sqrt{1-\bar{\alpha}_{t-1}} \epsilon_\theta
$$

**Advantage:** Can skip steps (e.g., 1000 → 50 steps) for faster generation.

---

## 📊 Noise Schedules

| Schedule | Formula | Properties |
|----------|---------|------------|
| **Linear** | $\beta\_t = \beta\_1 + (t-1)\frac{\beta\_T - \beta\_1}{T-1}$ | Simple, original DDPM |
| **Cosine** | $\bar{\alpha}\_t = \cos^2\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)$ | Smoother, better for images |
| **Sigmoid** | $\bar{\alpha}\_t = \sigma(-a + 2a \cdot t/T)$ | Flexible endpoints |

### Cosine Schedule Derivation

$$
f(t) = \cos^2\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)
\bar{\alpha}_t = \frac{f(t)}{f(0)}
\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}} = 1 - \frac{f(t)}{f(t-1)}
$$

---

## 🔬 Classifier-Free Guidance

**Conditional Generation:**

$$
\tilde{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \varnothing) + w \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing))
$$

Where:
- $c$ = conditioning (e.g., text prompt)
- $\varnothing$ = unconditional (null prompt)
- $w$ = guidance scale (typically 7.5)

**Intuition:** Amplify the direction pointing from "unconditional" to "conditional".

---

## 🔬 Latent Diffusion (Stable Diffusion)

**Problem:** Diffusion in pixel space is expensive ($256 \times 256 \times 3$ = 196K dimensions).

**Solution:** Diffuse in latent space of a pretrained VAE.

$$
z = \text{Encoder}(x), \quad \hat{x} = \text{Decoder}(\hat{z})
$$

**Compression:** $256 \times 256 \times 3 \rightarrow 32 \times 32 \times 4$ (64× reduction!)

**Architecture:**
```
Text → CLIP Text Encoder → Cross-Attention
                                 ↓
Random z_T → U-Net (denoising) → z_0 → VAE Decoder → Image
```

---

## 💻 Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DiffusionModel:
    def __init__(self, model, T=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.T = T
        
        # Linear schedule
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        # Precompute coefficients
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)
    
    def forward_diffusion(self, x0, t, noise=None):
        """Add noise to x0 at timestep t"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)
        
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise, noise
    
    def loss(self, x0):
        """Training loss: predict the noise"""
        batch_size = x0.shape[0]
        t = torch.randint(0, self.T, (batch_size,), device=x0.device)
        
        x_t, noise = self.forward_diffusion(x0, t)
        noise_pred = self.model(x_t, t)
        
        return F.mse_loss(noise_pred, noise)
    
    @torch.no_grad()
    def sample(self, shape, device):
        """Generate samples via reverse diffusion"""
        x = torch.randn(shape, device=device)
        
        for t in reversed(range(self.T)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.model(x, t_batch)
            
            # Compute x_{t-1}
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            beta = self.betas[t]
            
            mean = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_bar)) * noise_pred
            )
            
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta) * noise
            else:
                x = mean
        
        return x

class SinusoidalEmbedding(nn.Module):
    """Timestep embedding using sinusoidal functions"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

# Simple U-Net for diffusion
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
        )
        
        # Encoder
        self.down1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        
        # Middle
        self.mid = nn.Conv2d(128, 128, 3, padding=1)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.up2 = nn.Conv2d(64, out_channels, 3, padding=1)
        
        self.time_proj = nn.Linear(time_dim, 128)
    
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        h = F.gelu(self.down1(x))
        h = F.gelu(self.down2(h))
        
        # Add time embedding
        h = h + self.time_proj(t_emb)[:, :, None, None]
        
        h = F.gelu(self.mid(h))
        h = F.gelu(self.up1(h))
        h = self.up2(h)
        
        return h
```

---

## 🌍 Applications

| Model | Company | Use |
|-------|---------|-----|
| **Stable Diffusion** | Stability AI | Text-to-image |
| **DALL-E 2/3** | OpenAI | Text-to-image |
| **Midjourney** | Midjourney | Art generation |
| **Sora** | OpenAI | Text-to-video |
| **AudioLDM** | - | Text-to-audio |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | DDPM Paper | [arXiv](https://arxiv.org/abs/2006.11239) |
| 📄 | DDIM Paper | [arXiv](https://arxiv.org/abs/2010.02502) |
| 📄 | Stable Diffusion | [arXiv](https://arxiv.org/abs/2112.10752) |
| 📄 | Classifier-Free Guidance | [arXiv](https://arxiv.org/abs/2207.12598) |
| 🎥 | What are Diffusion Models? | [YouTube](https://www.youtube.com/watch?v=HoKDTa5jHvg) |
| 🇨🇳 | 扩散模型详解 | [知乎](https://zhuanlan.zhihu.com/p/525106459) |

---

⬅️ [Back: CNN](../01_cnn/README.md) | ➡️ [Next: MLP](../03_mlp/README.md)

---

⬅️ [Back: Architectures](../../README.md)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
