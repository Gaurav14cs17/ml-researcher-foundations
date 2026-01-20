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

## 📂 Overview

Diffusion models are generative models that learn to reverse a gradual noising process. They achieve state-of-the-art image generation by learning to denoise samples at each step.

---

## 📐 Mathematical Framework

### Forward Diffusion Process

Starting from data $x_0$, we gradually add Gaussian noise over $T$ steps:

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)
$$

where $\beta_t$ is the noise schedule ($\beta_t \in (0, 1)$).

**Key property: We can sample $x_t$ directly from $x_0$:**

$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)
$$

where:
- $\alpha_t = 1 - \beta_t$
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$

**Proof:**
```
By reparameterization, x_t = √α_t x_{t-1} + √β_t ε_{t-1}

Recursively:
x_t = √α_t (√α_{t-1} x_{t-2} + √β_{t-1} ε_{t-2}) + √β_t ε_{t-1}
    = √(α_t α_{t-1}) x_{t-2} + √(α_t β_{t-1}) ε_{t-2} + √β_t ε_{t-1}

Continuing to x_0:
x_t = √ᾱ_t x_0 + √(1 - ᾱ_t) ε

where ε ~ N(0, I) (sum of Gaussians is Gaussian)

This is because:
Var[√(α_t β_{t-1}) ε_{t-2} + √β_t ε_{t-1}] 
= α_t β_{t-1} + β_t 
= α_t(1-α_{t-1}) + (1-α_t)
= 1 - α_t α_{t-1}
= 1 - ᾱ_t (after recursion) ✓
```

### Reverse Process

We want to learn $p_\theta(x_{t-1} | x_t)$ to reverse the diffusion:

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

**The true reverse has a tractable form given $x_0$:**

$$
q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)
$$

where:

$$
\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t
\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t
$$

---

## 🔬 Training Objective

### Variational Lower Bound

$$
\log p(x_0) \geq \mathbb{E}_q\left[\log \frac{p(x_{0:T})}{q(x_{1:T}|x_0)}\right] = -L_{VLB}
$$

Decomposing the VLB:

$$
L_{VLB} = L_0 + L_1 + \cdots + L_{T-1} + L_T
$$

where:
- $L_0 = -\log p_\theta(x_0 | x_1)$ (reconstruction)
- $L_t = D_{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))$ (denoising)
- $L_T = D_{KL}(q(x_T|x_0) \| p(x_T))$ (prior, no learnable params)

### Simplified Training Loss (DDPM)

Instead of predicting $\mu_\theta$, we predict the noise $\epsilon_\theta$:

$$
L_{simple} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t)\|^2\right]
$$

**Why this works:**
```
The mean can be parameterized as:
μ_θ(x_t, t) = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))ε_θ(x_t, t))

If ε_θ(x_t, t) ≈ ε (the actual noise added), then:
μ_θ ≈ (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))ε)
    = (1/√α_t)(√ᾱ_t x_0 + √(1-ᾱ_t)ε - (β_t/√(1-ᾱ_t))ε)
    ≈ true posterior mean ✓
```

---

## 📊 Noise Schedules

### Linear Schedule (Original DDPM)

$$
\beta_t = \beta_1 + \frac{t-1}{T-1}(\beta_T - \beta_1)
$$

Typically: $\beta_1 = 10^{-4}$, $\beta_T = 0.02$, $T = 1000$

### Cosine Schedule (Improved)

$$
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2
$$

**Advantage:** More uniform signal-to-noise ratio across timesteps.

### Signal-to-Noise Ratio

$$
SNR(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}
$$

Linear schedule: SNR drops quickly, wastes steps
Cosine schedule: SNR decreases more uniformly

---

## 📐 Sampling Algorithms

### DDPM Sampling

```python

# Start from pure noise
x_T ~ N(0, I)

# Iteratively denoise
for t = T, T-1, ..., 1:
    z ~ N(0, I) if t > 1 else 0
    x_{t-1} = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))ε_θ(x_t, t)) + √β̃_t z
```

### DDIM (Deterministic Sampling)

Denoising Diffusion Implicit Models allow non-Markovian sampling:

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left(\frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{predicted } x_0} + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \cdot \epsilon_\theta(x_t, t) + \sigma_t \epsilon
$$

Setting $\sigma_t = 0$ gives deterministic sampling.

**Benefit:** Can skip steps (e.g., 50 steps instead of 1000).

---

## 💻 Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Time step embeddings using sinusoidal encoding
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings

class ResBlock(nn.Module):
    """
    Residual block with time embedding
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.norm1(F.silu(self.conv1(x)))
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        h = self.norm2(F.silu(self.conv2(h)))
        return h + self.shortcut(x)

class SimpleUNet(nn.Module):
    """
    Simplified U-Net for diffusion models
    """
    def __init__(self, in_channels=3, base_channels=64, time_emb_dim=256):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Encoder
        self.enc1 = ResBlock(in_channels, base_channels, time_emb_dim)
        self.enc2 = ResBlock(base_channels, base_channels * 2, time_emb_dim)
        self.enc3 = ResBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        
        # Bottleneck
        self.mid = ResBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        
        # Decoder
        self.dec3 = ResBlock(base_channels * 8, base_channels * 2, time_emb_dim)
        self.dec2 = ResBlock(base_channels * 4, base_channels, time_emb_dim)
        self.dec1 = ResBlock(base_channels * 2, base_channels, time_emb_dim)
        
        # Output
        self.out = nn.Conv2d(base_channels, in_channels, 1)
        
        # Downsampling and upsampling
        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x, t):
        """
        Args:
            x: (batch, channels, H, W) noisy input
            t: (batch,) timestep
        
        Returns:
            predicted noise (same shape as x)
        """

        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Encoder
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down(e1), t_emb)
        e3 = self.enc3(self.down(e2), t_emb)
        
        # Bottleneck
        m = self.mid(self.down(e3), t_emb)
        
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up(m), e3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1), t_emb)
        
        return self.out(d1)

class GaussianDiffusion:
    """
    DDPM Diffusion process
    """
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule='linear'):
        self.timesteps = timesteps
        
        # Noise schedule
        if schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule == 'cosine':
            self.betas = self._cosine_schedule(timesteps)
        
        # Pre-compute values
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1 - self.alpha_cumprod_prev) / (1 - self.alpha_cumprod)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alpha_cumprod_prev) / (1 - self.alpha_cumprod)
        self.posterior_mean_coef2 = (1 - self.alpha_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alpha_cumprod)
    
    def _cosine_schedule(self, timesteps, s=0.008):
        """Cosine schedule as proposed in improved DDPM"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alpha_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
        betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion: sample x_t from q(x_t | x_0)
        
        x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t][:, None, None, None]
        
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
    
    def p_mean_variance(self, model, x_t, t):
        """
        Compute mean and variance of p(x_{t-1} | x_t)
        """

        # Predict noise
        noise_pred = model(x_t, t)
        
        # Compute predicted x_0
        x_0_pred = (x_t - self.sqrt_one_minus_alpha_cumprod[t][:, None, None, None] * noise_pred) / \
                   self.sqrt_alpha_cumprod[t][:, None, None, None]
        x_0_pred = torch.clamp(x_0_pred, -1, 1)
        
        # Compute posterior mean
        posterior_mean = self.posterior_mean_coef1[t][:, None, None, None] * x_0_pred + \
                        self.posterior_mean_coef2[t][:, None, None, None] * x_t
        
        posterior_variance = self.posterior_variance[t][:, None, None, None]
        
        return posterior_mean, posterior_variance
    
    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        """
        Sample x_{t-1} from p(x_{t-1} | x_t)
        """
        mean, var = self.p_mean_variance(model, x_t, t)
        
        noise = torch.randn_like(x_t) if t[0] > 0 else 0
        
        return mean + torch.sqrt(var) * noise
    
    @torch.no_grad()
    def sample(self, model, shape, device):
        """
        Generate samples by iterative denoising
        """

        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch)
        
        return x
    
    def training_loss(self, model, x_0):
        """
        Compute training loss
        L = E[||ε - ε_θ(x_t, t)||²]
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Sample random timestep
        t = torch.randint(0, self.timesteps, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Get noisy x_t
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict noise
        noise_pred = model(x_t, t)
        
        # MSE loss
        return F.mse_loss(noise_pred, noise)

# DDIM Sampler (faster sampling)
class DDIMSampler:
    """
    DDIM: Denoising Diffusion Implicit Models
    Allows deterministic sampling and fewer steps
    """
    def __init__(self, diffusion, ddim_steps=50, eta=0.0):
        self.diffusion = diffusion
        self.ddim_steps = ddim_steps
        self.eta = eta  # 0 = deterministic, 1 = DDPM
        
        # Compute DDIM timesteps
        self.timesteps = torch.linspace(0, diffusion.timesteps - 1, ddim_steps).long()
    
    @torch.no_grad()
    def sample(self, model, shape, device):
        x = torch.randn(shape, device=device)
        
        for i in reversed(range(len(self.timesteps))):
            t = self.timesteps[i]
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = model(x, t_batch)
            
            # DDIM update
            alpha_t = self.diffusion.alpha_cumprod[t]
            alpha_prev = self.diffusion.alpha_cumprod[self.timesteps[i-1]] if i > 0 else torch.tensor(1.0)
            
            # Predicted x_0
            x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            
            # Direction pointing to x_t
            sigma = self.eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)
            
            # DDIM update
            x = torch.sqrt(alpha_prev) * x0_pred + \
                torch.sqrt(1 - alpha_prev - sigma**2) * noise_pred + \
                sigma * torch.randn_like(x)
        
        return x

# Example usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model
model = SimpleUNet(in_channels=3, base_channels=64).to(device)
diffusion = GaussianDiffusion(timesteps=1000, schedule='cosine')

# Training
x_0 = torch.randn(4, 3, 64, 64, device=device)  # Fake image batch
loss = diffusion.training_loss(model, x_0)
print(f"Training loss: {loss.item():.4f}")

# Sampling (slow with DDPM)
# samples = diffusion.sample(model, (4, 3, 64, 64), device)

# Fast sampling with DDIM
ddim = DDIMSampler(diffusion, ddim_steps=50)

# samples = ddim.sample(model, (4, 3, 64, 64), device)

print("Model ready for training!")
```

---

## 📊 Model Variants

| Model | Key Innovation | Use Case |
|-------|----------------|----------|
| **DDPM** | Original formulation | Foundation |
| **DDIM** | Deterministic sampling, fewer steps | Fast inference |
| **Stable Diffusion** | Latent space diffusion | Text-to-image |
| **Imagen** | Classifier-free guidance | High quality |
| **DALL-E 2** | CLIP + diffusion | Multi-modal |

---

## 📚 Key Insights

| Insight | Details |
|---------|---------|
| **Noise prediction** | Predicting ε is equivalent to predicting score |
| **Classifier-free guidance** | Interpolate conditional/unconditional for quality |
| **Latent diffusion** | Work in VAE latent space for efficiency |
| **Cosine schedule** | More uniform SNR than linear |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | DDPM | [Ho et al., 2020](https://arxiv.org/abs/2006.11239) |
| 📄 | DDIM | [Song et al., 2020](https://arxiv.org/abs/2010.02502) |
| 📄 | Stable Diffusion | [Rombach et al., 2022](https://arxiv.org/abs/2112.10752) |
| 🎥 | What are Diffusion Models? | [Lilian Weng](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [CNN](../02_cnn/README.md) | [Architectures](../README.md) | [Generative](../04_generative/README.md) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
