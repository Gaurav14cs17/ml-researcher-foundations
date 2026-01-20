<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Generative%20Models&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Mathematical Foundations

### Goal of Generative Models

Learn to sample from data distribution $p\_{data}(x)$:

```math
x_{new} \sim p_\theta(x) \approx p_{data}(x)

```

### Taxonomy

1. **Explicit Density:** Model $p\_\theta(x)$ directly (VAE, Flow)
2. **Implicit Density:** Learn to sample without explicit density (GAN)
3. **Score-Based:** Learn $\nabla\_x \log p(x)$ (Diffusion, Score Matching)

---

## 📐 Variational Autoencoder (VAE)

### Architecture

```math
\text{Encoder: } q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma^2_\phi(x))
\text{Decoder: } p_\theta(x|z) = \mathcal{N}(\mu_\theta(z), \sigma^2)

```

### Evidence Lower Bound (ELBO)

```math
\log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))

```

**Reconstruction term:** How well decoder reconstructs input.

**KL term:** Regularizes latent to match prior $p(z) = \mathcal{N}(0, I)$.

### Loss Function

```math
\mathcal{L}_{VAE} = -\mathbb{E}_{q_\phi}[\log p_\theta(x|z)] + D_{KL}(q_\phi(z|x) \| p(z))

```

For Gaussian decoder:

```math
= \frac{1}{2}\|x - \hat{x}\|^2 + \frac{1}{2}\sum_j\left(\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2\right)

```

### Reparameterization Trick

**Problem:** Can't backprop through sampling $z \sim q\_\phi(z|x)$.

**Solution:** 

```math
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)

```

Now $z$ is differentiable w.r.t. $\phi$!

---

## 📐 Generative Adversarial Network (GAN)

### Minimax Game

```math
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]

```

**Discriminator $D$:** Maximizes ability to distinguish real from fake.

**Generator $G$:** Minimizes discriminator's ability to detect fakes.

### Optimal Discriminator

Given fixed $G$, optimal discriminator:

```math
D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}

```

### Global Optimum

At optimum, $p\_g = p\_{data}$ and:

```math
V(D^*, G^*) = -\log 4
D^*(x) = \frac{1}{2} \quad \forall x

```

### Training Dynamics

**Discriminator update:**

```math
\nabla_{\theta_D} \frac{1}{m}\sum_{i=1}^m \left[\log D(x^{(i)}) + \log(1 - D(G(z^{(i)})))\right]

```

**Generator update:**

```math
\nabla_{\theta_G} \frac{1}{m}\sum_{i=1}^m \log(1 - D(G(z^{(i)})))

```

Or non-saturating loss:

```math
\nabla_{\theta_G} \frac{1}{m}\sum_{i=1}^m -\log D(G(z^{(i)}))

```

---

## 📐 Wasserstein GAN (WGAN)

### Motivation

Original GAN loss can have vanishing gradients when $D$ is too good.

### Wasserstein Distance

```math
W(p_r, p_g) = \inf_{\gamma \in \Pi(p_r, p_g)} \mathbb{E}_{(x,y) \sim \gamma}[\|x - y\|]

```

### Kantorovich-Rubinstein Duality

```math
W(p_r, p_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim p_r}[f(x)] - \mathbb{E}_{x \sim p_g}[f(x)]

```

### WGAN Objective

```math
\min_G \max_D \mathbb{E}_{x \sim p_{data}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]

```

Subject to $D$ being 1-Lipschitz.

### Lipschitz Constraint

**Weight Clipping (WGAN):** $w \leftarrow \text{clip}(w, -c, c)$

**Gradient Penalty (WGAN-GP):**

```math
\mathcal{L}_{GP} = \lambda \mathbb{E}_{\hat{x}}\left[(\|\nabla_{\hat{x}} D(\hat{x})\| - 1)^2\right]

```

Where $\hat{x} = \epsilon x + (1-\epsilon) G(z)$, $\epsilon \sim U(0,1)$.

---

## 📐 Normalizing Flows

### Key Idea

Learn invertible transformations from simple distribution.

```math
z_K = f_K \circ f_{K-1} \circ ... \circ f_1(z_0), \quad z_0 \sim \mathcal{N}(0, I)

```

### Change of Variables

```math
\log p(x) = \log p(z_0) - \sum_{k=1}^K \log\left|\det\frac{\partial f_k}{\partial z_{k-1}}\right|

```

### Requirements

- Each $f\_k$ must be invertible
- Jacobian determinant must be tractable

### Examples

- **RealNVP:** Affine coupling layers
- **Glow:** 1×1 convolutions + coupling
- **MAF:** Masked autoregressive flows

---

## 📊 Comparison

| Model | Training | Sample Quality | Diversity | Latent Space |
|-------|----------|----------------|-----------|--------------|
| **VAE** | Stable | Medium | High | Continuous |
| **GAN** | Unstable | High | Medium | Implicit |
| **Flow** | Stable | Medium-High | High | Invertible |
| **Diffusion** | Stable | Highest | High | Iterative |

---

## 💻 Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============ VAE ============

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20, hidden_dim=256):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def loss(self, x, recon, mu, logvar):
        # Reconstruction loss (BCE for binary data)
        recon_loss = F.binary_cross_entropy(recon, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kl_loss
    
    def sample(self, num_samples):
        z = torch.randn(num_samples, self.fc_mu.out_features)
        return self.decode(z)

# ============ GAN ============

class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_dim=784, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

def train_gan(G, D, dataloader, latent_dim, epochs=100):
    opt_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    for epoch in range(epochs):
        for real_batch in dataloader:
            batch_size = real_batch.size(0)
            
            # Train Discriminator
            z = torch.randn(batch_size, latent_dim)
            fake = G(z).detach()
            
            d_real = D(real_batch)
            d_fake = D(fake)
            
            loss_D = -torch.mean(torch.log(d_real + 1e-8) + torch.log(1 - d_fake + 1e-8))
            
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()
            
            # Train Generator
            z = torch.randn(batch_size, latent_dim)
            fake = G(z)
            d_fake = D(fake)
            
            loss_G = -torch.mean(torch.log(d_fake + 1e-8))  # Non-saturating loss
            
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

# ============ WGAN-GP ============

def gradient_penalty(D, real, fake):
    """Compute gradient penalty for WGAN-GP"""
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1).expand_as(real)
    
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)
    
    d_interpolated = D(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    
    return penalty

def train_wgan_gp(G, D, dataloader, latent_dim, lambda_gp=10, n_critic=5):
    opt_G = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0.0, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(0.0, 0.9))
    
    for real_batch in dataloader:
        batch_size = real_batch.size(0)
        
        # Train Critic (n_critic times)
        for _ in range(n_critic):
            z = torch.randn(batch_size, latent_dim)
            fake = G(z).detach()
            
            loss_D = D(fake).mean() - D(real_batch).mean()  # Wasserstein loss
            loss_D += lambda_gp * gradient_penalty(D, real_batch, fake)
            
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()
        
        # Train Generator
        z = torch.randn(batch_size, latent_dim)
        fake = G(z)
        loss_G = -D(fake).mean()
        
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | VAE Paper | [arXiv](https://arxiv.org/abs/1312.6114) |
| 📄 | GAN Paper | [arXiv](https://arxiv.org/abs/1406.2661) |
| 📄 | WGAN Paper | [arXiv](https://arxiv.org/abs/1701.07875) |
| 📄 | Normalizing Flows | [arXiv](https://arxiv.org/abs/1505.05770) |
| 🎥 | Generative Models (Stanford) | [YouTube](https://www.youtube.com/watch?v=5WoItGTWV54) |
| 🇨🇳 | 生成模型综述 | [知乎](https://zhuanlan.zhihu.com/p/34998569) |

---

## 🔗 Applications

| Model | Application |
|-------|-------------|
| **VAE** | Disentangled representations, anomaly detection |
| **GAN** | Image synthesis, super-resolution |
| **Flow** | Density estimation, compression |
| **Diffusion** | Image/video generation (SOTA) |

---

⬅️ [Back: Diffusion](../03_diffusion/README.md) | ➡️ [Next: GNN](../05_gnn/README.md)

---

⬅️ [Back: Architectures](../../README.md)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
