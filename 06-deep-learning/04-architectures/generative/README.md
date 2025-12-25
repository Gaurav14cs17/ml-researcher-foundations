<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Generative Models&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# 🎨 Generative Models

> **Learning to generate new data**

---

## 📐 Key Models

```
VAE (Variational Autoencoder):
  Loss = Reconstruction + KL(q(z|x) || p(z))

GAN (Generative Adversarial Network):
  min_G max_D E[log D(x)] + E[log(1-D(G(z)))]

Diffusion:
  Forward: Add noise gradually
  Reverse: Learn to denoise
```

---

## 💻 VAE Example

```python
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(784, 256)
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Linear(latent_dim, 784)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
```

---

## 🔗 Comparison

| Model | Training | Quality | Diversity |
|-------|----------|---------|-----------|
| **VAE** | Stable | Medium | High |
| **GAN** | Unstable | High | Medium |
| **Diffusion** | Stable | Highest | High |

---

⬅️ [Back: Architectures](../)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

