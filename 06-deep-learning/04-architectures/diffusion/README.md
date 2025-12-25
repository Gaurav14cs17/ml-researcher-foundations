<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Diffusion%20Models&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Diffusion Process

```
Forward (add noise):
  q(x_t|x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)

Reverse (learn to denoise):
  p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))

Training objective:
  L = E[||ε - ε_θ(x_t, t)||²]
```

---

## 💻 Simplified Example

```python
def diffusion_loss(model, x_0, t, noise):
    """Diffusion training step"""
    # Add noise to get x_t
    x_t = sqrt_alphas[t] * x_0 + sqrt_one_minus_alphas[t] * noise
    
    # Predict noise
    noise_pred = model(x_t, t)
    
    # MSE loss
    return F.mse_loss(noise_pred, noise)
```

---

## 🔗 Applications

| Model | Application |
|-------|-------------|
| **DDPM** | Image generation |
| **Stable Diffusion** | Text-to-image |
| **Video Diffusion** | Video generation |

---

⬅️ [Back: Architectures](../)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
