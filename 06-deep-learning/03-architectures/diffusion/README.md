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

## 🎯 Visual Overview

<img src="./images/diffusion-process.svg" width="100%">

*Caption: The diffusion process showing forward diffusion (adding noise progressively) and reverse diffusion (learning to denoise). The model learns to predict the noise at each step, enabling generation of new images from pure noise. This powers Stable Diffusion, DALL-E, and Midjourney.*

---

## 📐 Core Idea

```
Forward process: Add noise progressively
x₀ → x₁ → x₂ → ... → xₜ ≈ N(0, I)

Reverse process: Learn to denoise
xₜ → x̂ₜ₋₁ → ... → x̂₁ → x̂₀

Model learns: εθ(xₜ, t) ≈ ε (the noise added)
```

---

## 🔑 Training

```
1. Sample x₀ ~ data
2. Sample t ~ Uniform(1, T)
3. Sample ε ~ N(0, I)
4. Compute xₜ = √(ᾱₜ)x₀ + √(1-ᾱₜ)ε
5. Train: L = ||ε - εθ(xₜ, t)||²
```

---

## 🌍 Applications

| Model | Company | Use |
|-------|---------|-----|
| **Stable Diffusion** | Stability AI | Text-to-image |
| **DALL-E** | OpenAI | Text-to-image |
| **Midjourney** | Midjourney | Art generation |
| **Sora** | OpenAI | Text-to-video |

---

## 💻 Code (Simplified)

```python
def diffusion_loss(model, x0, t):
    # Add noise
    noise = torch.randn_like(x0)
    alpha_bar = get_alpha_bar(t)
    xt = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * noise
    
    # Predict noise
    noise_pred = model(xt, t)
    
    return F.mse_loss(noise_pred, noise)
```

---

## 📚 Resources

| Paper | Year |
|-------|------|
| [DDPM](https://arxiv.org/abs/2006.11239) | 2020 |
| [Stable Diffusion](https://arxiv.org/abs/2112.10752) | 2021 |


## 🔗 Where This Topic Is Used

| Application | Diffusion Model |
|-------------|----------------|
| **Stable Diffusion** | Text-to-image |
| **DALL-E 2/3** | Image generation |
| **Midjourney** | Art generation |
| **Video** | Video diffusion |
| **Audio** | AudioLM, MusicGen |

---

⬅️ [Back: Architectures](../)

---

⬅️ [Back: Cnn](../cnn/) | ➡️ [Next: Mlp](../mlp/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
