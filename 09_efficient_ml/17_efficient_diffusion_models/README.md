<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=120&section=header&text=Lecture%2017%20Efficient%20Diffusion%20Models&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-09-1ABC9C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

# Lecture 17: Efficient Diffusion Models

[← Back to Course](../) | [← Previous](../16_efficient_llms/) | [Next: Quantum ML →](../18_quantum_ml/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09_efficient_ml/17_efficient_diffusion_models/demo.ipynb) ← **Try the code!**

---

## 📺 Video Lecture

| Resource | Link |
|----------|------|
| 🎥 **Lecture Video** | [Watch on YouTube](https://www.youtube.com/watch?v=oVeaRWP1DYg&list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&index=17) |
| 📊 **Slides** | [MIT Course Page](https://hanlab.mit.edu/courses/2023-fall-65940) |
| ⏱️ **Duration** | ~90 minutes |

### 📝 Video Summary

This lecture covers **efficient diffusion model inference**:

- **Diffusion basics**: Forward and reverse processes
- **Why diffusion is slow**: 50-1000 neural network passes
- **Faster samplers**: DDIM, DPM-Solver for fewer steps
- **Latent diffusion**: Working in compressed latent space
- **Consistency models**: One-step generation
- **Distillation**: LCM, SDXL Turbo for real-time generation

> 💡 *"From 1000 steps to 1 step—that's the journey of diffusion model efficiency."* — Prof. Song Han

---

![Overview](overview.png)

## Diffusion Model Basics

Diffusion models generate images by **denoising**:

```
Training: Image → Add noise step by step → Pure noise
Inference: Pure noise → Remove noise step by step → Image

```

**Problem:** Requires many steps (50-1000) = Slow!

---

## 📐 Mathematical Foundations & Proofs

### Forward Diffusion Process

Add Gaussian noise gradually:

```math
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)

```

**Closed form for any \( t \):**

```math
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)

```

where \( \bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s) \).

**Reparameterization:**

```math
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)

```

---

### Reverse Process (DDPM)

**Learn to denoise:**

```math
p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)

```

**Training objective:**

```math
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]

```

**Sampling (stochastic):**

```math
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right) + \sigma_t z

```

where \( z \sim \mathcal{N}(0, I) \).

---

### DDIM: Deterministic Sampling

**Key insight:** Make the process deterministic by removing stochasticity.

**DDIM update:**

```math
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \epsilon_\theta(x_t, t)

```

where:

```math
\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}

```

**Advantage:** Can skip steps! Works with 50 steps instead of 1000.

---

### DPM-Solver: Higher-Order ODE Solver

**Diffusion as ODE:**

```math
\frac{dx}{dt} = f(x, t) = -\frac{\beta(t)}{2}\left(x + \frac{\epsilon_\theta(x,t)}{\sqrt{1-\bar{\alpha}_t}}\right)

```

**Euler method (first-order):** 100 steps needed.

**DPM-Solver (second-order):**

```math
x_{t-1} = e^{-h} x_t + (e^{-h} - 1) \epsilon_\theta(x_t, t) + \frac{e^{-h} - 1 + h}{h}(\epsilon_\theta(x_t, t) - \epsilon_\theta(x_{t-1}^{est}, t-1))

```

**20 steps sufficient!**

---

### Latent Diffusion (Stable Diffusion)

**Compress to latent space:**

```math
z = \mathcal{E}(x), \quad x = \mathcal{D}(z)

```

**Compression ratio:**

```math
\frac{|x|}{|z|} = \frac{512 \times 512 \times 3}{64 \times 64 \times 4} = 48\times

```

**FLOPs reduction:**

```math
\frac{\text{FLOPs}_{pixel}}{\text{FLOPs}_{latent}} \approx 8^2 = 64\times

```

(Factor of 8 in each spatial dimension)

---

### Progressive Distillation

**Teacher:** Diffusion model with T steps.
**Student:** Learns to match teacher in T/2 steps.

**Distillation loss:**

```math
\mathcal{L} = \mathbb{E}\left[\|x_{t-2}^{teacher} - \epsilon_\theta^{student}(x_t, t)\|^2\right]

```

Student predicts 2-step result in 1 step.

**Iterate:** T → T/2 → T/4 → ... → 4 → 2 → 1

---

### Consistency Models

**Key insight:** All points on the same trajectory should map to the same output.

**Consistency constraint:**

```math
f(x_t, t) = f(x_{t'}, t') \quad \forall t, t' \text{ on same trajectory}

```

**Training:** Enforce consistency between adjacent points:

```math
\mathcal{L} = \mathbb{E}\left[\|f_\theta(x_{t+1}, t+1) - f_{\theta^-}(x_t, t)\|^2\right]

```

where \( \theta^- \) is EMA of \( \theta \).

**One-step generation:** \( f(x_T, T) \) directly gives \( x_0 \)!

---

### LCM (Latent Consistency Models)

**Apply consistency to latent diffusion:**

```math
f_\theta(z_t, t, c) = z_0

```

**Classifier-free guidance in one step:**

Standard CFG:

```math
\hat{\epsilon} = \epsilon_\theta(z, c) + w(\epsilon_\theta(z, c) - \epsilon_\theta(z, \emptyset))

```

LCM learns to produce guided output directly.

**4 steps sufficient!**

---

## 🧮 Key Derivations

### Compute Cost Comparison

| Method | Steps | Time (A100) |
|--------|-------|-------------|
| DDPM | 1000 | 60s |
| DDIM | 50 | 3s |
| DPM-Solver | 20 | 1.2s |
| LCM | 4 | 0.3s |
| Consistency | 1 | 0.1s |

**1000× speedup** from DDPM to Consistency!

---

### Memory Requirements

**U-Net inference:**
- Model: ~3GB (SDXL)
- Activations: ~1GB per step
- VAE: ~0.5GB

**Total for 50 steps:** ~5GB peak.

**Optimization:** Recompute activations instead of storing.

---

### Quantization for Diffusion

**Challenge:** Diffusion models are sensitive to quantization.

**Solution:** INT8 weights + FP16 activations.

| Precision | FID Impact |
|-----------|-----------|
| FP16 | 0.0 |
| INT8 weights | +0.3 |
| INT8 full | +1.2 |

Keep activations in higher precision.

---

## 🎯 Where Used

| Concept | Applications |
|---------|-------------|
| DDIM/DPM++ | Fast image generation |
| LCM/Turbo | Real-time generation |
| Latent Diffusion | Stable Diffusion, SDXL |
| Consistency Models | Few-step generation |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [← Efficient LLMs](../16_efficient_llms/README.md) | [Efficient ML](../README.md) | [Quantum ML →](../18_quantum_ml/README.md) |

---
## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | DDPM | [arXiv](https://arxiv.org/abs/2006.11239) |
| 📄 | DDIM | [arXiv](https://arxiv.org/abs/2010.02502) |
| 📄 | Latent Diffusion | [arXiv](https://arxiv.org/abs/2112.10752) |
| 📄 | LCM | [arXiv](https://arxiv.org/abs/2310.04378) |
| 📄 | Consistency Models | [arXiv](https://arxiv.org/abs/2303.01469) |
| 🎥 | MIT 6.5940 TinyML | [Course](https://hanlab.mit.edu/courses/2024-fall-65940) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=80&section=footer" width="100%"/>
</p>
