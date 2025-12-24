# 🎯 Learning Frameworks

> **Types of machine learning**

---

## 🎯 Visual Overview

<img src="./images/learning-paradigms.svg" width="100%">

*Caption: The four main learning paradigms in ML. The modern training pipeline (GPT, Claude, LLaMA) combines self-supervised pretraining, supervised fine-tuning, and RLHF for alignment.*

---

## 📐 Mathematical Foundations

### Supervised Learning
```
Given: {(xᵢ, yᵢ)}ᵢ₌₁ᴺ
Goal: Learn f: X → Y minimizing:
      L = (1/N) Σᵢ ℓ(f(xᵢ), yᵢ)

Classification: ℓ = cross-entropy
Regression: ℓ = MSE
```

### Self-Supervised Learning
```
Given: {xᵢ}ᵢ₌₁ᴺ (no labels!)
Create: pseudo-labels from data

Masked LM (BERT):
L = -log P(x_masked | x_visible)

Next token (GPT):
L = -Σₜ log P(xₜ | x₁,...,xₜ₋₁)

Contrastive (SimCLR):
L = -log(exp(zᵢ·zⱼ/τ) / Σₖ exp(zᵢ·zₖ/τ))
```

### Unsupervised Learning
```
Given: {xᵢ}ᵢ₌₁ᴺ
Goal: Find structure (no labels at all)

Clustering: argmin Σᵢ ||xᵢ - μ_cluster(i)||²
PCA: argmax Var(Wx) s.t. WᵀW = I
VAE: Maximize ELBO = E[log p(x|z)] - KL(q(z|x)||p(z))
```

---

## 📂 Topics

| Folder | Topic | Key Idea |
|--------|-------|----------|
| [supervised/](./supervised/) | Labeled data | (x, y) pairs |
| [unsupervised/](./unsupervised/) | No labels | Find structure |
| [self-supervised/](./self-supervised/) | Create labels | BERT, SimCLR |

---

## 📊 Comparison

| Framework | Input | Output | Examples |
|-----------|-------|--------|----------|
| Supervised | (x, y) | Predictor | Classification, regression |
| Unsupervised | x | Structure | Clustering, PCA |
| Self-supervised | x | Features | BERT, GPT, SimCLR |
| Reinforcement | (s, a, r) | Policy | Games, robotics |

---

## 🔥 Modern Paradigm

```
1. Self-supervised pre-training (massive data, no labels)
   +-- BERT, GPT, SimCLR, MAE
   
2. Supervised fine-tuning (small labeled data)
   +-- Task-specific adaptation

3. (Optional) RLHF for alignment
   +-- ChatGPT, Claude
```

---

## 🔗 Where This Topic Is Used

| Framework | Used In These Topics |
|-----------|---------------------|
| **Supervised** | Classification, Regression, Object Detection, Segmentation |
| **Self-Supervised** | BERT (MLM), GPT (next token), SimCLR, MAE, CLIP |
| **Unsupervised** | Clustering, PCA, VAE, GAN (no labels) |
| **Semi-Supervised** | Pseudo-labeling, MixMatch, FixMatch |
| **Reinforcement** | Q-learning, PPO, RLHF |
| **Meta-Learning** | MAML, few-shot learning |
| **Contrastive** | SimCLR, CLIP, InfoNCE loss |

### Framework Used By Model

| Model | Learning Framework |
|-------|-------------------|
| **GPT** | Self-supervised (next token) |
| **BERT** | Self-supervised (masked LM) |
| **CLIP** | Contrastive (image-text) |
| **Stable Diffusion** | Self-supervised (denoising) |
| **ChatGPT** | Self-supervised → SFT → RLHF |
| **ResNet** | Supervised (ImageNet) |
| **SimCLR** | Self-supervised (contrastive) |

### Modern Training Pipeline

```
Self-Supervised Pretraining --> Supervised Fine-tuning --> RLHF
(GPT learns language)          (learns task)              (learns preference)
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | BERT Paper | [arXiv](https://arxiv.org/abs/1810.04805) |
| 📄 | SimCLR Paper | [arXiv](https://arxiv.org/abs/2002.05709) |
| 📖 | Goodfellow Deep Learning | [Book](https://www.deeplearningbook.org/) |
| 🇨🇳 | 学习范式对比 | [知乎](https://zhuanlan.zhihu.com/p/25110450) |
| 🇨🇳 | 自监督学习综述 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | 机器学习入门 | [B站](https://www.bilibili.com/video/BV164411b7dx) |

---

⬅️ [Back: ML Theory](../) | ➡️ [Next: 02-Generalization](../02-generalization/)

