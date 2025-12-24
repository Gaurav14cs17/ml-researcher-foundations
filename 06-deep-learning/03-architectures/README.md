<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=03 Architectures&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 🏗️ Architectures

> **Network designs for different tasks**

---

## 🎯 Visual Overview

<img src="./images/architecture-comparison.svg" width="100%">

*Caption: Evolution of neural network architectures from simple MLPs to modern Transformers and Diffusion models. Transformers (2017) revolutionized NLP and now dominate most domains. Diffusion models (2020) lead image generation.*

---

## 📐 Mathematical Foundations

### MLP (Multi-Layer Perceptron)
```
y = Wₙσ(Wₙ₋₁σ(...σ(W₁x + b₁)...) + bₙ₋₁) + bₙ

Parameters: Σᵢ dᵢ × dᵢ₊₁ + dᵢ₊₁
```

### CNN (Convolutional Neural Network)
```
Convolution:
(f * g)(x) = Σₖ f(k) g(x - k)

2D Image:
yᵢⱼ = Σₘ Σₙ Wₘₙ xᵢ₊ₘ,ⱼ₊ₙ + b

Output size: (W - K + 2P) / S + 1
```

### Transformer Self-Attention
```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V

Multi-head:
MultiHead = Concat(head₁, ..., headₕ) Wᴼ
headᵢ = Attention(XWᵢᴽ, XWᵢᴷ, XWᵢⱽ)
```

### Diffusion Forward/Reverse
```
Forward (add noise):
q(xₜ|xₜ₋₁) = N(xₜ; √(1-βₜ)xₜ₋₁, βₜI)

Reverse (denoise):
p_θ(xₜ₋₁|xₜ) = N(xₜ₋₁; μ_θ(xₜ, t), Σ_θ(xₜ, t))
```

---

## 📂 Topics

| Folder | Topic | Best For |
|--------|-------|----------|
| [mlp/](./mlp/) | Multi-layer perceptron | Tabular data |
| [cnn/](./cnn/) | Convolutional networks | Images |
| [rnn/](./rnn/) | Recurrent networks | Sequences |
| [transformer/](./transformer/) | 🔥 Attention | Everything! |
| [diffusion/](./diffusion/) | 🔥 Diffusion models | Generation |
| [moe/](./moe/) | Mixture of Experts | Scaling |

---

## 📊 Architecture Timeline

```
1980s: MLP
1990s: CNN (LeNet)
2010s: Deep CNN (AlexNet, ResNet)
2014:  GAN
2015:  LSTM/GRU for NLP
2017:  🔥 Transformer
2020:  ViT (Vision Transformer)
2020:  🔥 Diffusion Models
2022:  🔥 MoE (Mixtral)
```

---

## 🔑 Choosing Architecture

| Data Type | Architecture |
|-----------|-------------|
| Images | CNN or ViT |
| Text | Transformer |
| Sequences | Transformer (or RNN) |
| Generation | Diffusion, GAN |
| Tabular | MLP, Gradient Boosting |

---

## 🔗 Where Each Architecture Is Used

| Architecture | Used In These Topics/Models |
|-------------|----------------------------|
| **Transformer** | GPT, BERT, LLaMA, ViT, Whisper, CLIP, Stable Diffusion (cross-attention) |
| **CNN** | ResNet, YOLO, U-Net (diffusion), Image encoders in CLIP |
| **RNN/LSTM** | Seq2seq (older), Time series, Speech (older) |
| **Diffusion** | Stable Diffusion, DALL-E, Imagen, Video generation |
| **MoE** | Mixtral, Switch Transformer, GShard |
| **GAN** | StyleGAN, Image editing (older approach) |

### Architecture Dependencies

```
Neural Network Basics
    +--> MLP (simplest)
    +--> CNN --> ResNet --> U-Net (diffusion)
    +--> RNN --> LSTM --> (mostly replaced by Transformer)
    +--> Transformer --> GPT, BERT, ViT
                    --> Cross-attention in Diffusion
                    --> MoE (Transformer + routing)
```

### Which Architecture For Which Task

| Task | Architecture | Why |
|------|--------------|-----|
| Text generation | Transformer (decoder) | Long-range dependencies |
| Image classification | CNN or ViT | Spatial structure |
| Image generation | Diffusion + U-Net | State-of-art quality |
| Speech recognition | Transformer | Whisper architecture |
| Efficient scaling | MoE | Sparse activation |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Attention Is All You Need | [arXiv](https://arxiv.org/abs/1706.03762) |
| 📄 | ResNet Paper | [arXiv](https://arxiv.org/abs/1512.03385) |
| 📖 | Deep Learning Book | [Book](https://www.deeplearningbook.org/) |
| 🇨🇳 | 深度学习架构总结 | [知乎](https://zhuanlan.zhihu.com/p/54356280) |
| 🇨🇳 | 网络架构演进 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | 架构对比讲解 | [B站](https://www.bilibili.com/video/BV1J94y1f7u5) |

---

⬅️ [Back: 02-Backpropagation](../02-backpropagation/) | ➡️ [Next: 04-Training](../04-training/)


---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
