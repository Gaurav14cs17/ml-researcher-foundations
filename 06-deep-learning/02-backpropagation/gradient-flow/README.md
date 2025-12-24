<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Gradient Flow&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# Gradient Flow Problems

> **When gradients vanish or explode**

---

## 🎯 Visual Overview

<img src="./images/gradient-flow.svg" width="100%">

*Caption: Healthy gradient flow maintains O(1) gradients throughout the network. Vanishing gradients cause early layers to not learn; exploding gradients cause NaN. Solutions: skip connections, normalization, better activations (ReLU), proper initialization.*

---

## 📐 Mathematical Foundations

### Gradient Through Layers
```
For L-layer network:
∂L/∂W₁ = ∂L/∂hₗ · (∏ᵢ₌₂ᴸ ∂hᵢ/∂hᵢ₋₁) · ∂h₁/∂W₁

Jacobian at each layer:
∂hᵢ/∂hᵢ₋₁ = Wᵢ · diag(σ'(zᵢ₋₁))
```

### Spectral Analysis
```
Product of Jacobians:
∏ᵢ₌₂ᴸ ∂hᵢ/∂hᵢ₋₁

Singular values determine gradient magnitude:
• σₘₐₓ > 1 repeatedly → exploding
• σₘₐₓ < 1 repeatedly → vanishing

ResNet fix:
∂h/∂x = I + ∂F/∂x  (identity preserves gradient)
```

### Initialization for Stable Flow
```
Xavier (sigmoid/tanh):
W ~ N(0, 1/n_in)

He (ReLU):
W ~ N(0, 2/n_in)

Goal: Var(output) ≈ Var(input) at each layer
```

---

## 🔥 Vanishing Gradients

```
∂L/∂W₁ = ∂L/∂hₙ · (∏ᵢ₌₂ⁿ ∂hᵢ/∂hᵢ₋₁) · ∂h₁/∂W₁

If each |∂hᵢ/∂hᵢ₋₁| < 1:
Product → 0 as n grows!

Symptoms:
• Early layers don't learn
• Training stalls
```

---

## 💥 Exploding Gradients

```
If each |∂hᵢ/∂hᵢ₋₁| > 1:
Product → ∞ as n grows!

Symptoms:
• NaN loss
• Weights become very large
```

---

## ✅ Solutions

| Problem | Solution |
|---------|----------|
| Vanishing | ReLU, ResNet, LayerNorm |
| Exploding | Gradient clipping, careful init |
| Both | Skip connections (ResNet) |

```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Skip connection
y = x + F(x)  # ResNet style
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | ResNet Paper | [arXiv](https://arxiv.org/abs/1512.03385) |
| 📄 | Batch Normalization | [arXiv](https://arxiv.org/abs/1502.03167) |
| 🎥 | Karpathy: Building GPT | [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY) |
| 🇨🇳 | 梯度消失与爆炸 | [知乎](https://zhuanlan.zhihu.com/p/25631496) |
| 🇨🇳 | ResNet残差连接原理 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88692979) |
| 🇨🇳 | 深度学习梯度问题 | [B站](https://www.bilibili.com/video/BV1Y64y1Q7hi) |


## 🔗 Where This Topic Is Used

| Issue | Solution |
|-------|---------|
| **Vanishing Gradients** | ReLU, residuals, normalization |
| **Exploding Gradients** | Gradient clipping |
| **Dead ReLU** | Leaky ReLU, ELU |
| **Skip Connections** | ResNet, DenseNet |

---

⬅️ [Back: Backpropagation](../)

---

⬅️ [Back: Computational Graph](../computational-graph/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
