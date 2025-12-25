<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Adam%20Optimizer&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# Adam Optimizer

> **Ad**aptive **M**oment Estimation — The default optimizer for deep learning

## 📊 Visual Intuition

```
SGD:                    Adam:
    •                      •
    |╲                     ╲
    | ╲                     ╲
    |  ╲                     ╲
    |   ╲                     •
    •    •               (faster!)
    
Oscillates              Smooth path
in ravines              with momentum
```

## 📐 Key Formula

```
+-----------------------------------------------------+
|  First Moment (Momentum):                           |
|  m_t = β₁ m_{t-1} + (1 - β₁) g_t                   |
|                                                     |
|  Second Moment (Adaptive LR):                       |
|  v_t = β₂ v_{t-1} + (1 - β₂) g_t²                  |
|                                                     |
|  Bias Correction:                                   |
|  m̂_t = m_t / (1 - β₁ᵗ)                             |
|  v̂_t = v_t / (1 - β₂ᵗ)                             |
|                                                     |
|  Update:                                            |
|  θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)             |
+-----------------------------------------------------+
```

## ⚙️ Default Hyperparameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| α | 0.001 | Learning rate |
| β₁ | 0.9 | First moment decay |
| β₂ | 0.999 | Second moment decay |
| ε | 10⁻⁸ | Numerical stability |

## 📈 Why Adam Works

```
Component          What it does
-------------------------------------
First moment (m)   Like momentum, smooths gradients
                   → Faster through flat regions
                   
Second moment (v)  Scales by gradient magnitude  
                   → Smaller steps where gradients large
                   → Larger steps where gradients small
                   
Bias correction    Fixes initialization bias
                   → Important in early steps
```

## 🔄 Adam Variants

| Variant | Change | Paper |
|---------|--------|-------|
| **AdamW** | Decoupled weight decay | [arXiv](https://arxiv.org/abs/1711.05101) |
| **RAdam** | Rectified Adam | [arXiv](https://arxiv.org/abs/1908.03265) |
| **Lookahead** | K steps ahead | [arXiv](https://arxiv.org/abs/1907.08610) |
| **AdaBelief** | Adapts to gradient "belief" | [arXiv](https://arxiv.org/abs/2010.07468) |

## 💻 Code Example

```python
# PyTorch
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8
)

# AdamW (recommended for transformers)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)
```

## ⚠️ When NOT to Use Adam

- Simple convex problems (SGD is fine)
- When you need best generalization (SGD often better)
- Very large batch sizes (LAMB optimizer better)

## 📚 Resources

| Type | Resource | Link |
|------|----------|------|
| 📄 Paper | Adam Original | [arXiv:1412.6980](https://arxiv.org/abs/1412.6980) |
| 📄 Paper | AdamW | [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) |
| 📖 Docs | PyTorch Optimizers | [PyTorch](https://pytorch.org/docs/stable/optim.html) |
| 🎥 Video | Adam Explained | [YouTube](https://www.youtube.com/watch?v=JXQT_vxqwIs) |
| 🇨🇳 知乎 | Adam优化器详解 | [知乎](https://zhuanlan.zhihu.com/p/32230623) |
| 🇨🇳 CSDN | Adam原理 | [CSDN](https://blog.csdn.net/willduan1/article/details/78070086) |

---

---

➡️ [Next: Sgd](./sgd.md)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
