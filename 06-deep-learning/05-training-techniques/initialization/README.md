<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Weight Initialization&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# ⚙️ Weight Initialization

> **Starting weights for stable training**

---

## 📐 Methods

```
Xavier/Glorot (sigmoid/tanh):
  W ~ N(0, 2/(n_in + n_out))
  
He/Kaiming (ReLU):
  W ~ N(0, 2/n_in)

Why: Keep variance of activations constant across layers
```

---

## 💻 Code

```python
# Xavier
nn.init.xavier_uniform_(layer.weight)

# He/Kaiming
nn.init.kaiming_normal_(layer.weight, mode='fan_in')

# Orthogonal (RNNs)
nn.init.orthogonal_(layer.weight)
```

---

## 🔗 Guidelines

| Activation | Initialization |
|------------|----------------|
| **tanh/sigmoid** | Xavier |
| **ReLU** | He/Kaiming |
| **SELU** | LeCun |

---

⬅️ [Back: Training Techniques](../)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

