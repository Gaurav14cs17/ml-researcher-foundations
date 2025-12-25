<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Weight%20Initialization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

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

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
