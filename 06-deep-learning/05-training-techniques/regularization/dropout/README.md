<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Dropout&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# 🎲 Dropout

> **Random deactivation for regularization**

---

## 📐 How It Works

```
Training:
  h' = h ⊙ mask, where mask ~ Bernoulli(1-p)
  Scale by 1/(1-p)

Inference:
  Use all units (no dropout)
  
Effect: Ensemble of 2^n subnetworks
```

---

## 💻 Code

```python
dropout = nn.Dropout(p=0.5)

# Only applies during training
model.train()
out = dropout(hidden)  # Random zeros

model.eval()
out = dropout(hidden)  # No effect
```

---

## 🔗 Variants

| Variant | Application |
|---------|-------------|
| **Dropout** | Fully connected |
| **Dropout2D** | CNNs (drop channels) |
| **DropPath** | Transformers |

---

⬅️ [Back: Regularization](../)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

