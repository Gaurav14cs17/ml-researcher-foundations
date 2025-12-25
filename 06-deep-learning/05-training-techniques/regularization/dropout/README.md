<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Dropout&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

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

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
