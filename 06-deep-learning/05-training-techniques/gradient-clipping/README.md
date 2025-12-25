<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Gradient Clipping&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# ✂️ Gradient Clipping

> **Preventing exploding gradients**

---

## 📐 Methods

```
Clip by Value:
  g = clip(g, -threshold, threshold)
  
Clip by Norm:
  if ||g|| > threshold:
      g = g × (threshold / ||g||)
```

---

## 💻 Code

```python
# Clip by norm (most common)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

# Usage in training loop
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

---

## 🔗 When to Use

| Model | Clipping |
|-------|----------|
| **RNN/LSTM** | Essential |
| **Transformers** | Common (max_norm=1.0) |
| **CNNs** | Usually not needed |

---

⬅️ [Back: Training Techniques](../)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

