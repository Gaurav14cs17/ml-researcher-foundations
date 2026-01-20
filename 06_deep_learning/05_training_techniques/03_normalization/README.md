<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Normalization%20Techniques&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Types

```
Batch Norm: Normalize over batch + spatial
  μ = mean(x, axis=[0,2,3])
  
Layer Norm: Normalize over features
  μ = mean(x, axis=[-1])
  
Instance Norm: Normalize per sample
Group Norm: Normalize over groups
```

---

## 💻 Code

```python
# Batch Norm (CNNs)
bn = nn.BatchNorm2d(64)

# Layer Norm (Transformers)
ln = nn.LayerNorm(768)

# Group Norm
gn = nn.GroupNorm(num_groups=32, num_channels=64)
```

---

## 🔗 Usage

| Norm | Architecture |
|------|--------------|
| **BatchNorm** | CNNs (ResNet) |
| **LayerNorm** | Transformers |
| **GroupNorm** | Small batches |
| **RMSNorm** | LLaMA |

---

⬅️ [Back: Initialization](../02_initialization/README.md) | ➡️ [Next: Regularization](../04_regularization/README.md)

---

⬅️ [Back: Training Techniques](../../README.md)

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
