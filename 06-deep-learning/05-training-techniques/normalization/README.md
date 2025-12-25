<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Normalization&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# 📊 Normalization Techniques

> **Stabilizing training through normalization**

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

⬅️ [Back: Training Techniques](../)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

