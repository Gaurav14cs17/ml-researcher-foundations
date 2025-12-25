<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Batch%20Normalization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Algorithm

```
For mini-batch B:
    μ_B = (1/m) Σᵢ xᵢ           # Batch mean
    σ²_B = (1/m) Σᵢ (xᵢ - μ_B)²  # Batch variance
    x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)  # Normalize
    yᵢ = γx̂ᵢ + β                 # Scale & shift (learnable)
```

---

## 🔑 Why It Works

```
1. Reduces internal covariate shift
2. Allows higher learning rates
3. Acts as regularization (batch noise)
4. Reduces sensitivity to initialization
```

---

## ⚠️ Issues

| Issue | Solution |
|-------|----------|
| Small batches | Group Norm |
| RNNs | Layer Norm |
| Test time | Running averages |
| Distributed | Synced BN |

---

## 💻 Code

```python
import torch.nn as nn

# In a CNN
model = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.BatchNorm2d(64),  # After conv, before activation
    nn.ReLU(),
)

# Training vs Eval mode matters!
model.train()  # Use batch statistics
model.eval()   # Use running averages

# PyTorch tracks running mean/var automatically
bn = nn.BatchNorm2d(64)
print(bn.running_mean.shape)  # (64,)
print(bn.running_var.shape)   # (64,)
```

---

## 📊 Normalization Comparison

| Type | Normalizes Over | Use Case |
|------|-----------------|----------|
| Batch Norm | (N, H, W) | CNNs |
| Layer Norm | (C, H, W) | Transformers |
| Instance Norm | (H, W) | Style transfer |
| Group Norm | (G, H, W) | Small batch |

---

---

➡️ [Next: Layer Norm](./layer-norm.md)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
