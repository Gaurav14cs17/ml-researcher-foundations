<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Batch Normalization&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# 📊 Batch Normalization

> **Normalizing activations for faster training**

---

## 📐 Formula

```
BatchNorm:
  μ_B = (1/m) Σᵢ xᵢ
  σ²_B = (1/m) Σᵢ (xᵢ - μ_B)²
  x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)
  yᵢ = γ x̂ᵢ + β

Where γ, β are learnable parameters
```

---

## 💻 Code

```python
# For CNNs
bn = nn.BatchNorm2d(64)

# Training vs inference
model.train()  # Use batch statistics
model.eval()   # Use running statistics
```

---

## 🔗 Benefits

- Allows higher learning rates
- Reduces sensitivity to initialization
- Has regularization effect

---

⬅️ [Back: Normalization](../)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

