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

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
