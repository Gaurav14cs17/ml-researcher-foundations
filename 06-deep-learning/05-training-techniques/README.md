<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Training%20Techniques&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📂 Topics

| Folder | Technique | Purpose |
|--------|-----------|---------|
| [gradient-clipping/](./gradient-clipping/) | Gradient Clipping | Prevent exploding gradients |
| [initialization/](./initialization/) | Weight Initialization | Stable starting point |
| [normalization/](./normalization/) | Batch/Layer Norm | Stable training |
| [regularization/](./regularization/) | Dropout | Prevent overfitting |

---

## 📐 Key Techniques

### Gradient Clipping
```
Clip by value: g = clip(g, -threshold, threshold)
Clip by norm: g = g × min(1, threshold/||g||)

Prevents exploding gradients in RNNs and Transformers
```

### Normalization
```
Batch Norm: Normalize across batch dimension
Layer Norm: Normalize across feature dimension
Group Norm: Normalize across groups of channels

Stabilizes training, allows higher learning rates
```

### Dropout
```
During training: Randomly zero p% of activations
During inference: Scale by (1-p) or no scaling

Acts as ensemble regularization
```

---

## 🔗 Where Techniques Are Used

| Technique | Application |
|-----------|-------------|
| **Gradient Clipping** | LLM training, RNNs |
| **Batch Norm** | CNNs, ResNets |
| **Layer Norm** | Transformers |
| **Dropout** | Most neural networks |
| **Xavier/He Init** | All networks |

---

⬅️ [Back: 05-Scaling](../05-scaling/) | ➡️ [Next: 06-Hot Topics](../06-hot-topics/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
