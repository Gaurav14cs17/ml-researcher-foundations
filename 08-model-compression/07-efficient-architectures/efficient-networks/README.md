<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=120&section=header&text=Efficient%20Network%20Architecture&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-08-E74C3C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Mathematical Foundations

### Depthwise Separable Convolution
```
Standard: O(K² × Cᵢₙ × Cₒᵤₜ × H × W)
Depthwise + Pointwise: O((K² + Cₒᵤₜ) × Cᵢₙ × H × W)

Reduction ratio: (K² + Cₒᵤₜ)/(K² × Cₒᵤₜ)
                ≈ 1/K² + 1/Cₒᵤₜ ≈ 1/9 for K=3
```

### EfficientNet Compound Scaling
```
depth: d = α^φ
width: w = β^φ  
resolution: r = γ^φ

Constraint: α × β² × γ² ≈ 2

φ = 0 → EfficientNet-B0
φ = 1 → EfficientNet-B1
...
φ = 7 → EfficientNet-B7
```

### Inverted Residual (MobileNetV2)
```
Narrow → Wide → Narrow

1. 1×1 conv (expand: d → td)
2. 3×3 depthwise conv
3. 1×1 conv (project: td → d)

t = expansion ratio (typically 6)
```

---

## 📂 Topics

| File | Topic | Key Innovation |
|------|-------|----------------|

---

## 🎯 MobileNet: Depthwise Separable Conv

```
Standard Convolution:
Input: H×W×C_in → Output: H×W×C_out
Params: K² × C_in × C_out
FLOPs: H × W × K² × C_in × C_out

Depthwise Separable (MobileNet):
1. Depthwise: H×W×C_in → H×W×C_in (one filter per channel)
   Params: K² × C_in
2. Pointwise: H×W×C_in → H×W×C_out (1×1 conv)
   Params: C_in × C_out

Total: K² × C_in + C_in × C_out
Savings: ~8-9x fewer params and FLOPs!
```

---

## 📊 Comparison

| Model | Params | Top-1 Acc | FLOPs |
|-------|--------|-----------|-------|
| ResNet-50 | 25M | 76.0% | 4B |
| MobileNetV3 | 5.4M | 75.2% | 219M |
| EfficientNet-B0 | 5.3M | 77.3% | 390M |

---

## 🔗 Where This Topic Is Used

| Topic | How Efficient Architectures Are Used |
|-------|-------------------------------------|
| **Mobile Apps** | MobileNet for on-device inference |
| **Edge Devices** | EfficientNet for IoT |
| **Real-time Vision** | Fast inference needed |
| **TensorFlow Lite** | Optimized for mobile |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | MobileNetV3 | [arXiv](https://arxiv.org/abs/1905.02244) |
| 📄 | EfficientNet | [arXiv](https://arxiv.org/abs/1905.11946) |
| 📄 | ALBERT | [arXiv](https://arxiv.org/abs/1909.11942) |
| 🇨🇳 | 轻量化网络 | [知乎](https://zhuanlan.zhihu.com/p/70703846) |

---

⬅️ [Back: Efficient Architectures](../) | ➡️ [Next: Efficient Transformers](../efficient-transformers/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=80&section=footer" width="100%"/>
</p>
