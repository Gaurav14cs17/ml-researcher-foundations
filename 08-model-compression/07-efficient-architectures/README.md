# 🏗️ Efficient Network Architectures

> **Designed for efficiency from the start**

<img src="./images/efficient-arch.svg" width="100%">

---

## 📐 Mathematical Foundations

### Depthwise Separable Convolution
```
Standard conv: O(K² × Cᵢₙ × Cₒᵤₜ × H × W)
Depthwise + Pointwise: O(K² × Cᵢₙ × H × W + Cᵢₙ × Cₒᵤₜ × H × W)

Reduction ratio: 1/Cₒᵤₜ + 1/K²
For K=3, Cₒᵤₜ=256: ~8-9x reduction
```

### Linear Attention
```
Standard: Attention = softmax(QKᵀ/√d)V  O(n²d)

Linear: Attention = φ(Q)(φ(K)ᵀV)  O(nd²)

Key insight: Compute (K^T V) first → O(d²)
Then Q × (K^T V) → O(nd²)
```

### EfficientNet Compound Scaling
```
depth: d = α^φ
width: w = β^φ
resolution: r = γ^φ

Constraint: α × β² × γ² ≈ 2 (FLOPS double)

φ controls overall scale
```

---

## 📂 Topics

| Folder | Topic | Key Concepts |
|--------|-------|--------------|
| [efficient-networks/](./efficient-networks/) | Efficient CNNs | MobileNet, EfficientNet |
| [efficient-transformers/](./efficient-transformers/) | Efficient attention | Linformer, Performer |

---

## 🎯 Philosophy

```
Compression: Take large model → make smaller
             (post-hoc, may lose quality)

Efficient Architecture: Design small but powerful
                        (native efficiency)

Examples:
+----------------------------------------------------------+
| Task: Image Classification                                |
+----------------------------------------------------------+
| ResNet-50:     25M params,  4 GFLOPs                     |
| MobileNetV3:   5.4M params, 0.2 GFLOPs  (similar acc!)   |
|                                                          |
| 5x smaller, 20x less compute, ~same accuracy            |
+----------------------------------------------------------+
```

---

## 📊 Key Architectures

| Architecture | Innovation | Size | Use Case |
|--------------|------------|------|----------|
| **MobileNet** | Depthwise separable conv | 3-5M | Mobile vision |
| **EfficientNet** | Compound scaling | 5-66M | Vision (scalable) |
| **ALBERT** | Parameter sharing | 12M | NLP |
| **Linformer** | O(n) attention | Varies | Long sequences |
| **Performer** | Kernel attention | Varies | Very long sequences |

---

## 🔥 Key Innovations

### Depthwise Separable Convolution (MobileNet)

```
Standard Conv: (H×W×C_in) → (H×W×C_out)
Parameters: K² × C_in × C_out
If K=3, C_in=256, C_out=256: 589,824 params

Depthwise Separable:
1. Depthwise: (H×W×C_in) → (H×W×C_in)   [K² × C_in]
2. Pointwise: (H×W×C_in) → (H×W×C_out)  [C_in × C_out]
Parameters: K² × C_in + C_in × C_out
If K=3, C_in=256, C_out=256: 67,840 params (8.7x less!)
```

### Linear Attention (Linformer, Performer)

```
Standard Attention: O(n²)
softmax(QK^T / √d) · V

Problem: n=8192 → 67M operations per head!

Linear Attention: O(n)
φ(Q)(φ(K)^T V)  where φ is a feature map

Now: Can handle n=100K+ tokens!
```

---

## 🔗 Where This Topic Is Used

| Topic | How Efficient Architectures Are Used |
|-------|-------------------------------------|
| **Mobile Apps** | MobileNet for on-device vision |
| **Edge Devices** | EfficientNet for IoT |
| **Long Documents** | Linformer, Longformer |
| **Genomics** | Long-sequence models |
| **ALBERT** | Smaller BERT variant |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | MobileNets | [arXiv](https://arxiv.org/abs/1704.04861) |
| 📄 | EfficientNet | [arXiv](https://arxiv.org/abs/1905.11946) |
| 📄 | Linformer | [arXiv](https://arxiv.org/abs/2006.04768) |
| 🇨🇳 | MobileNet详解 | [知乎](https://zhuanlan.zhihu.com/p/70703846) |
| 🇨🇳 | 高效架构对比 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | 轻量化网络 | [B站](https://www.bilibili.com/video/BV1J94y1f7u5) |

---

⬅️ [Back: 06-Sparsity](../06-sparsity/) | ➡️ [Next: 08-PEFT](../08-peft/)
