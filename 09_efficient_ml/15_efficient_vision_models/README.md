<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=120&section=header&text=Lecture%2015%20Efficient%20Vision%20Models&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-09-1ABC9C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

# Lecture 15: Efficient Vision Models

[← Back to Course](../) | [← Previous](../14_distributed_training/) | [Next: Efficient LLMs →](../16_efficient_llms/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09_efficient_ml/15_efficient_vision_models/demo.ipynb) ← **Try the code!**

---

## 📺 Video Lecture

| Resource | Link |
|----------|------|
| 🎥 **Lecture Video** | [Watch on YouTube](https://www.youtube.com/watch?v=v5CgSOL4GlM&list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&index=15) |
| 📊 **Slides** | [MIT Course Page](https://hanlab.mit.edu/courses/2023-fall-65940) |
| ⏱️ **Duration** | ~90 minutes |

### 📝 Video Summary

This lecture covers **efficient architectures for computer vision**:

- **Depthwise separable convolutions**: The key to efficient CNNs

- **MobileNet family**: V1, V2, V3 evolution

- **EfficientNet**: Compound scaling of depth, width, resolution

- **Vision Transformers**: ViT, Swin, and efficiency challenges

- **ConvNeXt**: Modernizing CNNs to match ViT performance

- **Mobile deployment**: Practical tips for edge vision

> 💡 *"Depthwise separable convolution reduces FLOPs by 8-9× with minimal accuracy loss—it's the foundation of efficient CNNs."* — Prof. Song Han

---

![Overview](overview.png)

## Evolution of Efficient CNNs

```
LeNet (1998) → AlexNet (2012) → VGG (2014) → ResNet (2015)
     ↓
MobileNet (2017) → EfficientNet (2019) → ConvNeXt (2022)

```

---

## 📐 Mathematical Foundations & Proofs

### Depthwise Separable Convolution

**Standard convolution:**

$$Y = X * K$$

where \( K \in \mathbb{R}^{C_{out} \times C_{in} \times k \times k} \).

**FLOPs:**

$$\text{FLOPs}_{std} = C_{in} \times C_{out} \times k^2 \times H \times W$$

**Depthwise separable:**

1. **Depthwise:** \( K_{dw} \in \mathbb{R}^{C_{in} \times 1 \times k \times k} \)

2. **Pointwise:** \( K_{pw} \in \mathbb{R}^{C_{out} \times C_{in} \times 1 \times 1} \)

$$Y = (X *_{dw} K_{dw}) *_{pw} K_{pw}$$

**FLOPs:**

$$\text{FLOPs}_{dw+pw} = C_{in} \times k^2 \times H \times W + C_{in} \times C_{out} \times H \times W$$

**Speedup ratio:**

$$\frac{\text{FLOPs}_{std}}{\text{FLOPs}_{dw+pw}} = \frac{C_{in} \times C_{out} \times k^2}{C_{in} \times k^2 + C_{in} \times C_{out}} = \frac{C_{out} \times k^2}{k^2 + C_{out}}$$

For \( C_{out} = 256, k = 3 \): Speedup ≈ 8.5×.

---

### Inverted Residual Block (MobileNetV2)

**Standard residual:**

```
x → wide → narrow → wide → + x

```

**Inverted residual:**

```
x → narrow → wide → narrow → + x

```

**Rationale:** Compute in high-dimensional space (expressiveness), but store low-dimensional (memory efficiency).

**Memory analysis:**

$$M_{inverted} = C + eC = (1+e)C
M_{standard} = C + C/e = C(1 + 1/e)$$

For expansion e=6: Inverted stores 7C, standard stores 1.17C per block.

But inverted has skip at low dimension → overall memory efficient.

---

### Linear Bottleneck

**Observation:** ReLU destroys information in low-dimensional spaces.

**Proof sketch:**
For \( x \in \mathbb{R}^d \), ReLU zeros out ~50% of dimensions on average.
If \( d \) is small, this loses significant information.

**Solution:** Remove ReLU after last pointwise conv (before addition).

$$Y = X + \text{Linear}(\text{ReLU}(\text{DW}(\text{ReLU}(\text{Expand}(X)))))$$

---

### Squeeze-and-Excitation (SE)

**Channel attention:**

$$s = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot \text{GAP}(X)))
Y = s \odot X$$

where GAP = Global Average Pooling.

**Complexity:**

$$\text{FLOPs}_{SE} = 2 \times C \times C/r$$

where \( r \) is reduction ratio (typically 4-16).

**Negligible overhead** (~2% FLOPs increase, ~1% accuracy gain).

---

### EfficientNet Compound Scaling

**Single-dimension scaling problems:**
- Depth only: Vanishing gradients

- Width only: Diminishing returns

- Resolution only: Diminishing returns

**Compound scaling:**

$$d = \alpha^\phi, \quad w = \beta^\phi, \quad r = \gamma^\phi$$

**FLOP constraint:**

$$\text{FLOPs} \propto d \cdot w^2 \cdot r^2 = \alpha \cdot \beta^2 \cdot \gamma^2 = 2$$

**Grid search:** \( \alpha = 1.2, \beta = 1.1, \gamma = 1.15 \)

**Proof of optimality:**

For fixed compute, optimizing accuracy:

$$\max_{d,w,r} \text{Acc}(d,w,r) \quad \text{s.t.} \quad d \cdot w^2 \cdot r^2 = C$$

Empirically, joint scaling outperforms single-dimension scaling.

---

### Vision Transformer Complexity

**Patch embedding:**

$$\text{Patches} = \frac{H \times W}{P^2}$$

For 224×224 image with 16×16 patches: 196 tokens.

**Attention complexity:**

$$\text{FLOPs}_{attn} = O(N^2 \cdot d) = O\left(\frac{H^2 W^2}{P^4} \cdot d\right)$$

**Quadratic in image size!**

---

### Swin Transformer: Window Attention

**Local window attention:**

$$A_{ij} = 0 \text{ if } i,j \text{ not in same window}$$

**Complexity:**

$$\text{FLOPs}_{swin} = O\left(\frac{HW}{M^2} \cdot M^4 \cdot d\right) = O(HW \cdot M^2 \cdot d)$$

**Linear in image size** for fixed window size \( M \)!

**Shifted window:** Alternating shifts enable cross-window information flow.

---

## 🧮 Key Derivations

### ConvNeXt Modernization

| Change | From | To | Gain |
|--------|------|-----|------|
| Training | 90 epochs | 300 + augmentation | +2.7% |
| Stem | 7×7 stride 2 | 4×4 stride 4 (patchify) | +0.1% |
| Ratio | 3:4:6:3 | 3:3:9:3 | +0.1% |
| Kernel | 3×3 | 7×7 depthwise | +0.4% |
| Activation | ReLU | GELU | +0.1% |
| Norm | BatchNorm | LayerNorm | +0.1% |

**Total:** 78.8% → 82.0% (matches Swin-T!)

---

### Mobile Deployment Considerations

**Actual latency vs FLOPs:**

| Operation | FLOPs | Mobile Latency |
|-----------|-------|---------------|
| Conv 3×3 | High | Medium |
| Depthwise | Low | Low |
| Attention | High | Very High |
| SE | Low | High (memory) |

**Key insight:** Memory access patterns matter more than FLOPs on mobile.

---

## 🎯 Where Used

| Concept | Applications |
|---------|-------------|
| MobileNet | Mobile classification |
| EfficientNet | Cloud and edge |
| Swin Transformer | High-accuracy vision |
| ConvNeXt | General-purpose vision |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [← Distributed Training](../14_distributed_training/README.md) | [Efficient ML](../README.md) | [Efficient LLMs →](../16_efficient_llms/README.md) |

---
## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | MobileNetV1 | [arXiv](https://arxiv.org/abs/1704.04861) |
| 📄 | MobileNetV2 | [arXiv](https://arxiv.org/abs/1801.04381) |
| 📄 | EfficientNet | [arXiv](https://arxiv.org/abs/1905.11946) |
| 📄 | Swin Transformer | [arXiv](https://arxiv.org/abs/2103.14030) |
| 📄 | ConvNeXt | [arXiv](https://arxiv.org/abs/2201.03545) |
| 🎥 | MIT 6.5940 TinyML | [Course](https://hanlab.mit.edu/courses/2024-fall-65940) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=80&section=footer" width="100%"/>
</p>
