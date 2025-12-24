<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=02 Parameter Reduction&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# ✂️ Parameter Reduction Techniques

> **Reducing the number of parameters in a model**

## 🎯 Visual Overview

<img src="./images/parameter-reduction.svg" width="100%">

*Caption: Parameter reduction includes pruning (remove weights), weight sharing (tie parameters), and clustering (group similar weights). Often combined: Prune → Cluster → Quantize.*

---

## 📐 Mathematical Foundations

### Pruning Criterion
```
Magnitude pruning:
Keep weights where |wᵢⱼ| > threshold τ
W_pruned = W ⊙ M  (element-wise mask)

Sensitivity: S(w) = |∂L/∂w|
Prune weights with low |w × ∂L/∂w|
```

### Structured Pruning
```
Filter pruning in CNN:
Remove entire filters: W[:, i, :, :] = 0

Channel pruning:
Remove channels: activations[:, i] = 0

Reduces actual compute (not just memory)
```

### Lottery Ticket Hypothesis
```
Theorem (Frankle & Carlin, 2019):
Dense network contains sparse subnetwork ("winning ticket")
that can match full network accuracy when trained in isolation

Finding: Prune → Reset weights → Retrain
```

### Weight Sharing
```
Weight tying (Transformers):
W_embedding = W_output.T

Reduces params from 2×vocab×d to 1×vocab×d
```

---

## 📂 Topics

| Folder | Topic | Key Concepts |
|--------|-------|--------------|
| [pruning/](./pruning/) | Remove weights | Unstructured, structured |
| [weight-sharing/](./weight-sharing/) | Share parameters | Hash-based, tied |
| [weight-clustering/](./weight-clustering/) | Cluster weights | K-means, entropy coding |

---

## 🎯 Core Idea

```
Original Network:
+-------------------------------------+
|  W₁[1000×1000] → W₂[1000×1000]     |
|  = 2,000,000 parameters             |
+-------------------------------------+

After Pruning (50%):
+-------------------------------------+
|  W₁[sparse] → W₂[sparse]           |
|  = 1,000,000 non-zero parameters    |
+-------------------------------------+

After Weight Sharing:
+-------------------------------------+
|  W₁ = W₂ (tied)                    |
|  = 1,000,000 unique parameters      |
+-------------------------------------+
```

---

## 📊 Comparison

| Technique | Compression | Speed Gain | Accuracy | Hardware |
|-----------|-------------|------------|----------|----------|
| **Unstructured Pruning** | 2-10x | 1x (dense HW) | Good | Sparse HW |
| **Structured Pruning** | 2-4x | 2-4x | Moderate | All |
| **Weight Sharing** | 2-4x | 1x | Good | All |
| **Clustering** | 2-4x | 1x | Good | All |

---

## 🔗 Where This Topic Is Used

| Topic | How Parameter Reduction Is Used |
|-------|--------------------------------|
| **Lottery Ticket Hypothesis** | Find sparse subnetworks via pruning |
| **BERT Pruning** | Remove attention heads |
| **Neural Architecture Search** | Prune during search |
| **Mobile Models** | Structured pruning for efficiency |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Deep Compression | [arXiv](https://arxiv.org/abs/1510.00149) |
| 📄 | Lottery Ticket | [arXiv](https://arxiv.org/abs/1803.03635) |
| 📄 | ALBERT | [arXiv](https://arxiv.org/abs/1909.11942) |
| 🇨🇳 | 参数压缩技术 | [知乎](https://zhuanlan.zhihu.com/p/628120082) |
| 🇨🇳 | 剪枝技术详解 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/125689573) |
| 🇨🇳 | 模型压缩入门 | [B站](https://www.bilibili.com/video/BV1J94y1f7u5) |

---

⬅️ [Back: 01-Introduction](../01-introduction/) | ➡️ [Next: 03-Quantization](../03-quantization/)


---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
