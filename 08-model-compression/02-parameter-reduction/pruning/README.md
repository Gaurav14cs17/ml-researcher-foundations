# ✂️ Pruning

> **Removing unnecessary weights from neural networks**

<img src="./images/pruning-visual.svg" width="100%">

---

## 📂 Topics

| File | Topic | Key Concepts |
|------|-------|--------------|

---

## 🎯 The Core Idea

```
Unstructured Pruning:
+-------------------------------------+
|  Before:  [1.2, 0.01, -0.8, 0.002] |
|  After:   [1.2, 0,    -0.8, 0    ] |  (zeros where small)
+-------------------------------------+

Structured Pruning:
+-------------------------------------+
|  Before:  Conv with 64 filters     |
|  After:   Conv with 48 filters     |  (remove entire filters)
+-------------------------------------+
```

---

## 📐 Magnitude Pruning

```python
import torch

def magnitude_prune(weights, sparsity=0.5):
    """Prune smallest magnitude weights"""
    threshold = torch.quantile(weights.abs(), sparsity)
    mask = weights.abs() > threshold
    return weights * mask

# Example
W = torch.randn(100, 100)
W_pruned = magnitude_prune(W, sparsity=0.9)  # 90% zeros
```

---

## 📐 Mathematical Foundations

<img src="./images/pruning-math.svg" width="100%">

---

## 🔗 Where This Topic Is Used

| Topic | How Pruning Is Used |
|-------|---------------------|
| **Lottery Ticket Hypothesis** | Find winning tickets via pruning |
| **BERT Compression** | Prune attention heads |
| **Vision Models** | Filter pruning in CNNs |
| **Sparse Training** | Train with pruning from start |
| **Mobile Deployment** | Reduce model size |

### Prerequisite For

```
Pruning --> Sparse neural networks
       --> Hardware-aware compression
       --> Neural architecture search
       --> Lottery ticket research
```

---

## 📚 References & Resources

### 📄 Key Papers

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|------------------|
| [Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635) | Frankle & Carlin | 2018 | Sparse subnetworks from init |
| [Deep Compression](https://arxiv.org/abs/1510.00149) | Han et al. | 2015 | Prune+Quantize+Huffman |
| [Learning Structured Sparsity](https://arxiv.org/abs/1608.03665) | Wen et al. | 2016 | SSL for CNNs |
| [Movement Pruning](https://arxiv.org/abs/2005.07683) | Sanh et al. | 2020 | Pruning during fine-tuning |
| [SNIP](https://arxiv.org/abs/1810.02340) | Lee et al. | 2018 | Single-shot pruning at init |
| [Wanda](https://arxiv.org/abs/2306.11695) | Sun et al. | 2023 | Pruning LLMs without retraining |
| [SparseGPT](https://arxiv.org/abs/2301.00774) | Frantar & Alistarh | 2023 | One-shot 50% sparsity for GPT |
| 🇨🇳 神经网络剪枝综述 | [知乎](https://zhuanlan.zhihu.com/p/93876564) | - | 剪枝方法全面总结 |
| 🇨🇳 Lottery Ticket详解 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/125689573) | - | 彩票假设原理 |
| 🇨🇳 LLM剪枝技术 | [机器之心](https://www.jiqizhixin.com/articles/2023-02-10-4) | - | 大模型剪枝实践 |

### 🎓 Courses

| Course | Description | Link |
|--------|-------------|------|
| 🔥 MIT 6.5940 | Prof. Song Han's TinyML: Lectures 3-4 Pruning & Sparsity | [Course](https://hanlab.mit.edu/courses/2024-fall-65940) |

### 🛠️ Tools

| Tool | Description | Link |
|------|-------------|------|
| torch.nn.utils.prune | PyTorch pruning | [Docs](https://pytorch.org/docs/stable/nn.html#utilities) |
| Neural Magic | Sparse inference | [GitHub](https://github.com/neuralmagic) |

---

⬅️ [Back: Parameter Reduction](../) | ➡️ [Next: Weight Clustering](../weight-clustering/)

