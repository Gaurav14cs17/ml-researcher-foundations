# 📚 Introduction to Model Compression

> **Why smaller models matter**

## 🎯 Visual Overview

<img src="./images/introduction.svg" width="100%">

*Caption: Model compression addresses the problem of huge AI models (GPT-4: 1.8T params) by reducing size while maintaining accuracy for edge deployment, lower costs, and faster inference.*

---

## 📐 Mathematical Foundations

### Compression Ratio
```
Compression ratio: CR = Original Size / Compressed Size

Example:
Original: 340M params × 4 bytes = 1.36 GB
Compressed (INT8): 340M × 1 byte = 340 MB
CR = 1360 / 340 = 4x
```

### Compression-Accuracy Trade-off
```
Goal: min Size(M_c) s.t. |Acc(M_c) - Acc(M)| ≤ ε

Pareto frontier:
No compression can improve both size AND accuracy
```

### Memory and Compute
```
Memory: O(P × bytes_per_param)
Compute: O(FLOPs × ops_per_param)

FP32: 4 bytes, 1 op
FP16: 2 bytes, 2x tensor core speedup
INT8: 1 byte, 4x compute
INT4: 0.5 bytes, 8x compute (theoretical)
```

---

## 📂 Topics

| File | Topic | Key Concepts |
|------|-------|--------------|

---

## 🎯 The Problem

```
Modern AI models are HUGE:

GPT-4:        ~1.8 Trillion parameters
LLaMA-70B:    70 Billion parameters → 140 GB
Stable Diff:  1 Billion parameters → 4 GB
BERT-Large:   340 Million parameters → 1.3 GB

Problems:
• Won't fit in GPU memory
• Too slow for real-time
• Too expensive to serve
• Can't run on mobile/edge
```

---

## 💡 The Solution: Compression

```
Original Model          Compressed Model
+----------------+      +----------------+
|                |      |                |
|   340M params  | -->  |   66M params   |  (DistilBERT)
|   1.3 GB       |      |   260 MB       |
|   100ms        |      |   30ms         |
|                |      |                |
+----------------+      +----------------+
    BERT-Large              DistilBERT

Same task, 5x smaller, 3x faster, <1% accuracy drop!
```

---

## ⚖️ Trade-offs Visualization

<img src="./images/tradeoffs.svg" width="100%">

---

## 🔗 Where This Topic Is Used

| Topic | How Introduction Concepts Apply |
|-------|--------------------------------|
| **Production Systems** | Understand cost-accuracy tradeoffs |
| **Mobile Deployment** | Know why compression is needed |
| **LLM Serving** | Quantization reduces serving cost |
| **Fine-tuning** | LoRA makes fine-tuning affordable |

---

## 📚 References & Resources

### 📄 Survey Papers

| Type | Title | Link |
|------|-------|------|
| 📄 | Model Compression Survey | [arXiv](https://arxiv.org/abs/2103.13630) |
| 📄 | Efficient Deep Learning | [arXiv](https://arxiv.org/abs/2106.08962) |
| 📄 | LLM Compression Survey | [arXiv](https://arxiv.org/abs/2308.07633) |
| 🇨🇳 | 为什么需要模型压缩 | [知乎](https://zhuanlan.zhihu.com/p/628120082) |
| 🇨🇳 | 大模型部署指南 | [机器之心](https://www.jiqizhixin.com/articles/2023-08-31-2) |

---

⬅️ [Back: Model Compression](../) | ➡️ [Next: 02-Parameter Reduction](../02-parameter-reduction/)

