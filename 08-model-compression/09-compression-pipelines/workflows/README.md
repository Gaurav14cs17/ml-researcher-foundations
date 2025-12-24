# 🔄 Compression Workflows

> **Combining techniques for maximum compression**

<img src="./images/workflows-visual.svg" width="100%">

---

## 📐 Mathematical Foundations

### Combined Compression Ratio
```
CR_total = CR₁ × CR₂ × ... × CRₙ

Prune + Quantize:
CR = (1/sparsity) × (32/quantize_bits)
   = (1/0.5) × (32/8) = 2 × 4 = 8x
```

### Distillation + Quantize
```
Student model: S params (smaller)
Teacher model: T params

After distillation: S params
After INT8: S × (8/32) = S/4 effective

Total: T → S/4 (can be 100x+ reduction)
```

### QLoRA Memory
```
Base (4-bit): P × 0.5 bytes
LoRA: 2 × r × d × 2 bytes (FP16)

Total: 0.5P + 4rd bytes
For 7B model, r=16: ~3.5 GB
```

---

## 📂 Topics

| File | Topic | Techniques |
|------|-------|------------|

---

## 🎯 Common Pipelines

### Pipeline 1: Prune → Quantize

```
Original Model (100%)
       |
       v
+--------------+
|   Pruning    |  Remove 50% weights
+--------------+
       |
       v
+--------------+
|  Fine-tune   |  Recover accuracy
+--------------+
       |
       v
+--------------+
| Quantization |  FP32 → INT8
+--------------+
       |
       v
Compressed Model (12.5% size, 4-8x speedup)
```

### Pipeline 2: Distill → Quantize

```
Teacher Model (Large)
       |
       v
+--------------+
| Distillation |  Train smaller student
+--------------+
       |
       v
Student Model (Small)
       |
       v
+--------------+
| Quantization |  INT8
+--------------+
       |
       v
Production Model (10-100x smaller!)
```

---

## 🔥 QLoRA Pipeline (LLMs)

```
Pretrained LLM
       |
       v
+------------------+
| 4-bit Quantization|  bitsandbytes
+------------------+
       |
       v
+------------------+
|   Add LoRA       |  Tiny trainable adapters
+------------------+
       |
       v
+------------------+
|   Train LoRA     |  Your data
+------------------+
       |
       v
+------------------+
| Merge (optional) |  W = W_base + BA
+------------------+
       |
       v
Fine-tuned 4-bit Model
```

---

## 🔗 Where This Topic Is Used

| Topic | Pipeline Used |
|-------|--------------|
| **All LLM fine-tuning** | QLoRA pipeline |
| **Mobile deployment** | Prune → Quantize → Export |
| **Creating DistilBERT** | Distillation pipeline |
| **Production systems** | Full compression pipeline |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Deep Compression | [arXiv](https://arxiv.org/abs/1510.00149) |
| 📄 | QLoRA | [arXiv](https://arxiv.org/abs/2305.14314) |
| 📖 | HF Optimum | [Docs](https://huggingface.co/docs/optimum) |
| 🇨🇳 | 压缩流水线 | [知乎](https://zhuanlan.zhihu.com/p/628120082) |

---

⬅️ [Back: Deployment](../deployment/)
