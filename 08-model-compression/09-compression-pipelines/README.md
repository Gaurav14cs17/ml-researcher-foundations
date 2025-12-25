<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=120&section=header&text=Compression%20Pipelines&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-08-E74C3C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Mathematical Foundations

### Combined Compression
```
Total compression:
CR_total = CR_prune × CR_quant × CR_distill

Example:
2x (pruning) × 4x (INT8) × 2x (distill) = 16x compression
```

### Accuracy-Compression Trade-off
```
Δ_acc(compress) ≈ α × CR^β

Larger compression → more accuracy loss
β depends on method (quantization < pruning typically)
```

### QLoRA Memory Savings
```
Base model (4-bit): P × 0.5 bytes
LoRA adapters (FP16): r × d × 2 × 2 bytes
Total: ~P/2 + 4rd bytes

Example: 7B model, r=16, d=4096
= 3.5 GB + 0.5 MB = ~3.5 GB (vs 14 GB FP16!)
```

---

## 📂 Topics

| Folder | Topic | Key Concepts |
|--------|-------|--------------|
| [workflows/](./workflows/) | Combined pipelines | Prune+Quantize, Distill |
| [deployment/](./deployment/) | Deployment | Edge, cloud |

---

## 🎯 Typical Pipeline

```
+-------------------------------------------------------------+
|               Model Compression Pipeline                     |
+-------------------------------------------------------------+
|                                                              |
|  1. Train/Get Base Model                                     |
|         |                                                    |
|      ▼      |
|  2. (Optional) Knowledge Distillation                        |
|         |                                                    |
|      ▼      |
|  3. Pruning (remove redundant weights)                       |
|         |                                                    |
|      ▼      |
|  4. Fine-tune (recover accuracy)                             |
|         |                                                    |
|      ▼      |
|  5. Quantization (reduce precision)                          |
|         |                                                    |
|      ▼      |
|  6. Export (ONNX, TensorRT, etc.)                           |
|         |                                                    |
|      ▼      |
|  7. Deploy!                                                  |
|                                                              |
+-------------------------------------------------------------+
```

---

## 📊 Common Workflows

| Workflow | When to Use | Example |
|----------|-------------|---------|
| **Quantize Only** | Quick wins | INT8 for production |
| **Prune + Quantize** | Maximum compression | Mobile deployment |
| **Distill + Quantize** | New small model | DistilBERT |
| **QLoRA** | Fine-tuning LLMs | All LLM fine-tuning |

---

## 🔥 QLoRA Pipeline (Most Common for LLMs)

```
1. Load base model in 4-bit
   +-- bitsandbytes quantization

2. Add LoRA adapters
   +-- Only train 0.1% params

3. Train on your data
   +-- 8 GB GPU is enough!

4. Merge adapters (optional)
   +-- W_new = W_base + BA

5. Deploy
   +-- Still 4-bit, fast inference
```

---

## 🔗 Where This Topic Is Used

| Topic | Pipeline Used |
|-------|--------------|
| **Production LLMs** | Quantize → TensorRT → Deploy |
| **LLM Fine-tuning** | QLoRA pipeline |
| **Mobile Vision** | Prune → Quantize → TFLite |
| **Edge Deployment** | Full pipeline + hardware export |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | QLoRA Paper | [arXiv](https://arxiv.org/abs/2305.14314) |
| 📄 | Deep Compression | [arXiv](https://arxiv.org/abs/1510.00149) |
| 📖 | HuggingFace PEFT | [Docs](https://huggingface.co/docs/peft) |
| 🇨🇳 | 模型压缩流程 | [知乎](https://zhuanlan.zhihu.com/p/628120082) |
| 🇨🇳 | QLoRA实践 | [CSDN](https://blog.csdn.net/qq_37006625/article/details/129405866) |

---

⬅️ [Back: 08-PEFT](../08-peft/) | ➡️ [Next: 10-Tools](../10-tools/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=80&section=footer" width="100%"/>
</p>
