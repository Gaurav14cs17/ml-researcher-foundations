<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Distributed%20Training&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/distributed.svg" width="100%">

*Caption: Distributed training scales to large models via Data Parallelism (same model, different data), Model Parallelism (split model), and modern techniques like FSDP/ZeRO that shard optimizer states.*

---

## 📐 Mathematical Foundations

### Gradient Aggregation
```
With N GPUs, each computing gradients on batch Bᵢ:

All-reduce gradient:
∇L = (1/N) Σᵢ₌₁ᴺ ∇L(Bᵢ)

Equivalent to single GPU with batch size N × B
```

### Memory Analysis (per GPU)
```
Dense Training:
• Model params: P
• Gradients: P
• Optimizer (Adam): 2P  (momentum + variance)
• Total: 4P × bytes_per_param

FSDP/ZeRO Stage 3:
• Sharded across N GPUs
• Per GPU: 4P / N
• Enables training N× larger models!
```

### Communication Costs
```
All-reduce: 2(N-1)/N × P × sizeof(dtype)

Ring all-reduce:
• Time: 2(N-1) × P / (N × bandwidth)
• Nearly linear scaling for large N
```

---

## 📊 Strategies

| Strategy | What's Split | Use Case |
|----------|--------------|----------|
| **Data Parallel** | Data batches | Most common |
| **Model Parallel** | Model layers | Very large models |
| **Pipeline Parallel** | Sequential layers | GPipe |
| **FSDP** | Parameters + gradients | LLM training |

---

## 🔥 Data Parallel (DDP)

```
Each GPU has full model copy
Each GPU processes different batch
Gradients synchronized via all-reduce

torch.nn.parallel.DistributedDataParallel
```

---

## 🔥 FSDP

```
Parameters sharded across GPUs
Each GPU only holds 1/N of parameters
Gather before forward, release after

Enables training models larger than single GPU memory!
```

---

## 💻 Code

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize
dist.init_process_group("nccl")
model = DDP(model.cuda(), device_ids=[local_rank])

# Training is same as single GPU
loss = model(batch)
loss.backward()
optimizer.step()
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | PyTorch DDP | [Docs](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) |
| 📄 | FSDP Paper | [arXiv](https://arxiv.org/abs/2304.11277) |
| 📄 | ZeRO Paper | [arXiv](https://arxiv.org/abs/1910.02054) |
| 🇨🇳 | 分布式训练详解 | [知乎](https://zhuanlan.zhihu.com/p/343951042) |
| 🇨🇳 | DDP与FSDP对比 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/125689573) |
| 🇨🇳 | 大模型分布式训练 | [B站](https://www.bilibili.com/video/BV1J94y1f7u5) |

## 🔗 Where This Topic Is Used

| Strategy | Application |
|----------|------------|
| **Data Parallel** | Multi-GPU training |
| **Model Parallel** | Large models |
| **Pipeline Parallel** | LLM training |
| **ZeRO** | Memory efficient |

---

➡️ [Next: Efficient](../02_efficient/README.md)

---

⬅️ [Back: Scaling](../../README.md)

---

➡️ [Next: Efficient](../02_efficient/README.md)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
