<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Scaling&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="../06_hot_topics/01_flash_attention/images/flash-attention.svg" width="100%">

*Caption: Scaling techniques enable training billion-parameter models. Key methods include mixed precision (FP16/BF16), distributed training (DDP/FSDP), and FlashAttention for memory-efficient attention computation.*

---

## 📐 Mathematical Foundations

### Chinchilla Scaling Laws
```
L(N, D) = E + A/N^α + B/D^β

Where:
• L = loss
• N = model parameters
• D = data tokens
• α ≈ 0.34, β ≈ 0.28

Optimal compute allocation:
N_opt ∝ C^0.5, D_opt ∝ C^0.5
```

### Memory Requirements
```
Model parameters: P bytes (FP16 = 2P)
Gradients: P bytes
Optimizer states (Adam): 2P bytes (m, v)
Activations: O(batch × seq × hidden)

Total for training: ~16P (FP32 + Adam)
With mixed precision: ~8-12P
```

### FlashAttention I/O Complexity
```
Standard Attention: O(N² d) memory
FlashAttention: O(N) memory (uses tiling)

Key insight:
softmax(QKᵀ) V computed in blocks
avoiding N² intermediate storage
```

### Gradient Accumulation
```
Effective batch size:
B_eff = B × accumulation_steps × num_gpus

Update:
θ ← θ - η (1/K) Σₖ ∇L(B_k)
```

---

## 📂 Topics

| Folder | Topic | Key Idea |
|--------|-------|----------|
| [distributed/](./distributed/) | Multi-GPU training | DDP, FSDP |
| [mixed-precision/](./mixed-precision/) | FP16/BF16 | 2x faster |
| [efficient/](./efficient/) | Memory optimization | FlashAttention |

---

## 📊 Scaling Laws

```
Loss ∝ N^(-α) · D^(-β) · C^(-γ)

Where:
• N = model parameters
• D = dataset size
• C = compute (FLOPs)

More compute → better models (predictably!)
```

---

## 🔑 Techniques

| Technique | Saves | Trade-off |
|-----------|-------|-----------|
| Mixed precision | 2x memory | Slight accuracy risk |
| Gradient checkpointing | Memory | 30% slower |
| FlashAttention | Memory | None! |
| LoRA | Parameters | Slight accuracy |

---

## 🔗 Where This Topic Is Used

| Topic | How Scaling Is Used |
|-------|---------------------|
| **GPT-4** | Distributed training, mixed precision |
| **LLaMA 70B** | FSDP, tensor parallelism |
| **Stable Diffusion** | Mixed precision, FlashAttention |
| **Fine-tuning LLMs** | LoRA, QLoRA, gradient checkpointing |
| **BERT pretraining** | LAMB optimizer for large batch |
| **ViT Large** | Distributed data parallel |
| **Mixtral** | Expert parallelism + FSDP |

### Techniques Used In

| Technique | Used By |
|-----------|---------|
| **FlashAttention** | LLaMA 2, GPT-4, Claude |
| **LoRA** | Fine-tuning any LLM |
| **QLoRA** | Fine-tuning on consumer GPUs |
| **DeepSpeed** | Large model training |
| **Megatron-LM** | NVIDIA's large model training |
| **FSDP** | PyTorch native sharding |

### Scaling Dependencies

```
Scaling --> Training models > 1B params
       --> Long context (128K+ tokens)
       --> Affordable fine-tuning
       --> Research on limited hardware
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Scaling Laws | [arXiv](https://arxiv.org/abs/2001.08361) |
| 📄 | Flash Attention 2 | [arXiv](https://arxiv.org/abs/2307.08691) |
| 📖 | DeepSpeed | [Docs](https://www.deepspeed.ai/) |
| 📖 | Megatron-LM | [GitHub](https://github.com/NVIDIA/Megatron-LM) |
| 🇨🇳 | 大模型训练技术 | [知乎](https://zhuanlan.zhihu.com/p/343951042) |
| 🇨🇳 | Scaling Laws解读 | [CSDN](https://blog.csdn.net/v_JULY_v/article/details/134546262) |
| 🇨🇳 | 模型并行策略 | [B站](https://www.bilibili.com/video/BV1J94y1f7u5) |

---

⬅️ [Back: Training](../04_training/README.md) | ➡️ [Next: Hot Topics](../06_hot_topics/README.md)

---

⬅️ [Back: Deep Learning](../README.md)

---

⬅️ [Back: Training](../04_training/README.md) | ➡️ [Next: Hot Topics](../06_hot_topics/README.md)

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
