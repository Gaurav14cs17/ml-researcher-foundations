# 🔥 Hot Topics in Deep Learning

> **Cutting-edge techniques driving modern AI systems**

---

## 🎯 Overview

This section covers the latest breakthroughs and innovations that are reshaping deep learning - from efficient attention mechanisms enabling 100K+ token contexts, to parameter-efficient fine-tuning methods democratizing LLM adaptation.

---

## 📚 Topics Covered

### 1. **[Flash Attention](./flash-attention/)**
IO-aware exact attention algorithm that enables long-context models

**Key Innovation:** Never materialize the full n² attention matrix
- Memory: O(n²) → O(n)
- Speed: 2-7x faster
- Enables: 100K+ token sequences

### 2. **[LoRA (Low-Rank Adaptation)](./lora/)**
Efficient fine-tuning via low-rank decomposition

**Key Innovation:** ΔW = BA where r << d
- Parameters: 250x reduction
- Quality: Matches full fine-tuning
- Enables: Personal LLM adapters

---

## 🌍 Why These Matter

### Flash Attention
```
Problem: Standard attention O(n²) memory
         GPT-3: 2048 tokens max

Solution: Block-sparse computation in SRAM
         Modern LLMs: 100K+ tokens

Impact: Long documents, code, conversations
```

### LoRA
```
Problem: Fine-tuning 70B model = 280GB memory
         Storing 1000 adapters = 280TB

Solution: Low-rank updates = 200MB each
         1000 adapters = 200GB

Impact: Personalized AI, multi-task models
```

---

## 🔄 How They Relate

```
┌─────────────────────────────────────┐
│  Modern Efficient LLM Training      │
├─────────────────────────────────────┤
│                                     │
│  Flash Attention                    │
│  └─> Enables long context          │
│      training & inference           │
│                                     │
│  LoRA / QLoRA                       │
│  └─> Enables affordable             │
│      fine-tuning at scale           │
│                                     │
│  Together:                          │
│  Train large models on long docs    │
│  Adapt efficiently to tasks         │
└─────────────────────────────────────┘
```

---

## 📊 Comparison

| Technique | Problem Solved | Speedup | Memory Savings |
|-----------|----------------|---------|----------------|
| **Flash Attention** | Long sequences | 2-7x | O(n²) → O(n) |
| **LoRA** | Fine-tuning cost | ~1x | 99.6% fewer params |
| **QLoRA** | Both + quantization | ~1x | 99.7% + 4-bit base |

---

## 💻 Quick Start

### Flash Attention
```python
import torch.nn.functional as F

# PyTorch 2.0+ automatically uses Flash Attention
output = F.scaled_dot_product_attention(
    query, key, value,
    is_causal=True
)
```

### LoRA
```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(base_model, config)
```

---

## 🚀 Recent Extensions

### Flash Attention Family
- **Flash Attention 2** (2023): 5-7x faster
- **Flash Attention 3** (2024): Multi-query, sparse patterns
- **PagedAttention** (vLLM): KV cache optimization

### LoRA Family
- **QLoRA** (2023): 4-bit quantization + LoRA
- **AdaLoRA** (2023): Adaptive rank allocation
- **DoRA** (2024): Decomposed magnitude + direction

---

## 🌍 Real-World Impact

| Application | Technique | Result |
|-------------|-----------|--------|
| **ChatGPT** | Flash Attention | 32K token context |
| **Claude** | Flash + optimization | 200K tokens |
| **Llama adapters** | LoRA | 1000s of task-specific models |
| **Code completion** | Both | Long file context + personal style |

---

## 📚 Resources

### Papers
- **Flash Attention** (Dao et al., 2022)
- **Flash Attention 2** (Dao, 2023)
- **LoRA** (Hu et al., 2021)
- **QLoRA** (Dettmers et al., 2023)

### Libraries
- `flash-attn`: Official Flash Attention
- `peft`: Hugging Face Parameter-Efficient Fine-Tuning
- `bitsandbytes`: Quantization for QLoRA

---

⬅️ [Back: Scaling](../05-scaling/)

