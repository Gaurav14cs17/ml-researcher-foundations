# 🔧 Parameter-Efficient Fine-Tuning (PEFT)

> **Fine-tune large models with minimal parameters**

## 🎯 Visual Overview

<img src="./images/peft.svg" width="100%">

*Caption: PEFT methods (LoRA, QLoRA, Adapters) train only 0.01-1% of parameters instead of full fine-tuning. LoRA adds trainable low-rank matrices BA, merging at inference with no overhead.*

---

## 📂 Topics

| File | Topic | Key Concepts |
|------|-------|--------------|

---

## 🎯 Why PEFT Matters

```
Full Fine-Tuning LLaMA-7B:
+-- Parameters to train: 7,000,000,000
+-- GPU memory: 28+ GB (optimizer states)
+-- Storage per task: 14 GB
+-- Cost: $$$$$

PEFT (LoRA) Fine-Tuning:
+-- Parameters to train: 4,000,000 (0.06%)
+-- GPU memory: 8 GB (frozen base)
+-- Storage per task: 16 MB
+-- Cost: $
```

---

## 📊 PEFT Methods Comparison

| Method | Trainable Params | Quality | Speed | Memory |
|--------|-----------------|---------|-------|--------|
| **Full Fine-Tune** | 100% | Best | Slow | High |
| **LoRA** | 0.1% | ~Full | Fast | Low |
| **QLoRA** | 0.1% | ~Full | Fast | Very Low |
| **Adapters** | 1-5% | Good | Medium | Medium |
| **Prefix Tuning** | 0.1% | Good | Fast | Low |
| **IA³** | 0.01% | Good | Fastest | Lowest |

---

## 🔥 LoRA: The Standard

```
+-------------------------------------------------------------+
|                      Transformer Layer                       |
+-------------------------------------------------------------+
|                                                             |
|  Input x                                                    |
|      |                                                      |
|      ▼      |
|  +------------+-------------+                              |
|  | W (frozen) | B·A (LoRA)  |                              |
|  | 4096×4096  | 4096×16×16  |                              |
|  | 16M params | 130K params |                              |
|  +------------+-------------+                              |
|      |              |                                       |
|      +------+-------+                                       |
|             |                                               |
|      ▼      |
|         Output = Wx + BAx                                   |
|                                                             |
+-------------------------------------------------------------+

At inference: W_new = W + BA (merge, no overhead!)
```

---

## 💻 Code Example

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True  # QLoRA!
)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                                    # Rank
    lora_alpha=32,                           # Scaling factor
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Train only LoRA parameters!
for name, param in model.named_parameters():
    if "lora" not in name:
        param.requires_grad = False

# trainable params: 4,194,304 (0.06%)
```

---

## 🔗 Where This Topic Is Used

| Topic | How PEFT Is Used |
|-------|-----------------|
| **All LLM Fine-tuning** | LoRA/QLoRA is standard |
| **ChatGPT Fine-tuning** | Likely uses adapters |
| **Hugging Face PEFT** | Library for all methods |
| **Stable Diffusion LoRA** | Custom styles/characters |
| **Multi-task Learning** | One base, many adapters |
| **Personalization** | User-specific adapters |

### Prerequisite For

```
PEFT --> Fine-tuning LLMs affordably
    --> QLoRA (4-bit + LoRA)
    --> Multi-task adapters
    --> Stable Diffusion LoRA
    --> Production fine-tuning
```

---

## 📐 Mathematical Foundations

<img src="./images/lora-math.svg" width="100%">

---

## 📚 References & Resources

### 📄 Key Papers

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|------------------|
| [LoRA](https://arxiv.org/abs/2106.09685) | Hu et al. | 2021 | Low-rank adaptation (original) |
| [QLoRA](https://arxiv.org/abs/2305.14314) | Dettmers et al. | 2023 | 4-bit LoRA, fine-tune 65B on 48GB |
| [Adapters](https://arxiv.org/abs/1902.00751) | Houlsby et al. | 2019 | Adapter layers for NLP |
| [AdapterHub](https://arxiv.org/abs/2007.07779) | Pfeiffer et al. | 2020 | Adapter sharing platform |
| [Prefix Tuning](https://arxiv.org/abs/2101.00190) | Li & Liang | 2021 | Soft prompt tuning |
| [P-Tuning v2](https://arxiv.org/abs/2110.07602) | Liu et al. | 2021 | Deep prompt tuning |
| [IA³](https://arxiv.org/abs/2205.05638) | Liu et al. | 2022 | Learned rescaling vectors |
| [AdaLoRA](https://arxiv.org/abs/2303.10512) | Zhang et al. | 2023 | Adaptive rank allocation |
| [DoRA](https://arxiv.org/abs/2402.09353) | Liu et al. | 2024 | Weight decomposition |
| 🇨🇳 [LoRA原理详解](https://zhuanlan.zhihu.com/p/631077302) | 知乎 | - | 从SVD角度理解LoRA |
| 🇨🇳 [PEFT方法总结](https://www.jiqizhixin.com/articles/2023-08-07-7) | 机器之心 | 2023 | 参数高效微调综述 |
| 🇨🇳 [PEFT使用指南](https://huggingface.co/blog/zh/peft) | Hugging Face | - | 官方中文教程 |
| 🇨🇳 [LoRA微调实战](https://www.bilibili.com/video/BV1cP4y1Y7DN) | B站 | - | 视频教程 |
| 🇨🇳 [魔搭LoRA教程](https://modelscope.cn/docs) | ModelScope | - | 阿里魔搭教程 |

### 🔧 Hyperparameter Guide

| Parameter | Typical Value | Description |
|-----------|--------------|-------------|
| `r` (rank) | 8, 16, 32 | Lower = fewer params, higher = more capacity |
| `lora_alpha` | 16, 32 | Scaling factor (usually α = 2r) |
| `lora_dropout` | 0.05-0.1 | Regularization |
| `target_modules` | q,v,k,o_proj | Which layers to adapt |

### 🎓 Courses

| Course | Description | Link |
|--------|-------------|------|
| 🔥 MIT 6.5940 | Prof. Song Han's TinyML: LLM labs with LoRA | [Course](https://hanlab.mit.edu/courses/2024-fall-65940) |

### 🛠️ Tools

| Tool | Description | Link |
|------|-------------|------|
| PEFT | Official HF library | [GitHub](https://github.com/huggingface/peft) |
| LLaMA-Factory | LoRA training toolkit | [GitHub](https://github.com/hiyouga/LLaMA-Factory) |
| Axolotl | Fine-tuning framework | [GitHub](https://github.com/OpenAccess-AI-Collective/axolotl) |

---

⬅️ [Back: 07-Efficient Architectures](../07-efficient-architectures/) | ➡️ [Next: 09-Compression Pipelines](../09-compression-pipelines/)

