<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=04 Knowledge Distillation&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 🎓 Knowledge Distillation

> **Transferring knowledge from large to small models**

## 🎯 Visual Overview

<img src="./images/knowledge-distillation.svg" width="100%">

*Caption: Knowledge distillation transfers "dark knowledge" from a large teacher model to a smaller student. Student learns from soft labels (teacher's predictions) plus hard labels (ground truth).*

---

## 📂 Topics

| File | Topic | Key Concepts |
|------|-------|--------------|

---

## 🎯 The Core Idea

```
Teacher (Large):                     Student (Small):
+----------------+                   +----------------+
|                |                   |                |
|   BERT-Large   |   Knowledge       |   DistilBERT   |
|   340M params  | ------------>     |   66M params   |
|   Expert       |                   |   Learns from  |
|                |                   |   teacher      |
+----------------+                   +----------------+

Student learns:
1. Hard labels (ground truth)
2. Soft labels (teacher's probabilities)
3. Intermediate features (optional)
```

---

## 📐 Distillation Loss

```
Standard Cross-Entropy:
L_hard = CrossEntropy(student_output, true_labels)

Distillation Loss:
L_soft = KL(softmax(student/T), softmax(teacher/T))
         ↑
         Temperature T (usually 2-20)
         Higher T → softer probabilities → more info

Total Loss:
L = α·L_soft + (1-α)·L_hard
```

---

## 💻 Code Example

```python
import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, 
                      temperature=4.0, alpha=0.7):
    """
    Hinton's knowledge distillation loss
    """
    # Soft targets (from teacher)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # Hard targets (ground truth)
    hard_loss = F.cross_entropy(student_logits, labels)
    
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

---

## 🌍 Famous Distilled Models

| Teacher | Student | Size Reduction | Use Case |
|---------|---------|---------------|----------|
| **BERT-Base** | DistilBERT | 40% smaller | NLP |
| **GPT-3** | Alpaca, Vicuna | 100x+ smaller | Chat |
| **Whisper** | Distil-Whisper | 2x smaller | Speech |
| **Stable Diffusion** | SD Turbo | 4x faster | Image gen |

---

## 🔗 Where This Topic Is Used

| Topic | How Distillation Is Used |
|-------|-------------------------|
| **DistilBERT** | Distill BERT to smaller model |
| **Alpaca / Vicuna** | Distill GPT to open models |
| **TinyLLaMA** | Train smaller LLM with teacher |
| **Model Compression** | Part of compression pipeline |
| **Mobile Deployment** | Create small models from large |
| **RLHF** | Distill reward model |

### Prerequisite For

```
Knowledge Distillation --> Creating efficient models
                      --> Open-source LLMs (many distilled from GPT)
                      --> Production deployment
                      --> Mobile/edge AI
```

---

## 📐 Mathematical Foundations

<img src="./images/distillation-math.svg" width="100%">

---

## 📚 References & Resources

### 📄 Key Papers

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|------------------|
| [Distilling Knowledge](https://arxiv.org/abs/1503.02531) | Hinton et al. | 2015 | Original KD paper (dark knowledge) |
| [DistilBERT](https://arxiv.org/abs/1910.01108) | Sanh et al. | 2019 | Distilling BERT to 66M params |
| [TinyBERT](https://arxiv.org/abs/1909.10351) | Jiao et al. | 2019 | Two-stage distillation |
| [MobileBERT](https://arxiv.org/abs/2004.02984) | Sun et al. | 2020 | Task-agnostic distillation |
| [MiniLM](https://arxiv.org/abs/2002.10957) | Wang et al. | 2020 | Self-attention distillation |
| [Distillation Survey](https://arxiv.org/abs/2006.05525) | Gou et al. | 2020 | Comprehensive survey |
| [Patient KD](https://arxiv.org/abs/1908.09355) | Sun et al. | 2019 | Layer-wise distillation |
| 🇨🇳 [知识蒸馏原理详解](https://zhuanlan.zhihu.com/p/102038521) | 知乎 | - | 从Hinton论文讲起 |
| 🇨🇳 [模型蒸馏技术](https://www.jiqizhixin.com/articles/2020-01-14-10) | 机器之心 | 2020 | 各类蒸馏方法对比 |
| 🇨🇳 [知识蒸馏课程](https://www.bilibili.com/video/BV1J94y1f7u5) | B站 | - | 李宏毅ML课程 |
| 🇨🇳 [蒸馏教程](https://www.paddlepaddle.org.cn/tutorials) | 飞桨 | - | 百度蒸馏实践 |

### 🌟 Important Distilled Models

| Teacher → Student | Size Reduction | Method |
|-------------------|---------------|--------|
| BERT-Base → DistilBERT | 340M → 66M | Hinton KD |
| GPT-3/4 → Alpaca/Vicuna | 175B → 7B | Instruction tuning |
| Whisper → Distil-Whisper | 1.5B → 750M | Feature KD |
| CLIP → TinyCLIP | 400M → 19M | Multi-modal KD |

### 🎓 Courses

| Course | Description | Link |
|--------|-------------|------|
| 🔥 MIT 6.5940 | Prof. Song Han's TinyML: Lecture 9 Knowledge Distillation | [Course](https://hanlab.mit.edu/courses/2024-fall-65940) |

### 🛠️ Tools

| Tool | Description | Link |
|------|-------------|------|
| Hugging Face Transformers | DistilBERT, etc. | [HF](https://huggingface.co/distilbert-base-uncased) |
| TextBrewer | Knowledge distillation toolkit | [GitHub](https://github.com/airaria/TextBrewer) |

---

⬅️ [Back: 03-Quantization](../03-quantization/) | ➡️ [Next: 05-Factorization](../05-factorization/)


---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
