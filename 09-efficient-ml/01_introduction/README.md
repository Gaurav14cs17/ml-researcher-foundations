# Lecture 1: Introduction to Efficient ML

[← Back to Course](../README.md) | [Next: Basics →](../02_basics/README.md)

📺 [Watch Lecture 1 on YouTube](https://www.youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&index=1)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09-efficient-ml/01_introduction/demo.ipynb) ← **Try the code!**

---

![Overview](overview.png)

## Why Efficient ML Matters

Machine learning models are getting bigger every year:
- GPT-2 (2019): 1.5B parameters
- GPT-3 (2020): 175B parameters
- GPT-4 (2023): ~1.8T parameters (estimated)

**The Problem:** Bigger models = more compute, more memory, more energy, more cost.

---

## The Efficiency Challenge

| Metric | Challenge |
|--------|-----------|
| **Latency** | Users expect real-time responses |
| **Throughput** | Serving millions of requests |
| **Energy** | Data centers consume massive power |
| **Memory** | GPUs have limited VRAM |
| **Cost** | Training GPT-3 cost ~$4.6M |

---

## Course Topics Overview

1. **Model Compression**
   - Pruning (remove weights)
   - Quantization (use fewer bits)
   - Knowledge Distillation (train smaller models)

2. **Efficient Architectures**
   - Neural Architecture Search
   - Hardware-aware design

3. **Efficient Training**
   - Mixed precision
   - Gradient checkpointing
   - Distributed training

4. **Efficient Inference**
   - KV cache optimization
   - Speculative decoding
   - Batching strategies

---

## Key Insight

> "The goal is not just to make models smaller, but to make them faster and cheaper while maintaining accuracy."

---

## Real-World Impact

| Application | Why Efficiency Matters |
|------------|----------------------|
| Mobile apps | Limited battery and compute |
| Self-driving cars | Real-time decisions needed |
| Voice assistants | Low-latency responses |
| Edge IoT | Microcontrollers have KB of memory |

---

---

## 📐 Mathematical Foundations

### Model Efficiency Metrics

**FLOPs (Floating Point Operations):**
$$\text{FLOPs} = 2 \times \text{MACs}$$

**Memory Footprint:**
$$\text{Memory} = \text{Params} \times \text{Bytes per param}$$

**Efficiency Ratio:**
$$\text{Efficiency} = \frac{\text{Accuracy}}{\text{FLOPs}} \text{ or } \frac{\text{Accuracy}}{\text{Latency}}$$

---

## 🎯 Where Used

| Concept | Applications |
|---------|-------------|
| Model Compression | Mobile apps, Edge devices |
| Efficient Training | Large-scale pretraining |
| Efficient Inference | Real-time systems, Cloud serving |
| Hardware-aware Design | Custom accelerators, TinyML |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 🌐 | EfficientML Course | [Website](https://efficientml.ai/) |
| 🏛️ | Song Han's Lab | [MIT](https://songhan.mit.edu/) |
| 🎥 | MIT 6.5940 TinyML | [Course](https://hanlab.mit.edu/courses/2024-fall-65940) |
| 📄 | Efficient Deep Learning Survey | [arXiv](https://arxiv.org/abs/2106.08962) |
| 🇨🇳 | 知乎 - 高效机器学习 | [Zhihu](https://www.zhihu.com/topic/19813032) |

