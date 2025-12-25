<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=6C63FF&height=120&section=header&text=⚡%20Introduction&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Lecture-01_of_18-6C63FF?style=for-the-badge&logo=bookstack&logoColor=white" alt="Lecture"/>
  <img src="https://img.shields.io/badge/Watch-YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="YouTube"/>
  <img src="https://img.shields.io/badge/MIT_6.5940-TinyML-red?style=for-the-badge" alt="MIT"/>
</p>

<p align="center">
  <i>MIT 6.5940 - Efficient ML Course</i>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

**✍️ Author:** [Gaurav Goswami](https://github.com/Gaurav14cs17) • **📅 Updated:** December 2024

---

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
```
\text{FLOPs} = 2 \times \text{MACs}
```

**Memory Footprint:**
```
\text{Memory} = \text{Params} \times \text{Bytes per param}
```

**Efficiency Ratio:**
```
\text{Efficiency} = \frac{\text{Accuracy}}{\text{FLOPs}} \text{ or } \frac{\text{Accuracy}}{\text{Latency}}
```

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


---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=6C63FF&height=80&section=footer" width="100%"/>
</p>
