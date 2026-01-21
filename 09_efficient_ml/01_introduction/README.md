<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=120&section=header&text=Lecture%201%20Introduction%20to%20Efficient%20ML&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-09-1ABC9C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

# Lecture 1: Introduction to Efficient ML

[← Back to Course](../) | [Next: Basics →](../02_basics/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09_efficient_ml/01_introduction/demo.ipynb) ← **Try the code!**

---

## 📺 Video Lecture

| Resource | Link |
|----------|------|
| 🎥 **Lecture Video** | [Watch on YouTube](https://www.youtube.com/watch?v=rCFvPEQTxKI&list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&index=1) |
| 📊 **Slides** | [MIT Course Page](https://hanlab.mit.edu/courses/2023-fall-65940) |
| ⏱️ **Duration** | ~90 minutes |

### 📝 Video Summary

This introductory lecture covers **why efficient machine learning matters** and provides an overview of the entire course. Key topics include:

- **The efficiency crisis**: Model sizes have grown 10,000x in 5 years (GPT-2 to GPT-4)

- **Real-world constraints**: Latency, throughput, energy consumption, memory limitations

- **Course roadmap**: Pruning → Quantization → NAS → Distillation → Efficient Inference

- **Case studies**: Deploying models on mobile, MCUs, and cloud at scale

- **The efficiency-accuracy trade-off**: How to maximize performance within constraints

> 💡 *"The goal is not just compression—it's achieving the same accuracy with fewer resources, or better accuracy with the same resources."* — Prof. Song Han

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

## 📐 Mathematical Foundations & Proofs

### Model Efficiency Metrics

#### FLOPs (Floating Point Operations)

For a matrix multiplication \( Y = XW \) where \( X \in \mathbb{R}^{m \times n} \) and \( W \in \mathbb{R}^{n \times p} \):

$$\text{FLOPs} = 2 \times m \times n \times p$$

**Proof:** Each output element \( Y_{ij} = \sum_{k=1}^{n} X_{ik} W_{kj} \) requires:

- \( n \) multiplications

- \( n-1 \) additions ≈ \( n \) operations

Total: \( m \times p \times 2n = 2mnp \) FLOPs.

#### Memory Footprint

$$\text{Memory} = \sum_{l=1}^{L} |\theta_l| \times b_l$$

where \( |\theta_l| \) is the number of parameters in layer \( l \) and \( b_l \) is bytes per parameter.

**Example:** A model with 7B parameters in FP16:

$$\text{Memory} = 7 \times 10^9 \times 2 \text{ bytes} = 14 \text{ GB}$$

#### Efficiency Ratio

$$\eta = \frac{\text{Accuracy}(\mathcal{M})}{\text{Cost}(\mathcal{M})}$$

where Cost can be FLOPs, latency, energy, or memory.

### Pareto Optimality

A model \( \mathcal{M}^* \) is **Pareto optimal** if no other model achieves:

- Higher accuracy with same or lower cost

- Same accuracy with lower cost

$$\nexists \mathcal{M}: \text{Acc}(\mathcal{M}) \geq \text{Acc}(\mathcal{M}^*) \land \text{Cost}(\mathcal{M}) \leq \text{Cost}(\mathcal{M}^*)$$

with at least one strict inequality.

### Scaling Laws (Chinchilla)

The optimal relationship between compute \( C \), parameters \( N \), and data \( D \):

$$L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + E$$

where:

- \( \alpha \approx 0.34 \), \( \beta \approx 0.28 \)

- \( A, B, E \) are constants

- Optimal allocation: \( D \approx 20N \) (tokens ≈ 20× parameters)

---

## 🧮 Key Derivations

### Compute-Memory Trade-off

For transformer inference, the compute-to-memory ratio:

$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes}} = \frac{2 \times \text{batch} \times \text{seq} \times d^2}{d^2 \times \text{bytes_per_param}}$$

**Insight:** At batch size 1, arithmetic intensity is low → memory-bound.

### Energy Consumption Model

$$E_{\text{total}} = E_{\text{compute}} + E_{\text{memory}}
E_{\text{compute}} = \text{FLOPs} \times E_{\text{per_FLOP}}
E_{\text{memory}} = \text{Data_moved} \times E_{\text{per_byte}}$$

For modern hardware: \( E_{\text{memory}} \gg E_{\text{compute}} \)

Memory access costs **100-1000×** more energy than arithmetic!

---

## 🎯 Where Used

| Concept | Applications |
|---------|-------------|
| Model Compression | Mobile apps, Edge devices |
| Efficient Training | Large-scale pretraining |
| Efficient Inference | Real-time systems, Cloud serving |
| Hardware-aware Design | Custom accelerators, TinyML |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [← Section 08: Model Compression](../../08_model_compression/README.md) | [Efficient ML](../README.md) | [Basics →](../02_basics/README.md) |

---
## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 🌐 | EfficientML Course | [Website](https://efficientml.ai/) |
| 🏛️ | Song Han's Lab | [MIT](https://songhan.mit.edu/) |
| 🎥 | MIT 6.5940 TinyML | [Course](https://hanlab.mit.edu/courses/2024-fall-65940) |
| 📄 | Efficient Deep Learning Survey | [arXiv](https://arxiv.org/abs/2106.08962) |
| 📄 | Chinchilla Scaling Laws | [arXiv](https://arxiv.org/abs/2203.15556) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=80&section=footer" width="100%"/>
</p>
