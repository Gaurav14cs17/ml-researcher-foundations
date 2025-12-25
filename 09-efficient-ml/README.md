<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=120&section=header&text=Topic&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-09-1ABC9C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📊 Learning Path

```mermaid
graph LR
    A[🚀 Start] --> B[📊 Basics]
    B --> C[✂️ Pruning]
    C --> D[🔢 Quantize]
    D --> E[🔍 NAS]
    E --> F[📱 TinyML]
    F --> G[⚡ Training]
    G --> H[🏆 Expert]
```

## 🎯 What You'll Learn

> 🚀 **From Cloud to Edge**: Deploy AI on phones, microcontrollers, and edge devices

<table>
<tr>
<td align="center">

### ✂️ Pruning
50-90% sparse

</td>
<td align="center">

### 🔢 Quantization
GPTQ, AWQ, QLoRA

</td>
<td align="center">

### 📱 TinyML
256KB inference!

</td>
<td align="center">

### ⚡ Flash Attention
5x faster

</td>
</tr>
</table>

---

## 📚 Course Modules

### 📖 Lectures 1-2: Introduction & Basics

<img src="https://img.shields.io/badge/Time-3_hours-blue?style=flat-square"/>

```mermaid
graph LR
    A[Why Efficiency] --> B[FLOPs]
    B --> C[Memory]
    C --> D[Roofline]
    D --> E[Profiling]
```

> 💡 **"Memory is the bottleneck, not compute"**

<a href="./01_introduction/"><img src="https://img.shields.io/badge/📖_Notes-4CAF50?style=for-the-badge" alt="Notes"/></a>
<a href="https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09-efficient-ml/01_introduction/demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"/></a>

---

### ✂️ Lectures 3-4: Pruning & Sparsity

<img src="https://img.shields.io/badge/Time-4_hours-blue?style=flat-square"/>

```mermaid
graph LR
    A[Dense] --> B[Magnitude]
    B --> C[Sparse]
    C --> D[Lottery]
    D --> E[Structured]
```

**Key:** Lottery Ticket Hypothesis, 50-90% sparsity

<a href="./03_pruning_sparsity_1/"><img src="https://img.shields.io/badge/📖_Notes-4CAF50?style=for-the-badge" alt="Notes"/></a>
<a href="https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09-efficient-ml/03_pruning_sparsity_1/demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"/></a>

---

### 🔢 Lectures 5-6: Quantization ⭐

<img src="https://img.shields.io/badge/Time-6_hours-blue?style=flat-square"/> <img src="https://img.shields.io/badge/🔥_HOT-critical?style=flat-square"/>

```mermaid
graph LR
    A[FP32] --> B[INT8]
    B --> C[INT4]
    C --> D[GPTQ]
    D --> E[QLoRA]
```

> 🔥 **QLoRA: Train 65B on single GPU**

<a href="./05_quantization_1/"><img src="https://img.shields.io/badge/📖_Notes-4CAF50?style=for-the-badge" alt="Notes"/></a>
<a href="https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09-efficient-ml/05_quantization_1/demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"/></a>

---

### 🔍 Lectures 7-8: Neural Architecture Search

<img src="https://img.shields.io/badge/Time-4_hours-blue?style=flat-square"/>

```mermaid
graph LR
    A[Manual] --> B[AutoML]
    B --> C[DARTS]
    C --> D[Once-for-All]
    D --> E[Efficient]
```

<a href="./07_neural_architecture_search_1/"><img src="https://img.shields.io/badge/📖_Notes-4CAF50?style=for-the-badge" alt="Notes"/></a>
<a href="https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09-efficient-ml/07_neural_architecture_search_1/demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"/></a>

---

### 📱 Lectures 9-10: Distillation & TinyML

<img src="https://img.shields.io/badge/Time-4_hours-blue?style=flat-square"/>

```mermaid
graph LR
    A[Teacher] --> B[Distill]
    B --> C[Student]
    C --> D[MCUNet]
    D --> E[256KB]
```

> 📱 **MCUNet: Run ML on 256KB microcontrollers**

<a href="./09_knowledge_distillation/"><img src="https://img.shields.io/badge/📖_Notes-4CAF50?style=for-the-badge" alt="Notes"/></a>
<a href="https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09-efficient-ml/09_knowledge_distillation/demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"/></a>

---

### ⚡ Lectures 11-12: Efficient Transformers 🔥

<img src="https://img.shields.io/badge/Time-5_hours-blue?style=flat-square"/> <img src="https://img.shields.io/badge/🔥_ESSENTIAL-critical?style=flat-square"/>

```mermaid
graph LR
    A[Standard] --> B[Flash Attn]
    B --> C[Linear]
    C --> D[Sparse]
    D --> E[10x Fast]
```

> ⚡ **Flash Attention: 5x faster, O(n) memory** - In all modern LLMs

<a href="./11_efficient_transformers/"><img src="https://img.shields.io/badge/📖_Notes-4CAF50?style=for-the-badge" alt="Notes"/></a>
<a href="https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09-efficient-ml/11_efficient_transformers/demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"/></a>

---

### 🌐 Lectures 13-14: Distributed Training

<img src="https://img.shields.io/badge/Time-5_hours-blue?style=flat-square"/>

```mermaid
graph LR
    A[Single GPU] --> B[Data Par]
    B --> C[Model Par]
    C --> D[ZeRO]
    D --> E[Trillion]
```

**Core:** ZeRO, FSDP, DeepSpeed, Megatron

<a href="./14_distributed_training/"><img src="https://img.shields.io/badge/📖_Notes-4CAF50?style=for-the-badge" alt="Notes"/></a>
<a href="https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09-efficient-ml/14_distributed_training/demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"/></a>

---

### 🚀 Lectures 15-18: Efficient Models

<img src="https://img.shields.io/badge/Time-6_hours-blue?style=flat-square"/>

```mermaid
graph LR
    A[LLMs] --> B[Diffusion]
    B --> C[Vision]
    C --> D[Production]
```

**Covered:** LLaMA, Stable Diffusion, MobileNets, Edge Deployment

<a href="./16_efficient_llms/"><img src="https://img.shields.io/badge/📖_LLMs-4CAF50?style=for-the-badge" alt="LLMs"/></a>
<a href="./17_efficient_diffusion_models/"><img src="https://img.shields.io/badge/📖_Diffusion-4CAF50?style=for-the-badge" alt="Diffusion"/></a>
<a href="./15_efficient_vision_models/"><img src="https://img.shields.io/badge/📖_Vision-4CAF50?style=for-the-badge" alt="Vision"/></a>

---

## 💡 Key Takeaways

<table>
<tr>
<td>

### 📊 Roofline Model
```
Perf = min(Peak_Compute, 
           Peak_BW × Intensity)
```

</td>
<td>

### 🔢 Quantization
```
FP32 → INT8: 4x smaller
FP32 → INT4: 8x smaller
2-4x speedup
```

</td>
<td>

### ⚡ Flash Attention
```
Standard: O(N²) memory
Flash: O(N) memory
5x faster!
```

</td>
</tr>
</table>

---

## 🔗 Course Structure

| Week | Lectures | Topics | Lab |
|:----:|:--------:|--------|:---:|
| 1-2 | L1-L4 | Intro, Pruning | ✂️ |
| 3-4 | L5-L8 | Quantization, NAS | 🔢 |
| 5-6 | L9-L10 | Distillation, TinyML | 📱 |
| 7-8 | L11-L14 | Transformers, Training | ⚡ |
| 9-10 | L15-L18 | Production Models | 🚀 |

---

## 🛠️ Tools You'll Use

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/bitsandbytes-Quantization-blue?style=for-the-badge" alt="bitsandbytes"/>
  <img src="https://img.shields.io/badge/PEFT-LoRA-green?style=for-the-badge" alt="PEFT"/>
  <img src="https://img.shields.io/badge/DeepSpeed-Training-orange?style=for-the-badge" alt="DeepSpeed"/>
</p>

---

## 📚 Official Resources

| Resource | Link |
|:--------:|------|
| 📺 **Course Website** | [hanlab.mit.edu](https://hanlab.mit.edu/courses/2023-fall-65940) |
| 📺 **YouTube Playlist** | [Watch Lectures](https://www.youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB) |
| 📝 **Lecture Slides** | [Download PDFs](https://hanlab.mit.edu/courses/2023-fall-65940) |

---

## 🗺️ Quick Navigation

| Previous | Current | Next |
|:--------:|:-------:|:----:|
| [🗜️ Compression](../08-model-compression/README.md) | **⚡ Efficient ML** | 🏆 **Production Ready!** |

---

<p align="center">
  <b>🎓 Ready to deploy AI anywhere?</b>
  <br/><br/>
  <a href="./01_introduction/"><img src="https://img.shields.io/badge/Start_Learning_→-4CAF50?style=for-the-badge&logo=rocket&logoColor=white" alt="Start"/></a>
</p>

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=80&section=footer" width="100%"/>
</p>
