<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=5,6,7&height=180&section=header&text=🗜️%20Model%20Compression&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=32&desc=Make%20Models%20Small%20%26%20Fast&descAlignY=52&descSize=18" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/📚_Topics-6_Modules-blue?style=for-the-badge" alt="Topics"/>
  <img src="https://img.shields.io/badge/⏱️_Time-2_Weeks-green?style=for-the-badge" alt="Time"/>
  <img src="https://img.shields.io/badge/📊_Level-Advanced-red?style=for-the-badge" alt="Level"/>
</p>

<p align="center">
  <a href="#-main-topics"><img src="https://img.shields.io/badge/Start_Learning-607D8B?style=for-the-badge&logo=rocket&logoColor=white" alt="Start"/></a>
  <a href="../09-efficient-ml/README.md"><img src="https://img.shields.io/badge/Next:_Efficient_ML-00C853?style=for-the-badge&logo=arrow-right&logoColor=white" alt="Next"/></a>
</p>

---

**✍️ Author:** [Gaurav Goswami](https://github.com/Gaurav14cs17) • **📅 Updated:** December 2024

---

## 📊 Learning Path

```mermaid
graph LR
    A[🚀 Start] --> B[🔢 Quantization]
    B --> C[INT8/INT4]
    C --> D[✂️ Pruning]
    D --> E[🔧 LoRA]
    E --> F[🎓 Distill]
    F --> G[🧩 MoE]
    G --> H[🏭 Production]
```

## 🎯 What You'll Learn

> 💡 From **175B parameters to your phone**: Compress models 4-10x with minimal accuracy loss

<table>
<tr>
<td align="center">

### 🔢 Quantization
4x smaller ⭐

</td>
<td align="center">

### 🔧 LoRA
0.1% params 🔥

</td>
<td align="center">

### 🎓 Distillation
10x smaller

</td>
</tr>
</table>

---

## 📚 Main Topics

### 1️⃣ Quantization ⭐⭐⭐

<img src="https://img.shields.io/badge/Time-6_hours-blue?style=flat-square"/> <img src="https://img.shields.io/badge/🔥_MOST_PRACTICAL-critical?style=flat-square"/>

```mermaid
graph LR
    A[FP32] --> B[FP16]
    B --> C[INT8]
    C --> D[INT4]
    D --> E[4x Smaller]
```

> ⭐ **4x memory reduction with <1% accuracy loss**

| Type | Compression | Use Case |
|:----:|:-----------:|----------|
| INT8 | 4x | ⭐ Production default |
| INT4 | 8x | Aggressive |
| GPTQ/AWQ | LLMs | 🔥 Hot |

<a href="./03-quantization/README.md"><img src="https://img.shields.io/badge/📖_Dive_In-607D8B?style=for-the-badge" alt="Learn"/></a>

---

### 2️⃣ LoRA & PEFT 🔥🔥🔥

<img src="https://img.shields.io/badge/Time-6_hours-blue?style=flat-square"/> <img src="https://img.shields.io/badge/🔥_HOTTEST_2024-critical?style=flat-square"/>

```mermaid
graph LR
    A[Full Finetune] --> B[Expensive]
    C[LoRA] --> D[Low-Rank]
    D --> E[0.1% Params]
```

> 🔥 **Fine-tune LLMs with 0.1% parameters** - Industry standard

**Used in:** Stable Diffusion, LLaMA, every major LLM

<a href="./08-peft/README.md"><img src="https://img.shields.io/badge/📖_Dive_In-607D8B?style=for-the-badge" alt="Learn"/></a>

---

### 3️⃣ Knowledge Distillation

<img src="https://img.shields.io/badge/Time-4_hours-blue?style=flat-square"/>

```mermaid
graph LR
    A[Teacher] --> B[Soft Labels]
    B --> C[Student]
    C --> D[90% Acc<br/>10x Smaller]
```

**Example:** BERT → DistilBERT (40% smaller, 97% accuracy)

<a href="./04-knowledge-distillation/README.md"><img src="https://img.shields.io/badge/📖_Dive_In-607D8B?style=for-the-badge" alt="Learn"/></a>

---

### 4️⃣ Pruning

<img src="https://img.shields.io/badge/Time-4_hours-blue?style=flat-square"/>

```mermaid
graph LR
    A[Dense] --> B[Identify]
    B --> C[Remove]
    C --> D[Finetune]
    D --> E[Sparse]
```

**Core:** Magnitude Pruning, Lottery Ticket Hypothesis

<a href="./02-parameter-reduction/pruning/README.md"><img src="https://img.shields.io/badge/📖_Dive_In-607D8B?style=for-the-badge" alt="Learn"/></a>

---

### 5️⃣ Mixture of Experts (MoE)

<img src="https://img.shields.io/badge/Time-4_hours-blue?style=flat-square"/>

```mermaid
graph LR
    A[Dense] --> B[Split]
    B --> C[Router]
    C --> D[Activate Few]
    D --> E[Sparse Act]
```

> Scale to **trillions of parameters** - Used in GPT-4 (rumored)

<a href="./06-sparsity/moe/README.md"><img src="https://img.shields.io/badge/📖_Dive_In-607D8B?style=for-the-badge" alt="Learn"/></a>

---

### 6️⃣ Efficient Architectures

<img src="https://img.shields.io/badge/Time-4_hours-blue?style=flat-square"/>

```mermaid
graph LR
    A[Standard] --> B[Flash Attn]
    B --> C[Linear Attn]
    C --> D[Efficient]
    D --> E[10x Faster]
```

> ⚡ **Flash Attention in all modern LLMs**

<a href="./07-efficient-architectures/README.md"><img src="https://img.shields.io/badge/📖_Dive_In-607D8B?style=for-the-badge" alt="Learn"/></a>

---

## 🔄 Comparison

| Technique | Compression | Accuracy Loss | Best For |
|:---------:|:-----------:|:-------------:|----------|
| **INT8** | 4x | <1% | ⭐ Production |
| **INT4** | 8x | 1-3% | Aggressive |
| **LoRA** | N/A | 0% | 🔥 Fine-tuning |
| **Distill** | 2-10x | 3-10% | Deployment |
| **Pruning** | 2-10x | 0-5% | Research |

---

## 💡 Key Formulas

<table>
<tr>
<td>

### 🔢 Quantization
```
x_quant = round(x/scale) + zero
x_dequant = (x_quant - zero) × scale
```

</td>
<td>

### 🔧 LoRA
```
W' = W + BA  (r << n,m)
Only train B, A
```

</td>
</tr>
</table>

---

## 🔗 Prerequisites & Next Steps

```mermaid
graph LR
    A[🧬 Deep Learning] --> B[🗜️ Compression]
    B --> C[⚡ Efficient ML]
    C --> D[🏭 Production]
```

<p align="center">
  <a href="../06-deep-learning/README.md"><img src="https://img.shields.io/badge/←_Prerequisites:_Deep_Learning-gray?style=for-the-badge" alt="Prev"/></a>
  <a href="../09-efficient-ml/README.md"><img src="https://img.shields.io/badge/Next:_Efficient_ML_→-00C853?style=for-the-badge" alt="Next"/></a>
</p>

---

## 📚 Recommended Resources

| Type | Resource | Focus |
|:----:|----------|-------|
| 📄 | [LoRA Paper](https://arxiv.org/abs/2106.09685) | Low-Rank Adaptation |
| 📄 | [QLoRA Paper](https://arxiv.org/abs/2305.14314) | Quantized LoRA |
| 🛠️ | [PEFT](https://github.com/huggingface/peft) | LoRA library |
| 🛠️ | [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | Quantization |

---

## 🗺️ Quick Navigation

| Previous | Current | Next |
|:--------:|:-------:|:----:|
| [🎮 RL](../07-reinforcement-learning/README.md) | **🗜️ Compression** | [⚡ Efficient ML →](../09-efficient-ml/README.md) |

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=5,6,7&height=100&section=footer" width="100%"/>
</p>
