<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=120&section=header&text=Quantization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-08-E74C3C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📂 Topics

| File | Topic | Key Concepts |
|------|-------|--------------|

---

## 🎯 The Core Idea

```
FP32 (32 bits):
+-------------------------------------+
|  1 sign | 8 exponent | 23 mantissa  |
|  High precision, large memory       |
+-------------------------------------+

INT8 (8 bits):
+-------------------------------------+
|  1 sign | 7 value bits              |
|  Range: -128 to 127                 |
|  4x smaller, 2-4x faster!           |
+-------------------------------------+

INT4 (4 bits):
+-------------------------------------+
|  4 value bits                       |
|  Range: 0 to 15 (or -8 to 7)       |
|  8x smaller, even faster!           |
+-------------------------------------+
```

---

## 📊 Comparison

| Format | Size | Speed | Accuracy | Use Case |
|--------|------|-------|----------|----------|
| **FP32** | 1x | 1x | Baseline | Training |
| **FP16/BF16** | 2x | 2x | ~Same | Training + inference |
| **INT8** | 4x | 2-4x | <1% drop | Production inference |
| **INT4** | 8x | 4-8x | 1-3% drop | LLM inference |

---

## 🔥 Quantization for LLMs

```
LLaMA-7B:
+-- FP16:  14 GB (fits 24GB GPU barely)
+-- INT8:   7 GB (fits 8GB GPU!)
+-- INT4:  3.5 GB (fits laptop!)

QLoRA = INT4 base + FP16 LoRA adapters
     = Best of both worlds!
```

---

## 💻 Code Example

```python
from transformers import AutoModelForCausalLM
import torch

# Load in 4-bit for inference
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    device_map="auto"
)

# 14 GB → 3.5 GB, runs on consumer GPU!
```

---

## 🔗 Where This Topic Is Used

| Topic | How Quantization Is Used |
|-------|-------------------------|
| **llama.cpp** | INT4/INT8 for CPU inference |
| **GGML/GGUF** | Quantization formats |
| **QLoRA** | 4-bit base model for fine-tuning |
| **TensorRT** | INT8 for NVIDIA GPUs |
| **CoreML** | Apple device optimization |
| **Mobile Models** | INT8 for edge deployment |
| **Production LLMs** | Reduce serving costs |

### Prerequisite For

```
Quantization --> Running LLMs on consumer hardware
            --> QLoRA fine-tuning
            --> Production deployment
            --> Mobile/edge AI
```

---

## 📐 Mathematical Foundations

<img src="./images/quantization-math.svg" width="100%">

### Uniform Quantization

```
Quantize: Q(x) = round(x/s) + z
Dequantize: x̂ = s · (Q(x) - z)

Where:
• s = scale = (x_max - x_min) / (2^b - 1)
• z = zero-point = round(-x_min / s)
• b = bit-width (e.g., 8 for INT8)

Symmetric (z=0):
s = max(|x_max|, |x_min|) / (2^(b-1) - 1)
Q(x) = clamp(round(x/s), -2^(b-1), 2^(b-1)-1)
```

### Per-Channel vs Per-Tensor

```
Per-tensor: One scale for entire tensor
           Simpler but less accurate

Per-channel: One scale per output channel
            W_q[c,i] = round(W[c,i] / s[c])
            More accurate, slightly more complex

Per-group (GPTQ/AWQ):
           Groups of weights share scale
           Balance between accuracy and efficiency
```

### Quantization Error Analysis

```
Error = x - Q⁻¹(Q(x))
      ≤ s/2  (maximum quantization error)

Mean Squared Error:
MSE = E[(x - x̂)²] ≈ s²/12  (uniform distribution)

Minimize by choosing optimal scale s
```

### Matrix Multiplication in INT8

```
Y = XW (FP32)
↓
Y = s_x · s_w · (X_int8 @ W_int8) (INT8)
↓
Y_fp32 = dequantize(Y)

Speed: INT8 ops are 2-4x faster on modern hardware
```

---

## 📚 References & Resources

### 📄 Key Papers

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|------------------|
| [LLM.int8()](https://arxiv.org/abs/2208.07339) | Dettmers et al. | 2022 | 8-bit matrix multiplication for LLMs |
| [GPTQ](https://arxiv.org/abs/2210.17323) | Frantar et al. | 2022 | One-shot quantization for GPT models |
| [QLoRA](https://arxiv.org/abs/2305.14314) | Dettmers et al. | 2023 | 4-bit quantization + LoRA |
| [AWQ](https://arxiv.org/abs/2306.00978) | Lin et al. | 2023 | Activation-aware weight quantization |
| [SmoothQuant](https://arxiv.org/abs/2211.10438) | Xiao et al. | 2022 | Migrate quantization difficulty |
| [ZeroQuant](https://arxiv.org/abs/2206.01861) | Yao et al. | 2022 | End-to-end quantization |
| [Deep Compression](https://arxiv.org/abs/1510.00149) | Han et al. | 2015 | Classic pruning+quantization+Huffman |
| 🇨🇳 模型量化技术详解 | [知乎](https://www.zhihu.com/question/627401910) | - | INT8/INT4量化原理 |
| 🇨🇳 LLM量化方法总结 | [CSDN](https://blog.csdn.net/v_JULY_v/article/details/134546262) | - | GPTQ/AWQ/GGML对比 |
| 🇨🇳 QLoRA深度解析 | [机器之心](https://www.jiqizhixin.com/articles/2023-05-25-6) | - | QLoRA原理与实践 |
| 🇨🇳 bitsandbytes教程 | [HF中文](https://huggingface.co/blog/zh/hf-bitsandbytes-integration) | - | 使用教程 |

### 🎓 Courses

| Course | Description | Link |
|--------|-------------|------|
| 🔥 MIT 6.5940 | Prof. Song Han's TinyML: Quantization lectures 5-6 | [Course](https://hanlab.mit.edu/courses/2024-fall-65940) |

### 🛠️ Tools & Libraries

| Tool | Description | Link |
|------|-------------|------|
| bitsandbytes | 4/8-bit quantization | [GitHub](https://github.com/TimDettmers/bitsandbytes) |
| llama.cpp | GGML/GGUF quantization | [GitHub](https://github.com/ggerganov/llama.cpp) |
| AutoGPTQ | GPTQ implementation | [GitHub](https://github.com/PanQiWei/AutoGPTQ) |
| AutoAWQ | AWQ implementation | [GitHub](https://github.com/casper-hansen/AutoAWQ) |

---

⬅️ [Back: 02-Parameter Reduction](../02-parameter-reduction/) | ➡️ [Next: 04-Knowledge Distillation](../04-knowledge-distillation/)


---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=80&section=footer" width="100%"/>
</p>
