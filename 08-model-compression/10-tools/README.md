<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=10 Tools&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 🛠️ Tools and Frameworks

> **Libraries for model compression**

<img src="./images/tools-overview.svg" width="100%">

---

## 📐 Mathematical Foundations

### Quantization in Tools
```
PyTorch quantization:
x_q = round(x / scale) + zero_point
x ≈ (x_q - zero_point) × scale

bitsandbytes NF4:
Uses normalized float with 4-bit precision
16 values: [-1.0, ..., 0, ..., 1.0]
```

### LoRA in PEFT
```
W' = W + BA

Where:
• W: frozen base weights (d × d)
• B: (d × r) trainable
• A: (r × d) trainable
• r << d (rank)
```

### TensorRT Optimization
```
Layer fusion: Conv + BN + ReLU → single kernel
Precision calibration: Find optimal scales for INT8
Memory optimization: Workspace allocation
```

---

## 📂 Topics

| File | Tool | Key Features |
|------|------|--------------|

---

## 📊 Tools Overview

| Tool | Quantization | Pruning | Distillation | PEFT |
|------|-------------|---------|--------------|------|
| **PyTorch** | ✅ | ✅ | ✅ | ❌ |
| **HF PEFT** | ❌ | ❌ | ❌ | ✅ |
| **bitsandbytes** | ✅ | ❌ | ❌ | ❌ |
| **TensorRT** | ✅ | ✅ | ❌ | ❌ |
| **ONNX Runtime** | ✅ | ❌ | ❌ | ❌ |

---

## 🔥 Most Common Tools

### For LLM Fine-tuning

```python
# QLoRA with Hugging Face
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
)

# Add LoRA
lora_config = LoraConfig(r=16, lora_alpha=32, ...)
model = get_peft_model(model, lora_config)
```

### For Production Inference

```python
# TensorRT for NVIDIA GPUs
import tensorrt as trt

# Convert ONNX to TensorRT
# trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
```

### For CPU Inference

```bash
# llama.cpp for CPU
./main -m llama-7b-q4.gguf -p "Hello" -n 100
```

---

## 🔗 Where This Topic Is Used

| Topic | Tools Used |
|-------|-----------|
| **QLoRA Fine-tuning** | bitsandbytes + PEFT |
| **LLM Inference** | llama.cpp, vLLM, TensorRT-LLM |
| **Mobile Deployment** | TensorFlow Lite, CoreML |
| **Server Deployment** | TensorRT, ONNX Runtime |
| **Edge Devices** | TFLite, OpenVINO |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | HuggingFace PEFT | [Docs](https://huggingface.co/docs/peft) |
| 📖 | bitsandbytes | [GitHub](https://github.com/TimDettmers/bitsandbytes) |
| 📖 | TensorRT | [NVIDIA](https://developer.nvidia.com/tensorrt) |
| 📖 | llama.cpp | [GitHub](https://github.com/ggerganov/llama.cpp) |
| 🇨🇳 | 模型部署工具 | [知乎](https://zhuanlan.zhihu.com/p/628120082) |

---

⬅️ [Back: 09-Compression Pipelines](../09-compression-pipelines/) | ➡️ [Next: 11-Case Studies](../11-case-studies/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
