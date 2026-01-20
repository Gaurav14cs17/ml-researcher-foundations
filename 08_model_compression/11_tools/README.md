<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=120&section=header&text=Tools%20and%20Frameworks&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-08-E74C3C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

<p align="center">
<img src="./images/tools-overview.svg" width="100%">
</p>

## 📐 Mathematical Foundations (Tool-Specific)

### 1. bitsandbytes NF4 Quantization

**NormalFloat4 (NF4):**
Quantization levels optimized for zero-mean, unit-variance normal distribution:

```math
q_i = \Phi^{-1}\left(\frac{2i + 1}{32}\right), \quad i \in \{0, 1, ..., 15\}
```

Where $\Phi^{-1}$ is inverse normal CDF.

**Values:** $\{-1.0, -0.7, ..., 0, ..., 0.7, 1.0\}$ (approximately)

**Double Quantization:**
Quantize the quantization constants themselves:

```math
\text{FP32 scale} \to \text{FP8 scale}
```

Additional 0.5 bits/param savings.

### 2. GPTQ Algorithm

**Layer-wise Quantization:**

```math
\arg\min_{\hat{W}} \|WX - \hat{W}X\|_2^2
```

**Optimal Brain Quantization (OBQ):**

```math
\hat{w}_q = \text{round}(w_q / s) \cdot s
\delta_{-q} = -\frac{w_q - \hat{w}_q}{[H^{-1}]_{qq}} H^{-1}_{:,q}
```

Where $H = XX^T$ (Hessian).

**GPTQ Optimization:** Process columns in order with lazy batch updates.

### 3. AWQ Importance Metric

**Activation-Aware Importance:**

```math
s_j = \mathbb{E}[|X_j|]
```

**Per-channel Scaling:**

```math
\hat{W}[:,j] = W[:,j] \cdot s_j^\alpha
\hat{X}[j] = X[j] / s_j^\alpha
```

With $\alpha \in [0, 1]$ balancing weight and activation quantization.

### 4. TensorRT Optimization

**Layer Fusion:**

```math
\text{Conv} \to \text{BN} \to \text{ReLU} \Rightarrow \text{ConvBNReLU}
```

**Kernel Auto-tuning:**
Select best CUDA kernel for each layer based on shape.

**INT8 Calibration:**

```math
s = \frac{\max(|x|)}{127} \quad \text{(per-tensor)}
```

---

## 📊 Tools Overview

| Tool | Quantization | Pruning | PEFT | LLM Inference |
|------|-------------|---------|------|---------------|
| **PyTorch** | ✅ PTQ, QAT | ✅ | ❌ | ❌ |
| **HF PEFT** | ❌ | ❌ | ✅ | ❌ |
| **bitsandbytes** | ✅ 4/8-bit | ❌ | ❌ | ✅ |
| **AutoGPTQ** | ✅ GPTQ | ❌ | ❌ | ✅ |
| **AutoAWQ** | ✅ AWQ | ❌ | ❌ | ✅ |
| **llama.cpp** | ✅ GGML | ❌ | ❌ | ✅ CPU |
| **vLLM** | ✅ | ❌ | ❌ | ✅ GPU |
| **TensorRT** | ✅ INT8 | ✅ | ❌ | ✅ NVIDIA |

---

## 🔥 Most Common Tools by Use Case

### LLM Fine-tuning (QLoRA)

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 4-bit quantization with bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# Add LoRA with PEFT
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_config)
```

### LLM Inference (GPTQ/AWQ)

```python

# GPTQ with AutoGPTQ
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-GPTQ",
    device="cuda:0",
    use_triton=True,  # Fast GPTQ kernels
)

# AWQ with AutoAWQ
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf"
)
model.quantize(
    tokenizer,
    quant_config={"w_bit": 4, "q_group_size": 128}
)
```

### CPU Inference (llama.cpp)

```bash

# Quantize to GGUF
./quantize model-fp16.gguf model-q4_k_m.gguf Q4_K_M

# Run inference
./main -m model-q4_k_m.gguf \
       -p "Once upon a time" \
       -n 256 \
       -t 8 \
       --ctx-size 4096
```

### High-Throughput GPU (vLLM)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=1,
    quantization="awq",
    dtype="float16",
)

# Continuous batching + PagedAttention
outputs = llm.generate(
    prompts=["Hello", "How are you"],
    sampling_params=SamplingParams(max_tokens=100)
)
```

### Production (TensorRT-LLM)

```python

# Build TensorRT engine
from tensorrt_llm import LLaMAForCausalLM

model = LLaMAForCausalLM.from_hugging_face(
    "meta-llama/Llama-2-7b-hf",
    dtype="float16",
    quantization="int8_sq",  # SmoothQuant INT8
)

# Deploy with Triton Inference Server
```

---

## 📊 Quantization Format Comparison

| Format | Tool | GPU | CPU | Quality | Size |
|--------|------|-----|-----|---------|------|
| **NF4** | bitsandbytes | ✅ | ❌ | Good | 4.5-bit |
| **GPTQ** | AutoGPTQ | ✅ | ❌ | Good | 4-bit |
| **AWQ** | AutoAWQ | ✅ | ❌ | Better | 4-bit |
| **GGML Q4_K_M** | llama.cpp | ❌ | ✅ | Good | 4.5-bit |
| **INT8** | TensorRT | ✅ | ✅ | Best | 8-bit |

---

## 🛠️ Installation Guide

```bash

# bitsandbytes (CUDA required)
pip install bitsandbytes

# PEFT
pip install peft

# AutoGPTQ
pip install auto-gptq

# AutoAWQ
pip install autoawq

# vLLM
pip install vllm

# llama.cpp (build from source)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j
```

---

## 📚 References

| Tool | Documentation | GitHub |
|------|--------------|--------|
| **PEFT** | [Docs](https://huggingface.co/docs/peft) | [GitHub](https://github.com/huggingface/peft) |
| **bitsandbytes** | [Docs](https://huggingface.co/docs/bitsandbytes) | [GitHub](https://github.com/TimDettmers/bitsandbytes) |
| **AutoGPTQ** | - | [GitHub](https://github.com/PanQiWei/AutoGPTQ) |
| **AutoAWQ** | - | [GitHub](https://github.com/casper-hansen/AutoAWQ) |
| **llama.cpp** | [Docs](https://github.com/ggerganov/llama.cpp/blob/master/README.md) | [GitHub](https://github.com/ggerganov/llama.cpp) |
| **vLLM** | [Docs](https://vllm.readthedocs.io/) | [GitHub](https://github.com/vllm-project/vllm) |
| **TensorRT** | [Docs](https://developer.nvidia.com/tensorrt) | [GitHub](https://github.com/NVIDIA/TensorRT) |

---

⬅️ [Back: Compression Pipelines](../10_compression_pipelines/README.md) | ➡️ [Next: Case Studies](../12_case_studies/README.md)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=80&section=footer" width="100%"/>
</p>
