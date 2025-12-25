<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=LoRA%20Low-Rank%20Adaptation&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 The Problem

Fine-tuning large language models is expensive:

```
Model: LLaMA 70B → 70 billion parameters
Full fine-tuning:
  • Memory: ~280 GB (fp16)
  • Time: Days on expensive GPUs
  • Storage: 280 GB per task
  • Cost: $1000s per adapter

For 1000 tasks → 280 TB storage + $1M+ cost
```

---

## 💡 Key Insight

**Observation:** Weight updates during fine-tuning have low intrinsic rank!

```
Full fine-tuning: W → W + ΔW

Where ΔW is approximately low-rank:
ΔW ≈ B · A

Where:
  B ∈ ℝᵈˣʳ  (d = model dim, r = rank)
  A ∈ ℝʳˣᵏ  (k = hidden dim)
  r << min(d, k)  (typically r = 1-64)
```

---

## 📐 Method

### Architecture

```
Original layer: y = Wx

With LoRA:     y = Wx + (α/r)·BAx
               
Where:
  • W: Frozen pretrained weights
  • B, A: Trainable low-rank matrices
  • α: Scaling hyperparameter
  • r: Rank (controls # parameters)
```

### Visualization

```
Input x (k-dim)
     ↓
     ├─→ Wx (frozen)     ─┐
     │                    ├─→ + → output
     └─→ BAx (trainable) ─┘
         ↓        ↓
        A(rxk)  B(dxr)
        
Parameters:
  Original: d × k
  LoRA:     r(d + k)
```

---

## 📊 Parameter Reduction

### Example: Transformer Layer

```python
# Self-attention: Q, K, V, O projections
d_model = 4096
d_inner = 4096
r = 8

# Full fine-tuning parameters:
full_params = 4 * (d_model * d_inner)  # 67M parameters

# LoRA parameters:
lora_params = 4 * r * (d_model + d_inner)  # 262K parameters

# Reduction: 67M / 262K = 256x
```

### Scaling to Large Models

| Model | Full Params | LoRA (r=8) | Reduction |
|-------|-------------|------------|-----------|
| BERT-Base | 110M | 294K | 374x |
| GPT-2 | 1.5B | 3.5M | 428x |
| LLaMA-7B | 7B | 4.2M | 1,666x |
| LLaMA-70B | 70B | 67M | 1,044x |

---

## 💻 Implementation

### From Scratch

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16
    ):
        super().__init__()
        self.original = original_layer
        self.original.weight.requires_grad = False  # Freeze
        
        d_in = original_layer.in_features
        d_out = original_layer.out_features
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        
        self.scaling = alpha / rank
    
    def forward(self, x):
        # Original output (frozen)
        result = self.original(x)
        
        # Add low-rank adaptation
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        result += self.scaling * lora_out
        
        return result

# Usage
original_linear = nn.Linear(512, 512)
lora_layer = LoRALayer(original_linear, rank=8)
```

### Using PEFT Library

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16
)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                          # Rank
    lora_alpha=16,                # Scaling
    lora_dropout=0.1,
    target_modules=[              # Which layers to adapt
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
)

# Wrap model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 4.2M || all params: 6.7B || trainable%: 0.063%

# Train normally!
# ...

# Save only LoRA weights (tiny!)
model.save_pretrained("./lora_adapter")  # ~17 MB vs 13 GB
```

---

## 🔑 Connection to Linear Algebra

### SVD Perspective

```
Any matrix can be decomposed:
W = UΣVᵀ  (SVD)

Low-rank approximation:
W ≈ Uᵣ Σᵣ Vᵣᵀ
  = (Uᵣ Σᵣ) · Vᵣᵀ
  = B · A

LoRA learns the "important directions" for adaptation!
```

### Why Low-Rank Works

```
Hypothesis: Fine-tuning adjusts a small subspace

Evidence:
  • Top r=8 singular values capture most variance
  • Downstream tasks are "nearby" in parameter space
  • Empirical: r=8-16 matches full fine-tuning
```

---

## 🌍 Variants and Extensions

### QLoRA (2023)
```python
from transformers import BitsAndBytesConfig

# 4-bit base model + LoRA adapters
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,  # 70B → 35 GB!
    device_map="auto"
)

# Add LoRA on top
model = get_peft_model(model, lora_config)
```

### AdaLoRA (2023)
- Adaptive rank allocation
- Different ranks for different layers
- Prunes less important ranks during training

### DoRA (2024)
- Decomposes into magnitude + direction
- `W + ΔW = ||W|| · (W/||W|| + Δd)`
- Better performance than LoRA

---

## 📊 Performance

### Quality Comparison

| Model | Method | GLUE | SQuAD | Params |
|-------|--------|------|-------|--------|
| BERT | Full FT | 84.5 | 88.3 | 110M |
| BERT | LoRA (r=8) | 84.4 | 88.1 | 294K |
| GPT-3 | Full FT | - | - | 175B |
| GPT-3 | LoRA (r=4) | Same | Same | 4.7M |

**Insight:** LoRA matches full fine-tuning with 0.01% parameters!

---

## 🚀 Practical Applications

### Multi-Task Learning
```
Base model: LLaMA-7B (13 GB)

Task 1: Code generation     + adapter_1 (17 MB)
Task 2: Math reasoning      + adapter_2 (17 MB)
Task 3: Medical QA          + adapter_3 (17 MB)
...
Task 1000                   + adapter_1000 (17 MB)

Total: 13 GB + 17 GB adapters
vs. 13 TB for full fine-tuning!
```

### Personal AI
```
Single base model shared
+ Personal adapter for your:
  - Writing style
  - Domain knowledge
  - Preferences
  
Storage: 17 MB per person
vs. 13 GB per person
```

---

## 📖 Detailed Content

[→ LoRA Technical Details](./lora.md)

---

## 📚 Resources

### Papers
- **LoRA** (Hu et al., ICLR 2022)
- **QLoRA** (Dettmers et al., NeurIPS 2023)
- **AdaLoRA** (Zhang et al., ICLR 2023)
- **DoRA** (Liu et al., 2024)

### Libraries
- **PEFT** (Hugging Face): `pip install peft`
- **bitsandbytes**: Quantization for QLoRA
- **Axolotl**: Training framework with LoRA

### Tutorials
- Hugging Face: PEFT documentation
- Lightning AI: Fine-tuning with LoRA
- Weights & Biases: LoRA experiments

---

⬅️ [Back: Flash Attention](../flash-attention/)


---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
