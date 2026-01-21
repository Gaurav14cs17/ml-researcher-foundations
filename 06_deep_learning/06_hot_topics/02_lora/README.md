<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=LoRA%20Low-Rank%20Adaptation&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
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

## 📐 Mathematical Foundation

### Key Insight

**Observation:** Weight updates during fine-tuning have low intrinsic rank!

$$\text{Full fine-tuning: } W \rightarrow W + \Delta W$$

**Hypothesis:** $\Delta W$ can be approximated by low-rank matrices:

$$\Delta W \approx BA$$

Where:

- $B \in \mathbb{R}^{d \times r}$ (down-projection)

- $A \in \mathbb{R}^{r \times k}$ (up-projection)

- $r \ll \min(d, k)$ (typically $r = 1-64$)

### LoRA Formulation

**Original layer:**

$$y = Wx$$

**With LoRA:**

$$y = Wx + \frac{\alpha}{r} \cdot BAx$$

Where:

- $W \in \mathbb{R}^{d \times k}$: Frozen pretrained weights

- $B \in \mathbb{R}^{d \times r}$: Trainable down-projection

- $A \in \mathbb{R}^{r \times k}$: Trainable up-projection

- $\alpha$: Scaling hyperparameter

- $r$: Rank (controls parameter count)

### Architecture Visualization

```
Input x (k-dim)
     ↓
     +-→ Wx (frozen, d×k)     -+
     |                          +-→ + → output (d-dim)
     +-→ (α/r)·BAx (trainable) -+
         ↓        ↓
        A(r×k)  B(d×r)

```

---

## 🔬 Parameter Reduction Analysis

### Count Comparison

| Component | Full Fine-tuning | LoRA (rank r) |
|-----------|------------------|---------------|
| Single layer | $d \times k$ | $r(d + k)$ |
| Reduction factor | 1 | $\frac{dk}{r(d+k)}$ |

**Example: Transformer Layer**

```python
d_model = 4096
d_inner = 4096
r = 8

# Q, K, V, O projections
full_params = 4 * (d_model * d_inner)  # 67M parameters

# LoRA parameters
lora_params = 4 * r * (d_model + d_inner)  # 262K parameters

# Reduction: 67M / 262K = 256x fewer parameters!

```

### Scaling to Large Models

| Model | Full Params | LoRA (r=8) | Reduction |
|-------|-------------|------------|-----------|
| BERT-Base | 110M | 294K | 374x |
| GPT-2 | 1.5B | 3.5M | 428x |
| LLaMA-7B | 7B | 4.2M | 1,666x |
| LLaMA-70B | 70B | 67M | 1,044x |

---

## 🔬 Connection to Linear Algebra

### SVD Perspective

Any matrix can be decomposed:

$$W = U\Sigma V^\top \quad \text{(SVD)}$$

Low-rank approximation:

$$W \approx U_r \Sigma_r V_r^\top = (U_r \Sigma_r) \cdot V_r^\top = B \cdot A$$

**Interpretation:** LoRA learns the "important directions" for adaptation.

### Eckart-Young Theorem

The best rank-$r$ approximation of $\Delta W$ is:

$$\Delta W_r = \sum_{i=1}^{r} \sigma_i u_i v_i^\top$$

LoRA learns $B$ and $A$ such that $BA \approx \Delta W_r$.

### Why Low-Rank Works

**Hypothesis:** Fine-tuning adjusts weights in a low-dimensional subspace.

**Evidence:**

1. Top $r=8$ singular values capture most variance of $\Delta W$

2. Downstream tasks are "nearby" in parameter space

3. Empirical: $r=8-16$ matches full fine-tuning performance

---

## 📐 Initialization and Scaling

### Initialization Strategy

**For $A$:** Gaussian initialization

$$A \sim \mathcal{N}(0, \sigma^2)$$

**For $B$:** Zero initialization

$$B = 0$$

**Reason:** At initialization, $\Delta W = BA = 0$, so we start from the pretrained weights.

### Scaling Factor

$$\text{scaling} = \frac{\alpha}{r}$$

**Intuition:**
- $\alpha$ is the "learning rate multiplier" for LoRA

- Dividing by $r$ normalizes for different rank choices

- Typical: $\alpha = 16$ or $\alpha = 2r$

### Gradient Analysis

**Gradients w.r.t. LoRA parameters:**

$$\frac{\partial L}{\partial A} = \frac{\alpha}{r} \cdot B^\top \frac{\partial L}{\partial y} x^\top
\frac{\partial L}{\partial B} = \frac{\alpha}{r} \cdot \frac{\partial L}{\partial y} (Ax)^\top$$

The scaling ensures gradients are well-behaved regardless of rank.

---

## 💻 Implementation

### From Scratch

```python
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    """LoRA: Low-Rank Adaptation layer"""
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Store reference to original (frozen) layer
        self.original = original_layer
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        
        d_in = original_layer.in_features
        d_out = original_layer.out_features
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * math.sqrt(2.0 / d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        
        # Scaling and dropout
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        # Original output (frozen)
        result = self.original(x)
        
        # LoRA path: x → A → B → scale
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        result = result + self.scaling * lora_out
        
        return result
    
    def merge_weights(self):
        """Merge LoRA weights into original for inference efficiency"""
        with torch.no_grad():
            self.original.weight.data += self.scaling * (self.lora_B @ self.lora_A)
    
    def unmerge_weights(self):
        """Unmerge LoRA weights (for continued training)"""
        with torch.no_grad():
            self.original.weight.data -= self.scaling * (self.lora_B @ self.lora_A)

def add_lora_to_model(model, target_modules, rank=8, alpha=16):
    """Add LoRA to specified modules in a model"""
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Replace with LoRA layer
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                
                lora_layer = LoRALayer(module, rank=rank, alpha=alpha)
                setattr(parent, child_name, lora_layer)
    
    # Freeze all non-LoRA parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze LoRA parameters
    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
    
    return model

```

### Using PEFT Library

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                          # Rank
    lora_alpha=16,                # Scaling (alpha/r applied internally)
    lora_dropout=0.1,
    target_modules=[              # Which layers to adapt
        "q_proj",                 # Query projection
        "k_proj",                 # Key projection
        "v_proj",                 # Value projection
        "o_proj",                 # Output projection
        "gate_proj",              # MLP gate
        "up_proj",                # MLP up
        "down_proj"               # MLP down
    ],
    bias="none"                   # Don't train biases
)

# Wrap model with LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 4.2M || all params: 6.7B || trainable%: 0.063%

# Train normally
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# ... training loop ...

# Save only LoRA weights (tiny!)
model.save_pretrained("./lora_adapter")  # ~17 MB vs 13 GB

# Load adapter later
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base_model, "./lora_adapter")

```

### QLoRA: 4-bit Quantization + LoRA

```python
from transformers import BitsAndBytesConfig

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # Normalized float 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True       # Nested quantization
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,  # 70B → ~35 GB!
    device_map="auto"
)

# Add LoRA on top of quantized model
model = get_peft_model(model, lora_config)
# Now trainable with 24GB GPU!

```

---

## 🌍 Variants and Extensions

### AdaLoRA (2023)

- **Adaptive rank allocation:** Different ranks for different layers

- Prunes less important singular values during training

- Better performance with same parameter budget

### DoRA (2024)

- **Decomposed LoRA:** Separate magnitude and direction

$$W + \Delta W = m \cdot \frac{W + BA}{\|W + BA\|}$$

- Learns magnitude $m$ and directional update $BA$

- Often outperforms standard LoRA

### LoRA+ (2024)

- Different learning rates for A and B matrices

- Typically: $\text{lr}_B = \lambda \cdot \text{lr}_A$ with $\lambda \approx 16$

- Faster convergence

### rsLoRA (2024)

- Rank-stabilized scaling: $\alpha = r$ instead of constant

- More stable across different rank choices

---

## 📊 Performance Comparison

| Model | Method | GLUE | SQuAD | Parameters |
|-------|--------|------|-------|------------|
| BERT | Full FT | 84.5 | 88.3 | 110M |
| BERT | LoRA (r=8) | 84.4 | 88.1 | 294K |
| GPT-3 | Full FT | - | - | 175B |
| GPT-3 | LoRA (r=4) | Same | Same | 4.7M |

**Key insight:** LoRA matches full fine-tuning with 0.01% parameters!

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

Total: 13 GB + 17 GB adapters = 30 GB
vs. 13 TB for full fine-tuning (1000 × 13 GB)

```

### Serving Multiple Adapters

```python
# Load base model once
base_model = load_model("llama-7b")

# Hot-swap adapters based on user/task
adapters = {
    "code": load_adapter("code_adapter"),
    "math": load_adapter("math_adapter"),
    "chat": load_adapter("chat_adapter")
}

def inference(prompt, task):
    adapter = adapters[task]
    # Apply adapter (just add BA to forward pass)
    output = base_model(prompt, adapter=adapter)
    return output

```

### Efficient Batching

With base model shared:

- Different requests can use different adapters

- Batch together requests with same adapter

- Base model weights shared across all

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | LoRA Paper | [arXiv](https://arxiv.org/abs/2106.09685) |
| 📄 | QLoRA Paper | [arXiv](https://arxiv.org/abs/2305.14314) |
| 📄 | AdaLoRA Paper | [arXiv](https://arxiv.org/abs/2303.10512) |
| 📄 | DoRA Paper | [arXiv](https://arxiv.org/abs/2402.09353) |
| 💻 | PEFT Library | [GitHub](https://github.com/huggingface/peft) |
| 📖 | HuggingFace Docs | [Docs](https://huggingface.co/docs/peft) |
| 🇨🇳 | LoRA原理详解 | [知乎](https://zhuanlan.zhihu.com/p/631077612) |
| 🇨🇳 | QLoRA解读 | [CSDN](https://blog.csdn.net/v_JULY_v/article/details/132116988) |
| 🇨🇳 | 大模型微调技术 | [B站](https://www.bilibili.com/video/BV1Mv4y1Y7E3) |

---

⬅️ [Back: Flash Attention](../01_flash_attention/README.md)

---

⬅️ [Back: Hot Topics](../../README.md)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
