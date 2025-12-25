<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=120&section=header&text=Matrix%20and%20Tensor%20Factorizatio&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
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
Original weight matrix W ∈ ℝ^(d×k):
+-------------------------------------+
|  W: d × k matrix                    |
|  Parameters: d × k                  |
|  Example: 4096 × 4096 = 16M params |
+-------------------------------------+

Low-rank factorization W ≈ BA:
+-------------------------------------+
|  B: d × r matrix                    |
|  A: r × k matrix                    |
|  Parameters: d×r + r×k              |
|  If r = 16: 4096×16 + 16×4096       |
|           = 131K params (100x less!)|
+-------------------------------------+
```

---

## 🔥 LoRA: The Key Innovation

```
Pretrained Model (Frozen):
W₀ ∈ ℝ^(d×k)  [millions of params, frozen]

LoRA Adapter:
ΔW = BA where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k)
              ↑
           r = 8, 16, 32 (rank)

Forward pass:
h = (W₀ + ΔW)x = W₀x + BAx
    ---------    ---   ---
    frozen       trainable (tiny!)

Benefits:
• Train only BA (0.1% of parameters)
• Merge at inference: W = W₀ + BA (no overhead!)
• Switch adapters easily
```

---

## 💻 LoRA Code Example

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# LoRA config
lora_config = LoraConfig(
    r=16,                    # Rank
    lora_alpha=32,           # Scaling
    target_modules=["q_proj", "v_proj"],  # Which layers
    lora_dropout=0.05,
)

# Wrap model with LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# "trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%"
```

---

## 🔗 Where This Topic Is Used

| Topic | How Factorization Is Used |
|-------|--------------------------|
| **LoRA** | Low-rank adapters for fine-tuning |
| **QLoRA** | LoRA + 4-bit quantization |
| **SVD Compression** | Compress dense layers |
| **Vision Models** | Factorize conv weights |
| **Recommendation** | Matrix factorization |
| **NMF** | Non-negative decomposition |

### Prerequisite For

```
Factorization --> LoRA fine-tuning (most popular!)
             --> QLoRA (LoRA + quantization)
             --> Efficient fine-tuning
             --> Understanding weight compression
```

---

## 📐 SVD Mathematics

<img src="./images/svd-math.svg" width="100%">

---

## 📚 References & Resources

### 📄 Key Papers

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|------------------|
| [LoRA](https://arxiv.org/abs/2106.09685) | Hu et al. | 2021 | Low-rank adaptation |
| [QLoRA](https://arxiv.org/abs/2305.14314) | Dettmers et al. | 2023 | Quantized LoRA |
| [Low-rank Matrix Factorization](https://arxiv.org/abs/1312.4659) | Sainath et al. | 2013 | SVD for DNNs |
| [Tensor Decomposition](https://arxiv.org/abs/1511.06530) | Lebedev et al. | 2015 | CP decomposition for CNNs |
| [Tucker Decomposition](https://arxiv.org/abs/1412.6553) | Kim et al. | 2015 | Tucker for compression |
| [AdaLoRA](https://arxiv.org/abs/2303.10512) | Zhang et al. | 2023 | Adaptive rank |
| 🇨🇳 SVD分解详解 | [知乎](https://zhuanlan.zhihu.com/p/29846048) | - | 奇异值分解原理 |
| 🇨🇳 矩阵分解压缩 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88785898) | - | 低秩分解应用 |
| 🇨🇳 线性代数SVD | [B站](https://www.bilibili.com/video/BV1ys411472E) | - | 3B1B风格讲解 |

### 📐 Mathematical Background

| Concept | Description |
|---------|-------------|
| SVD | W = UΣVᵀ, optimal low-rank |
| Eckart-Young | Truncated SVD is optimal |
| Intrinsic Dimension | NNs have low effective rank |

### 🛠️ Tools

| Tool | Description | Link |
|------|-------------|------|
| PEFT Library | LoRA implementation | [GitHub](https://github.com/huggingface/peft) |
| torch.linalg.svd | PyTorch SVD | [Docs](https://pytorch.org/docs/stable/generated/torch.linalg.svd.html) |

---

⬅️ [Back: 04-Knowledge Distillation](../04-knowledge-distillation/) | ➡️ [Next: 06-Sparsity](../06-sparsity/)


---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=80&section=footer" width="100%"/>
</p>
