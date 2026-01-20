<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=120&section=header&text=Case%20Studies&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-08-E74C3C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

<p align="center">
<img src="./images/case-studies.svg" width="100%">
</p>

## 📐 Mathematical Analysis of Real Models

### 1. DistilBERT (Distillation)

**Compression Details:**

$$
\text{Teacher: BERT-Base} = 110M \text{ params}
\text{Student: DistilBERT} = 66M \text{ params}
$$

**Distillation Loss:**

$$
\mathcal{L} = \alpha \mathcal{L}_{CE} + (1-\alpha) T^2 \mathcal{L}_{KL} + \beta \mathcal{L}_{cos}
$$

Where $\mathcal{L}\_{cos}$ is cosine embedding loss for hidden states.

**Results:**
- Size: 40% smaller (110M → 66M)
- Speed: 60% faster
- GLUE Score: 97% of BERT-Base

### 2. QLoRA Fine-tuning Analysis

**Memory Comparison (LLaMA-7B):**

| Method | Model | Optimizer | Gradient | Total |
|--------|-------|-----------|----------|-------|
| Full FT | 14GB | 28GB | 14GB | 56GB |
| LoRA | 14GB | 16MB | 16MB | 14GB |
| QLoRA | 3.5GB | 8MB | 8MB | 3.5GB |

**Accuracy (Alpaca benchmark):**
- Full fine-tune: 88.2%
- LoRA: 87.9%
- QLoRA: 87.6%

### 3. Mixtral MoE Efficiency

**Architecture:**
- Total params: 46.7B
- Active params: 12.9B (per token)
- 8 experts, top-2 routing

**Efficiency Analysis:**

$$
\text{Compute Savings} = \frac{N_{total}}{N_{active}} = \frac{46.7B}{12.9B} = 3.6\times
$$

**Quality vs LLaMA:**
| Model | Params (Active) | MMLU | HumanEval |
|-------|-----------------|------|-----------|
| LLaMA-70B | 70B | 68.9 | 37.8 |
| Mixtral-8x7B | 12.9B | 70.6 | 40.2 |

Same quality, 5× fewer active params!

### 4. MobileNet Efficiency

**Depthwise Separable Analysis:**

$$
\frac{\text{Standard Conv}}{\text{DW Separable}} = \frac{D_K^2 \cdot M \cdot N}{D_K^2 \cdot M + M \cdot N} = \frac{1}{1/N + 1/D_K^2}
$$

For $D\_K=3$, $N=256$: $\frac{1}{1/256 + 1/9} \approx 8.2\times$

**MobileNetV3 vs ResNet-50:**
| Model | Params | FLOPs | Top-1 |
|-------|--------|-------|-------|
| ResNet-50 | 25.6M | 4.1B | 76.0% |
| MobileNetV3-L | 5.4M | 0.22B | 75.2% |
| Ratio | 4.7× | 18.6× | 99% |

---

## 🔥 LLM Compression (Most Important)

```
Running LLaMA-70B:
+-- FP32: 280 GB (impossible on consumer HW)
+-- FP16: 140 GB (needs 4× A100 80GB)
+-- INT8: 70 GB (needs 2× A100)
+-- INT4: 35 GB (fits 1× A100!)
+-- GGML Q4: ~35 GB (runs on CPU!)

Fine-tuning Costs:
+-- Full FT LLaMA-7B: ~1000 USD (A100 rental)
+-- LoRA: ~100 USD (A6000)
+-- QLoRA: ~10 USD (RTX 3090)
```

---

## 📊 Famous Compressed Models

### Language Models

| Original | Compressed | Method | Size | Quality |
|----------|------------|--------|------|---------|
| BERT-Base | DistilBERT | Distillation | 40%↓ | 97% |
| BERT-Base | TinyBERT | Distillation | 87%↓ | 96% |
| BERT-Base | MobileBERT | Architecture | 77%↓ | 99% |
| GPT-3 175B | Alpaca 7B | Distillation | 96%↓ | ~80% |
| LLaMA FP16 | GGML Q4 | Quantization | 75%↓ | ~99% |

### Vision Models

| Original | Compressed | Method | Size | Quality |
|----------|------------|--------|------|---------|
| ResNet-50 | MobileNetV3 | Architecture | 79%↓ | 99% |
| VGG-16 | SqueezeNet | Architecture | 98%↓ | 95% |
| EfficientNet-B7 | EfficientNet-B0 | Scaling | 92%↓ | 91% |
| CLIP 400M | TinyCLIP 19M | Distillation | 95%↓ | 93% |

### Speech Models

| Original | Compressed | Method | Size | Quality |
|----------|------------|--------|------|---------|
| Whisper Large | Distil-Whisper | Distillation | 50%↓ | 98% |
| Wav2Vec 300M | Wav2Vec-Small | Architecture | 70%↓ | 95% |

---

## 💻 Reproducing Key Results

### DistilBERT Training

```python
from transformers import (
    DistilBertConfig, DistilBertForMaskedLM,
    BertForMaskedLM, Trainer, TrainingArguments
)

# Teacher
teacher = BertForMaskedLM.from_pretrained("bert-base-uncased")

# Student (smaller architecture)
config = DistilBertConfig(
    vocab_size=30522,
    n_layers=6,          # 12 → 6
    n_heads=12,
    dim=768,
    hidden_dim=3072,
)
student = DistilBertForMaskedLM(config)

# Distillation training
class DistillationTrainer(Trainer):
    def __init__(self, teacher, temperature=4.0, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher
        self.T = temperature
        self.alpha = alpha
    
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        student_logits = outputs.logits
        
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            teacher_logits = teacher_outputs.logits
        
        # Distillation loss
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=-1),
            F.softmax(teacher_logits / self.T, dim=-1),
            reduction='batchmean'
        ) * self.T ** 2
        
        hard_loss = outputs.loss
        
        loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        return (loss, outputs) if return_outputs else loss
```

### QLoRA Fine-tuning

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Load 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
    device_map="auto",
)

# Add LoRA
model = get_peft_model(model, LoraConfig(
    r=64, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
))

# Train
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    max_seq_length=512,
)
trainer.train()

# Achieves ~99% of full fine-tune quality!
```

---

## 📚 References

| Model | Paper | Link |
|-------|-------|------|
| DistilBERT | Sanh et al. 2019 | [arXiv](https://arxiv.org/abs/1910.01108) |
| TinyBERT | Jiao et al. 2019 | [arXiv](https://arxiv.org/abs/1909.10351) |
| MobileNetV3 | Howard et al. 2019 | [arXiv](https://arxiv.org/abs/1905.02244) |
| Mixtral | Mistral AI 2024 | [arXiv](https://arxiv.org/abs/2401.04088) |
| QLoRA | Dettmers et al. 2023 | [arXiv](https://arxiv.org/abs/2305.14314) |

---

⬅️ [Back: Tools](../11_tools/README.md) | ➡️ [Next: Future Directions](../13_future_directions/README.md)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=80&section=footer" width="100%"/>
</p>
