<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=11 Case Studies&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 📱 Case Studies and Applications

> **Real-world model compression examples**

<img src="./images/case-studies.svg" width="100%">

---

## 📐 Mathematical Foundations

### Distillation Loss (DistilBERT)
```
L = α × L_CE(y, student) + (1-α) × L_KL(teacher, student)

Where:
• L_CE = cross-entropy with true labels
• L_KL = KL divergence with teacher
• α = balancing factor (typically 0.5)
```

### QLoRA Savings
```
Full fine-tuning: All P params updated
LoRA: Only 2rd params updated

Savings: P / (2rd) = d / (2r)
For d=4096, r=16: 128x fewer params!
```

### Compression Metrics
```
Size reduction: original_size / compressed_size
Speedup: original_latency / compressed_latency
Accuracy retention: compressed_acc / original_acc

Quality metric: speedup × accuracy_retention
```

---

## 📂 Topics

| File | Domain | Examples |
|------|--------|----------|

---

## 🔥 LLM Compression (Most Important)

```
Running LLaMA-70B:
+-- Original: 140 GB (needs 4x A100)
+-- INT8: 70 GB (needs 2x A100)
+-- INT4: 35 GB (fits 1x A100!)
+-- QLoRA: 35 GB base + tiny adapters

Fine-tuning LLaMA-7B:
+-- Full: 28+ GB GPU, $$$
+-- LoRA: 8 GB GPU, $
+-- QLoRA: 4 GB GPU (laptop!), ¢
```

---

## 📊 Famous Compressed Models

### Language Models

| Model | Original | Compressed | Method |
|-------|----------|------------|--------|
| BERT | 340M | DistilBERT 66M | Distillation |
| GPT-3 | 175B | Alpaca/Vicuna 7B | Distillation |
| LLaMA | FP16 | GGML Q4 | Quantization |
| Any LLM | 7B params | LoRA 4M params | PEFT |

### Vision Models

| Model | Original | Compressed | Method |
|-------|----------|------------|--------|
| ResNet-50 | 25M | MobileNetV3 5M | Architecture |
| ImageNet models | FP32 | INT8 | Quantization |
| CLIP | 400M | TinyCLIP 19M | Distillation |

### Speech Models

| Model | Original | Compressed | Method |
|-------|----------|------------|--------|
| Whisper | 1.5B | Distil-Whisper 750M | Distillation |
| Wav2Vec | 300M | Quantized | Quantization |

---

## 🔗 Where This Topic Is Used

| Application | Compression Used |
|-------------|-----------------|
| **ChatGPT API** | Server-side quantization |
| **Local LLM (Ollama)** | GGML quantization |
| **Mobile Vision** | MobileNet architecture |
| **Voice Assistants** | Compressed speech models |
| **Edge AI** | All techniques combined |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | DistilBERT | [arXiv](https://arxiv.org/abs/1910.01108) |
| 📄 | MobileNetV3 | [arXiv](https://arxiv.org/abs/1905.02244) |
| 📄 | Distil-Whisper | [arXiv](https://arxiv.org/abs/2311.00430) |
| 🇨🇳 | 模型压缩案例 | [知乎](https://zhuanlan.zhihu.com/p/628120082) |
| 🇨🇳 | 实战部署 | [机器之心](https://www.jiqizhixin.com/) |

---

⬅️ [Back: 10-Tools](../10-tools/) | ➡️ [Next: 12-Future Directions](../12-future-directions/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
