<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=04 Training&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# ⚡ Training

> **How to train deep learning models**

---

## 🎯 Visual Overview

<img src="./images/training-loop.svg" width="100%">

*Caption: The training loop consists of forward pass (compute predictions and loss), backward pass (compute gradients via backprop), and update (adjust weights). Key components include the optimizer (Adam, SGD), LR scheduler, and regularization techniques.*

---

## 📂 Topics

| Folder | Topic | Key Concepts |
|--------|-------|--------------|
| [optimizers/](./optimizers/) | SGD, Adam | Weight updates |
| [normalization/](./normalization/) | LayerNorm, BatchNorm | Stability |
| [regularization/](./regularization/) | Dropout, weight decay | Generalization |
| [scheduling/](./scheduling/) | Learning rate | Warmup, cosine |

---

## 📐 Training Loop

```python
for epoch in range(epochs):
    for batch in dataloader:
        # Forward
        outputs = model(batch['input'])
        loss = criterion(outputs, batch['target'])
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Update
        optimizer.step()
        scheduler.step()
```

---

## 🔑 Hyperparameters

| Parameter | Typical Value | Impact |
|-----------|---------------|--------|
| Learning rate | 1e-4 to 1e-3 | Critical |
| Batch size | 32 to 512 | Speed vs memory |
| Weight decay | 0.01 to 0.1 | Regularization |
| Warmup steps | 1000-10000 | Stability |

---

## 🔗 Where This Topic Is Used

| Topic | How Training Techniques Are Used |
|-------|----------------------------------|
| **GPT / LLaMA** | AdamW + cosine LR + warmup |
| **BERT** | Adam + linear decay |
| **ResNet** | SGD + momentum + step LR |
| **Stable Diffusion** | AdamW + EMA |
| **Fine-tuning** | Lower LR + weight decay |
| **LoRA** | Train only adapters, freeze base |
| **Distillation** | Teacher-student training |
| **Contrastive Learning** | SimCLR, CLIP training |
| **RLHF** | PPO optimizer for policy |

### Techniques Used In

| Technique | Used By |
|-----------|---------|
| **LayerNorm** | All Transformers (GPT, BERT, LLaMA) |
| **Dropout** | BERT, older CNNs |
| **Weight Decay** | Almost all models |
| **Warmup** | Large models (prevents instability) |
| **Gradient Clipping** | RNNs, large Transformers |
| **Mixed Precision** | All large-scale training |

### Training Recipe by Model Type

| Model | Optimizer | LR Schedule | Regularization |
|-------|-----------|-------------|----------------|
| Transformer | AdamW | Cosine + warmup | Weight decay |
| CNN | SGD + momentum | Step decay | Dropout + WD |
| Fine-tuning | AdamW | Linear decay | Low LR |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Adam Paper | [arXiv](https://arxiv.org/abs/1412.6980) |
| 📄 | AdamW Paper | [arXiv](https://arxiv.org/abs/1711.05101) |
| 📖 | PyTorch Training Loop | [Tutorial](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) |
| 🇨🇳 | 训练技巧总结 | [知乎](https://zhuanlan.zhihu.com/p/32230623) |
| 🇨🇳 | 深度学习训练实战 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88318597) |
| 🇨🇳 | 大模型训练教程 | [B站](https://www.bilibili.com/video/BV1J94y1f7u5) |

---

⬅️ [Back: 03-Architectures](../03-architectures/) | ➡️ [Next: 05-Scaling](../05-scaling/)


---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
