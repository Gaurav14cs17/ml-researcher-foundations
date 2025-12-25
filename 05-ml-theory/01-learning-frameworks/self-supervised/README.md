<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Self-Supervised%20Learning&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/self-supervised.svg" width="100%">

*Caption: Self-supervised learning creates labels from the data itself via pretext tasks: masked prediction (BERT), contrastive learning (SimCLR, CLIP), or next-token prediction (GPT). Foundation of modern AI.*

---

## 📐 Key Idea

```
No labels needed! Create pseudo-labels from data:

Contrastive: Same image = similar, different = dissimilar
Masked: Predict hidden parts (BERT, MAE)
Predictive: Predict next token (GPT)
```

---

## 📊 Methods

| Method | Pretext Task | Example |
|--------|--------------|---------|
| **Contrastive** | Same image augmentations similar | SimCLR, MoCo |
| **Masked Language** | Predict [MASK] tokens | BERT |
| **Masked Image** | Predict masked patches | MAE |
| **Next Token** | Predict next token | GPT |

---

## 💻 Contrastive Example

```python
# SimCLR style
def contrastive_loss(z1, z2, temperature=0.5):
    """z1, z2 are embeddings of augmented views"""
    z = torch.cat([z1, z2])
    sim = F.cosine_similarity(z.unsqueeze(0), z.unsqueeze(1), dim=2)
    sim = sim / temperature
    
    # Positive pairs: (i, i+batch_size)
    labels = torch.arange(batch_size)
    labels = torch.cat([labels + batch_size, labels])
    
    return F.cross_entropy(sim, labels)
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | SimCLR Paper | [arXiv](https://arxiv.org/abs/2002.05709) |
| 📄 | BERT Paper | [arXiv](https://arxiv.org/abs/1810.04805) |
| 📄 | MAE Paper | [arXiv](https://arxiv.org/abs/2111.06377) |
| 🇨🇳 | 自监督学习综述 | [知乎](https://zhuanlan.zhihu.com/p/381354026) |
| 🇨🇳 | 对比学习详解 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | BERT/GPT预训练 | [B站](https://www.bilibili.com/video/BV1J94y1f7u5) |


## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

⬅️ [Back: Learning Frameworks](../)

---

➡️ [Next: Supervised](../supervised/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
