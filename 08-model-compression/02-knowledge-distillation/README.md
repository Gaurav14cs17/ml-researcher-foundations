<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=120&section=header&text=Knowledge%20Distillation&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-08-E74C3C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Distillation Loss

```
L = α × L_CE(y, p_student) + (1-α) × T² × KL(p_teacher/T, p_student/T)

Where:
- T = temperature (softens probabilities)
- α = balance between hard and soft targets
```

---

## 💻 Code

```python
def distillation_loss(student_logits, teacher_logits, labels, T=4, alpha=0.7):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * T * T
    
    hard_loss = F.cross_entropy(student_logits, labels)
    
    return alpha * hard_loss + (1 - alpha) * soft_loss
```

---

## 🔗 Examples

| Teacher | Student | Task |
|---------|---------|------|
| **BERT-Large** | DistilBERT | NLP |
| **ResNet-152** | ResNet-18 | Vision |
| **GPT-4** | Smaller LLM | Generation |

---

⬅️ [Back: Model Compression](../)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=80&section=footer" width="100%"/>
</p>
