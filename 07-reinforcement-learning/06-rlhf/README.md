<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=Reinforcement%20Learning%20from%20Hu&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 RLHF Pipeline

```
1. Supervised Fine-tuning (SFT)
   Train on human demonstrations

2. Reward Model Training
   Collect preference data: (x, y_w, y_l)
   Train reward model: r(x, y)
   Loss: -log σ(r(x, y_w) - r(x, y_l))

3. RL Optimization (PPO)
   Maximize: E[r(x, y) - β KL(π || π_ref)]
```

---

## 💻 Key Methods

| Method | Description |
|--------|-------------|
| **RLHF** | PPO with reward model |
| **DPO** | Direct preference optimization (no RL) |
| **RLAIF** | AI feedback instead of human |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | InstructGPT | [arXiv](https://arxiv.org/abs/2203.02155) |
| 📄 | DPO | [arXiv](https://arxiv.org/abs/2305.18290) |
| 🇨🇳 | RLHF详解 | [知乎](https://zhuanlan.zhihu.com/p/595579042) |

---

⬅️ [Back: 05-Model-Based](../05-model-based/) | ➡️ [Back: RL](../)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
