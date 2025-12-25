<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=06 RLHF&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 🤖 Reinforcement Learning from Human Feedback

> **Aligning LLMs with human preferences**

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

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

