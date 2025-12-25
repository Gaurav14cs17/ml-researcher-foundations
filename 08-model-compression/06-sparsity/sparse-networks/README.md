<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=120&section=header&text=Sparse%20Neural%20Networks&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-08-E74C3C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Mathematical Foundations

### Sparsity Ratio
```
Sparsity = (# zeros) / (total params) × 100%

90% sparse: Only 10% of weights are non-zero
```

### Sparse Matrix Representation
```
CSR (Compressed Sparse Row):
• values: [non-zero values]
• col_indices: [column of each value]
• row_ptr: [start of each row]

Memory: O(nnz) instead of O(m×n)
```

### 2:4 Structured Sparsity
```
Every 4 consecutive elements: 2 are zero

[a, 0, b, 0] or [0, a, 0, b] or [a, b, 0, 0] etc.

50% sparsity with 2x speedup on Ampere GPUs
```

---

## 📂 Topics

| File | Topic | Key Concepts |
|------|-------|--------------|

---

## 🎯 Types of Sparsity

```
Unstructured Sparsity:
+-------------------------------------+
| [1.2, 0, 0.8, 0, 0, 0.3, 0, 0.9]   |
| Zeros anywhere                      |
| Flexible but hardware-unfriendly    |
+-------------------------------------+

Structured Sparsity (2:4):
+-------------------------------------+
| [1.2, 0, | 0.8, 0, | 0.3, 0, | 0.9, 0] |
| 2 zeros per 4 elements              |
| NVIDIA Ampere supports natively!    |
+-------------------------------------+

Block Sparsity:
+-------------------------------------+
| [Block1, 0, 0, Block4]              |
| Entire blocks are zero              |
| Very hardware-friendly              |
+-------------------------------------+
```

---

## 📊 Sparse Attention

```
Full Attention O(n²):
+-------------------------------------+
| Every token attends to every token  |
| n=8192 → 67M attention scores!      |
+-------------------------------------+

Sparse Attention:
+-------------------------------------+
| Local: Attend to nearby tokens      |
| Global: Some tokens attend to all   |
| Random: Random attention patterns   |
| → O(n√n) or O(n)                   |
+-------------------------------------+
```

---

## 🎰 Lottery Ticket Hypothesis

<img src="./images/lottery-ticket.svg" width="100%">

---

## 🔗 Where This Topic Is Used

| Topic | How Sparse Networks Are Used |
|-------|----------------------------|
| **Sparse Transformers** | Sparse attention for long docs |
| **BigBird** | Sparse attention patterns |
| **Longformer** | Local + global attention |
| **NVIDIA Ampere** | 2:4 structured sparsity |
| **Lottery Ticket** | Find sparse subnetworks |

---

## 📚 References & Resources

### 📄 Key Papers

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|------------------|
| [Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635) | Frankle & Carlin | 2018 | Original paper |
| [Stabilizing LTH](https://arxiv.org/abs/1903.01611) | Frankle et al. | 2019 | Late rewinding |
| [Sparse Transformers](https://arxiv.org/abs/1904.10509) | Child et al. | 2019 | Sparse attention |
| [2:4 Sparsity](https://arxiv.org/abs/2104.08378) | Mishra et al. | 2021 | NVIDIA hardware |
| 🇨🇳 彩票假设详解 | [知乎](https://zhuanlan.zhihu.com/p/84176060) | - | Lottery Ticket原理 |
| 🇨🇳 稀疏网络综述 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/125689573) | - | 稀疏训练技术总结 |

---

⬅️ [Back: MOE](../moe/)


---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=80&section=footer" width="100%"/>
</p>
