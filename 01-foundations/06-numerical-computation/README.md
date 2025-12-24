<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F38181&height=120&section=header&text=💻%20Numerical%20Computation&fontSize=36&fontColor=fff&animation=fadeIn&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-6_of_6-F38181?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Reading-15_min-00C853?style=for-the-badge&logo=clock&logoColor=white" alt="Reading Time"/>
  <img src="https://img.shields.io/badge/Level-Intermediate-FF9800?style=for-the-badge&logo=signal&logoColor=white" alt="Difficulty"/>
</p>

<p align="center">
  <i>How computers (and GPUs) actually do math</i>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

**✍️ Author:** [Gaurav Goswami](https://github.com/Gaurav14cs17)  
**📅 Published:** December 2024  
**🏷️ Tags:** `floating-point` `numerical-stability` `mixed-precision` `NaN` `overflow`

---

**🏠 [Home](../README.md)** · **📚 Series:** [Mathematical Thinking](../01-mathematical-thinking/README.md) → [Proof Techniques](../02-proof-techniques/README.md) → [Set Theory](../03-set-theory/README.md) → [Logic](../04-logic/README.md) → [Asymptotic Analysis](../05-asymptotic-analysis/README.md) → Numerical Computation

---

## 📌 TL;DR

Understanding floating-point math prevents mysterious training failures. This article covers:
- **IEEE 754 Format** — How FP32, FP16, BF16 represent numbers
- **Numerical Issues** — Overflow, underflow, catastrophic cancellation
- **Stable Algorithms** — Log-sum-exp trick, stable softmax
- **Mixed Precision** — Train faster with FP16/BF16

> [!CAUTION]
> **Common cause of `loss = NaN`:**
> ```python
> # ❌ BAD: Overflow in exp()
> softmax = np.exp(x) / np.sum(np.exp(x))
> 
> # ✅ GOOD: Subtract max first
> softmax = np.exp(x - x.max()) / np.sum(np.exp(x - x.max()))
> ```

---

## 📚 What You'll Learn

- [ ] Understand IEEE 754 floating-point representation
- [ ] Identify and fix numerical instability issues
- [ ] Implement stable softmax and log-sum-exp
- [ ] Use mixed precision training effectively
- [ ] Debug NaN/Inf errors in training

---

## 📑 Table of Contents

- [Visual Overview](#-visual-overview)
- [Why This Matters for ML](#-why-this-matters-for-ml)
- [Floating Point Formats](#-floating-point-formats)
- [Common Numerical Issues](#-common-numerical-issues-in-ml)
- [Mixed Precision Training](#-mixed-precision-training)
- [Key Numbers to Remember](#-key-numbers-to-remember)
- [Resources](#-references)
- [Navigation](#-navigation)

---

## 🎯 Visual Overview

<img src="./images/floating-point-representation.svg" width="100%">

*Caption: IEEE 754 floating-point representation showing sign bit, exponent, and mantissa. Understanding this is crucial for avoiding NaN/Inf issues in training and for mixed-precision optimization (FP16, BF16, FP8).*

<img src="./images/floating-point.svg" width="100%">

---

## 📂 Topics in This Folder

| File | Topic | Application |
|------|-------|-------------|
| [floating-point.md](./floating-point.md) | IEEE 754 representation | Understanding precision |

---

## 🎯 Why This Matters for ML

```
🏋️ Training Model
       │
       ▼
Epoch 1: loss = 2.543
       │
       ▼
Epoch 2: loss = 1.876
       │
       ▼
      ...
       │
       ▼
Epoch 50: loss = NaN 💀
       │
       ▼
🔍 What happened?
       │
   ┌───┼───┬───────────────┐
   │   │   │               │
   ▼   ▼   ▼               ▼
💥 Overflow  ➗ Division   🎯 Cancellation  📈 Accumulation
 (exp large)  by zero      (similar nums)   (rounding)
```

### 🚨 Common NaN Causes

| Issue | Example | Fix |
|:-----:|:--------|:---:|
| 💥 **Overflow** | e¹⁰⁰⁰ | Subtract max first |
| ➗ **Div by 0** | 1/10⁻⁴⁵ | Add epsilon |
| 🎯 **Cancellation** | a - b where a ≈ b | Reformulate |
| 📈 **Accumulation** | Sum of 10⁶ floats | Use Kahan sum |

---

## 📊 Floating Point Formats

```
FP32 (32 bits): [S:1] ──▶ [Exponent:8 bits] ──▶ [Mantissa:23 bits]

FP16 (16 bits): [S:1] ──▶ [Exp:5]           ──▶ [Mantissa:10]

BF16 (16 bits): [S:1] ──▶ [Exp:8]           ──▶ [Mant:7]
```

| Format | Bits | Range | Precision | Best For |
|:------:|:----:|:-----:|:---------:|:---------|
| **FP32** | 32 | ±3.4×10³⁸ | ~7 digits | 🎯 Training |
| **FP16** | 16 | ±65,504 | ~3 digits | ⚡ Mixed precision |
| **BF16** | 16 | ±3.4×10³⁸ | ~2 digits | 🚀 TPU/A100+ |
| **FP8** | 8 | ±240 | ~1 digit | 💨 H100 training |

### Memory and Speed Trade-offs

<!-- Memory Bar Chart -->
<p align="center">
  <img src="https://quickchart.io/chart?c={type:'horizontalBar',data:{labels:['FP32','FP16/BF16','FP8'],datasets:[{label:'Memory (GB) for 7B model',data:[28,14,7],backgroundColor:['%236C63FF','%234ECDC4','%2300C853']}]},options:{title:{display:true,text:'GPU Memory Requirements'}}}&width=500&height=200" alt="Memory Chart"/>
</p>

| Format | Memory | Fits On |
|:------:|:------:|:--------|
| FP32 | 28 GB | 🖥️ A100 80GB |
| FP16 | 14 GB | 🎮 RTX 4090 |
| FP8 | 7 GB | 🎮 RTX 3080 |

---

## 🔥 Common Numerical Issues in ML

### 1. Softmax Overflow

<table>
<tr>
<td width="50%">

**❌ BAD: Overflow!**

```python
def unstable_softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# x = [1000, 1001, 1002]
# Returns [nan, nan, nan] 💥
```

</td>
<td width="50%">

**✅ GOOD: Subtract max**

```python
def stable_softmax(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)
# Returns [0.09, 0.24, 0.67] ✅
```

</td>
</tr>
</table>

> [!TIP]
> **Key insight:** softmax(x - c) = softmax(x) for any constant c!

---

### 2. Log-Sum-Exp Underflow

<table>
<tr>
<td width="50%">

**❌ BAD: Underflow!**

```python
def unstable_logsumexp(x):
    return np.log(np.sum(np.exp(x)))

# x = [-1000, -1001, -1002]
# Returns -inf (wrong!) 💥
```

</td>
<td width="50%">

**✅ GOOD: Log-sum-exp trick**

```python
def stable_logsumexp(x):
    x_max = np.max(x)
    return x_max + np.log(
        np.sum(np.exp(x - x_max)))
# Returns -999.59 ✅
```

</td>
</tr>
</table>

```python
x = np.array([-1000, -1001, -1002])
# unstable_logsumexp(x)  # Returns -inf (wrong!)

# GOOD: Use log-sum-exp trick
def stable_logsumexp(x):
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max)))

stable_logsumexp(x)  # Returns -999.59 ✓
```

### 3. Variance Catastrophic Cancellation

```python
# BAD: Catastrophic cancellation
def unstable_variance(x):
    return np.mean(x**2) - np.mean(x)**2

# For large mean, small variance:
x = np.array([1e8, 1e8 + 1, 1e8 + 2])
unstable_variance(x)  # Returns 0.0 (wrong!) or negative!

# GOOD: Two-pass or Welford's algorithm
def stable_variance(x):
    mean = np.mean(x)
    return np.mean((x - mean)**2)

stable_variance(x)  # Returns 0.667 ✓
```

---

## 💻 Mixed Precision Training

```
⬇️ Forward Pass         📉 Loss              ⬆️ Backward Pass
┌─────────────────┐    ┌─────────────────┐   ┌─────────────────┐
│ FP16 Computation│───▶│FP32 Accumulation│──▶│ Scaled Gradients│
└─────────────────┘    └─────────────────┘   │ FP32 Update     │
                                             └─────────────────┘
```

<table>
<tr>
<td width="60%">

```python
import torch
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()  # Prevents FP16 underflow

for data, target in dataloader:
    optimizer.zero_grad()
    
    # Forward pass in FP16
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    # Backward with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

</td>
<td width="40%">

### ✅ Benefits

| Metric | Improvement |
|:------:|:-----------:|
| 🚀 **Speed** | 2× faster |
| 💾 **Memory** | 2× less |
| 🎯 **Accuracy** | Same |

### 🔧 Key Components

| Component | Purpose |
|:---------:|:--------|
| `autocast()` | FP16 forward |
| `GradScaler` | Prevent underflow |

</td>
</tr>
</table>

> [!TIP]
> Use `torch.cuda.amp` for automatic mixed precision — it handles the complexity for you!

---

## 📐 Key Numbers to Remember

```
📊 Floating Point Limits
┌──────────────────────────────────────────────────────────────┐
│  Machine ε (FP32): 1.19 × 10⁻⁷    Machine ε (FP16): 9.77×10⁻⁴│
│  Max FP32: 3.40 × 10³⁸            Max FP16: 65,504 ⚠️        │
└──────────────────────────────────────────────────────────────┘
```

| Constant | FP32 | FP16 | Meaning |
|:--------:|:----:|:----:|:--------|
| **Machine ε** | 1.19×10⁻⁷ | 9.77×10⁻⁴ | Smallest x where 1 + x ≠ 1 |
| **Max** | 3.40×10³⁸ | 65,504 | Before overflow 💥 |
| **Min positive** | 1.18×10⁻³⁸ | 6.10×10⁻⁵ | Before underflow |

> [!WARNING]
> FP16 max is only **65,504**! Logits > 11 will overflow in softmax!

---

## 🔗 Dependency Graph

| Topic | Leads To | Application |
|:------|:---------|:------------|
| 📊 **Floating Point** | Machine epsilon, Overflow/Underflow, Rounding | Foundation |
| 💥 **Overflow/Underflow** | Softmax stability | Training issues |
| 🔄 **Rounding Errors** | Numerical stability | Accuracy |
| ⚡ **Numerical Stability** | Training stability, Mixed precision | Core |
| 🚀 **Mixed Precision** | Efficient training | Speed + Memory |

```
📊 floating-point.md
        │
        ├──▶ 🔢 machine-epsilon.md
        │
        ├──▶ 💥 overflow-underflow.md ──▶ 🎯 softmax stability
        │
        └──▶ 🔄 rounding-errors.md
                      │
                      ▼
              ⚡ numerical-stability.md
                      │
                      ├──▶ 🏋️ training stability
                      │
                      └──▶ 🚀 mixed-precision.md ──▶ ⚡ efficient training
```

---

## 📐 Mathematical Formulas

### IEEE 754 Float Representation

```
Value = (-1)ˢ × (1 + m/2²³) × 2⁽ᵉ⁻¹²⁷⁾
```

| Component | Bits | Meaning |
|:---------:|:----:|:--------|
| s | 1 | Sign (0 = +, 1 = -) |
| e | 8 | Exponent |
| m | 23 | Mantissa |

**Special Values:**

| Value | Condition |
|:-----:|:----------|
| ±∞ | e = 255, m = 0 |
| NaN | e = 255, m ≠ 0 |
| 0 | e = 0, m = 0 |

---

### Condition Number

```
κ(A) = ‖A‖ · ‖A⁻¹‖
```

| Value | Interpretation |
|:-----:|:---------------|
| κ ≈ 1 | ✅ Well-conditioned |
| κ ≫ 1 | ⚠️ Ill-conditioned |
| κ = ∞ | ❌ Singular |

**Relative error bound:**

```
‖δx‖/‖x‖ ≤ κ(A) · ‖δb‖/‖b‖
```

---

### Numerical Gradient Check

```
∂f/∂x ≈ (f(x + ε) - f(x - ε)) / 2ε
```

| Parameter | Recommended | Note |
|:---------:|:-----------:|:-----|
| ε | √machine_epsilon ≈ 10⁻⁴ | For FP32 |
| Error | O(ε²) | Centered difference |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Goldberg: Floating-Point | [ACM](https://dl.acm.org/doi/10.1145/103162.103163) |
| 📄 | Mixed Precision Training | [arXiv](https://arxiv.org/abs/1710.03740) |
| 📖 | Trefethen & Bau | [Book](https://people.maths.ox.ac.uk/trefethen/text.html) |
| 🎥 | Computerphile: Floating Point | [YouTube](https://www.youtube.com/watch?v=PZRI1IfStY0) |
| 🇨🇳 | 浮点数与数值稳定性 | [知乎](https://zhuanlan.zhihu.com/p/103482462) |
| 🇨🇳 | IEEE 754详解 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88776412) |
| 🇨🇳 | 混合精度训练教程 | [B站](https://www.bilibili.com/video/BV1Y64y1Q7hi)

---

## 🔗 Where This Topic Is Used

| Topic | How Numerical Computation Is Used |
|-------|-----------------------------------|
| **Mixed Precision Training** | FP16/BF16 for speed |
| **Gradient Clipping** | Prevent exploding gradients |
| **Loss Scaling** | Prevent FP16 underflow |
| **Softmax Stability** | log-sum-exp trick |
| **BatchNorm** | Running statistics accumulation |
| **Quantization** | INT8/INT4 inference |

---

## 🧭 Navigation

<table width="100%">
<tr>
<td align="left" width="33%">

⬅️ **Previous**<br>
[⏱️ Asymptotic Analysis](../05-asymptotic-analysis/README.md)

</td>
<td align="center" width="34%">

📍 **Current: 6 of 6**<br>
**Numerical Computation**

</td>
<td align="right" width="33%">

🏁 **Complete!**<br>
[🏠 Back to Home](../README.md)

</td>
</tr>
</table>

---

<!-- Completion Banner -->
<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=24&pause=1000&color=F38181&center=true&vCenter=true&width=600&lines=🎉+Congratulations!;You've+completed+the+Foundations+series!" alt="Completion" />
</p>

<!-- Animated Footer -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F38181&height=80&section=footer&animation=fadeIn" width="100%"/>
</p>

<p align="center">
  <a href="../README.md"><img src="https://img.shields.io/badge/📚_Part_of-ML_Researcher_Foundations-F38181?style=for-the-badge" alt="Series"/></a>
</p>

<p align="center">
  <sub>Made with ❤️ by <a href="https://github.com/Gaurav14cs17">Gaurav Goswami</a></sub>
</p>


