<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Adversarial%20Robustness&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/adversarial-robustness-complete.svg" width="100%">

*Caption: Adversarial examples are inputs designed to fool models. Small, imperceptible perturbations can cause misclassification. Adversarial training is the main defense.*

---

## 📐 Mathematical Foundations

### Adversarial Examples

```
Find perturbation δ such that:
• f(x + δ) ≠ f(x)  (causes misclassification)
• ||δ||_p ≤ ε      (perturbation is small)

Common norms:
• L∞: max pixel change ≤ ε
• L2: Euclidean distance ≤ ε
• L0: Number of changed pixels ≤ ε
```

### Attack Methods

```
FGSM (Fast Gradient Sign Method):
δ = ε · sign(∇ₓL(f(x), y))

Single step, fast but weak

PGD (Projected Gradient Descent):
x^(t+1) = Π_{||δ||≤ε}(x^(t) + α · sign(∇ₓL(f(x^(t)), y)))

Multi-step, stronger attack

C&W Attack:
min_δ ||δ||_p + c · max(Z(x+δ)_y - max_{i≠y}Z(x+δ)_i, -κ)

Optimization-based, very strong
```

### Adversarial Training

```
Standard training:
min_θ E_{(x,y)}[L(f_θ(x), y)]

Adversarial training:
min_θ E_{(x,y)}[max_{||δ||≤ε} L(f_θ(x+δ), y)]

Min-max optimization: Train on worst-case perturbations
```

---

## 🎯 Attack Types

| Attack | Method | Strength | Speed |
|--------|--------|----------|-------|
| **FGSM** | Single gradient step | Weak | Fast |
| **PGD** | Multi-step gradient | Strong | Slow |
| **C&W** | Optimization | Very strong | Very slow |
| **AutoAttack** | Ensemble | State-of-art | Slow |

---

## 💻 Code Examples

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# FGSM Attack
def fgsm_attack(model, x, y, epsilon):
    """Fast Gradient Sign Method"""
    x.requires_grad = True
    output = model(x)
    loss = F.cross_entropy(output, y)
    loss.backward()
    
    # Create perturbation
    perturbation = epsilon * x.grad.sign()
    x_adv = x + perturbation
    x_adv = torch.clamp(x_adv, 0, 1)  # Keep in valid range
    
    return x_adv

# PGD Attack
def pgd_attack(model, x, y, epsilon, alpha, num_steps):
    """Projected Gradient Descent"""
    x_adv = x.clone().detach()
    
    for _ in range(num_steps):
        x_adv.requires_grad = True
        output = model(x_adv)
        loss = F.cross_entropy(output, y)
        loss.backward()
        
        # Update with gradient
        x_adv = x_adv + alpha * x_adv.grad.sign()
        
        # Project back to epsilon ball
        perturbation = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = torch.clamp(x + perturbation, 0, 1).detach()
    
    return x_adv

# Adversarial Training
def adversarial_training_step(model, x, y, optimizer, epsilon):
    """Single adversarial training step"""
    # Generate adversarial examples
    model.eval()
    x_adv = pgd_attack(model, x, y, epsilon, epsilon/4, 10)
    
    # Train on adversarial examples
    model.train()
    optimizer.zero_grad()
    output = model(x_adv)
    loss = F.cross_entropy(output, y)
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Evaluate robustness
def evaluate_robustness(model, test_loader, epsilon):
    """Evaluate model under attack"""
    model.eval()
    correct_clean = 0
    correct_adv = 0
    total = 0
    
    for x, y in test_loader:
        # Clean accuracy
        output_clean = model(x)
        pred_clean = output_clean.argmax(dim=1)
        correct_clean += (pred_clean == y).sum().item()
        
        # Adversarial accuracy
        x_adv = pgd_attack(model, x, y, epsilon, epsilon/4, 20)
        output_adv = model(x_adv)
        pred_adv = output_adv.argmax(dim=1)
        correct_adv += (pred_adv == y).sum().item()
        
        total += y.size(0)
    
    print(f"Clean accuracy: {100*correct_clean/total:.2f}%")
    print(f"Adversarial accuracy: {100*correct_adv/total:.2f}%")

# Using torchattacks library
try:
    import torchattacks
    
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    
    # Various attacks
    fgsm = torchattacks.FGSM(model, eps=0.3)
    pgd = torchattacks.PGD(model, eps=0.3, alpha=0.01, steps=40)
    autoattack = torchattacks.AutoAttack(model, eps=0.3)
except ImportError:
    print("torchattacks not installed")
```

---

## 🌍 ML Applications

| Application | Concern | Defense |
|-------------|---------|---------|
| **Autonomous Vehicles** | Fooling perception | Adversarial training |
| **Malware Detection** | Evasion attacks | Robust features |
| **Face Recognition** | Adversarial glasses | Certified defenses |
| **Content Moderation** | Bypass filters | Ensemble methods |
| **Medical Imaging** | Reliability | Input preprocessing |

---

## 📊 Robustness vs Accuracy Trade-off

```
Standard Model:
• High clean accuracy
• Low adversarial accuracy

Robust Model:
• Slightly lower clean accuracy
• Much higher adversarial accuracy

Trade-off is fundamental (theoretically proven for some settings)
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Explaining Adversarial | [Paper](https://arxiv.org/abs/1412.6572) |
| 📄 | PGD Training | [Paper](https://arxiv.org/abs/1706.06083) |
| 📄 | AutoAttack | [Paper](https://arxiv.org/abs/2003.01690) |
| 📖 | RobustBench | [Website](https://robustbench.github.io/) |
| 🇨🇳 | 对抗样本详解 | [知乎](https://zhuanlan.zhihu.com/p/32784766) |

---

## 🔗 Where This Topic Is Used

| Application | How Robustness Is Used |
|-------------|------------------------|
| **Safety-Critical ML** | Ensure reliable predictions |
| **Security** | Defend against attacks |
| **Model Evaluation** | Stress testing |
| **Regularization** | Improve generalization |
| **Understanding Models** | Probe decision boundaries |

---

⬅️ [Back: 10-Interpretability](../10-interpretability/) | ➡️ [Next: 12-Model Calibration](../12-model-calibration/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
