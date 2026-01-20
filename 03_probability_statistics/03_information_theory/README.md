<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Information%20Theory&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📂 Topics in This Folder

| Folder | Topics | ML Application |
|--------|--------|----------------|
| [01_cross_entropy/](./01_cross_entropy/) | Cross-entropy loss | 🔥 Classification loss |
| [02_entropy/](./02_entropy/) | Shannon entropy, max-entropy | Uncertainty measurement |
| [03_kl_divergence/](./03_kl_divergence/) | KL divergence, properties | 🔥 VAE, distillation |
| [04_mutual_information/](./04_mutual_information/) | Mutual information, InfoNCE | Feature selection, contrastive learning |

---

## 📐 Shannon Entropy

### Definition

**Discrete:**

```math
H(X) = -\sum_{x} p(x) \log_2 p(x) \quad \text{(bits)}
```

**Continuous (Differential Entropy):**

```math
h(X) = -\int f(x) \log f(x) \, dx
```

### Properties

| Property | Formula | Interpretation |
|----------|---------|----------------|
| Non-negative | $H(X) \geq 0$ | Always |
| Maximum | $H(X) \leq \log\|X\|$ | Uniform maximizes |
| Zero entropy | $H(X) = 0 \iff$ deterministic | No uncertainty |
| Chain rule | $H(X,Y) = H(X) + H(Y\|X)$ | Joint = marginal + conditional |

### Key Examples

| Distribution | Entropy |
|--------------|---------|
| Fair coin | $H = 1$ bit |
| Biased coin ($p$) | $H = -p\log\_2 p - (1-p)\log\_2(1-p)$ |
| Uniform on $n$ | $H = \log\_2 n$ |
| Gaussian $\mathcal{N}(\mu, \sigma^2)$ | $h = \frac{1}{2}\log(2\pi e \sigma^2)$ |

---

## 📐 Cross-Entropy

### Definition

```math
H(p, q) = -\sum_{x} p(x) \log q(x) = H(p) + D_{KL}(p \| q)
```

### Key Relationship

```math
H(p, q) = \underbrace{H(p)}_{\text{min possible}} + \underbrace{D_{KL}(p \| q)}_{\geq 0 \text{ (extra bits)}}
```

### Classification Loss

For true label $y$ (one-hot) and predicted distribution $\hat{p}$:

```math
\mathcal{L}_{CE} = -\sum_{k} y_k \log \hat{p}_k = -\log \hat{p}_{y_{true}}
```

**With softmax:**

```math
\hat{p}_k = \frac{e^{z_k}}{\sum_j e^{z_j}}
```

**Gradient (remarkably simple):**

```math
\frac{\partial \mathcal{L}}{\partial z_k} = \hat{p}_k - y_k
```

---

## 📐 KL Divergence

### Definition

```math
D_{KL}(P \| Q) = \sum_x p(x) \log\frac{p(x)}{q(x)} = E_p\left[\log\frac{p(x)}{q(x)}\right]
```

### Properties

| Property | Statement |
|----------|-----------|
| Non-negative | $D\_{KL}(P \| Q) \geq 0$ (Gibbs' inequality) |
| Zero | $D\_{KL} = 0 \iff P = Q$ |
| Asymmetric | $D\_{KL}(P \| Q) \neq D\_{KL}(Q \| P)$ |
| Additive | $D\_{KL}(P\_1 P\_2 \| Q\_1 Q\_2) = D\_{KL}(P\_1 \| Q\_1) + D\_{KL}(P\_2 \| Q\_2)$ |

### Gibbs' Inequality Proof

```math
D_{KL}(P \| Q) = E_p\left[\log\frac{p}{q}\right] = -E_p\left[\log\frac{q}{p}\right]
```

Using Jensen's inequality ($\log$ is concave):

```math
-E_p\left[\log\frac{q}{p}\right] \geq -\log E_p\left[\frac{q}{p}\right] = -\log\sum_x p(x)\frac{q(x)}{p(x)} = -\log 1 = 0 \quad \blacksquare
```

### Forward vs Reverse KL

| Direction | Formula | Behavior | Use Case |
|-----------|---------|----------|----------|
| Forward | $D\_{KL}(P \| Q)$ | Zero-avoiding (q covers p) | Variational Inference |
| Reverse | $D\_{KL}(Q \| P)$ | Zero-forcing (q focuses) | Mode-seeking |

---

## 📐 KL Divergence for Gaussians

```math
D_{KL}(\mathcal{N}_1 \| \mathcal{N}_2) = \frac{1}{2}\left[\log\frac{\sigma_2^2}{\sigma_1^2} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{\sigma_2^2} - 1\right]
```

**VAE Loss (KL to standard normal):**

```math
D_{KL}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1)) = \frac{1}{2}\left(\mu^2 + \sigma^2 - 1 - \log\sigma^2\right)
```

---

## 📐 Mutual Information

### Definition

```math
I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
```

**As KL Divergence:**

```math
I(X; Y) = D_{KL}(p(x, y) \| p(x)p(y))
```

### Properties

| Property | Formula |
|----------|---------|
| Symmetric | $I(X;Y) = I(Y;X)$ |
| Non-negative | $I(X;Y) \geq 0$ |
| Zero | $I(X;Y) = 0 \iff X \perp Y$ |
| Data Processing | $X \to Y \to Z \implies I(X;Z) \leq I(X;Y)$ |

### InfoNCE Loss

```math
\mathcal{L}_{InfoNCE} = -E\left[\log\frac{e^{f(x_i, y_i)/\tau}}{\sum_{j=1}^{N} e^{f(x_i, y_j)/\tau}}\right]
```

**Lower bounds mutual information:** $I(X; Y) \geq \log N - \mathcal{L}\_{InfoNCE}$

---

## 💻 Code Examples

```python
import numpy as np
import torch
import torch.nn.functional as F

# Shannon Entropy
def entropy(p, eps=1e-10):
    """H(X) = -Σ p(x) log p(x)"""
    p = np.clip(p, eps, 1)
    return -np.sum(p * np.log2(p))

# Fair vs biased coin
print(f"Fair coin: {entropy([0.5, 0.5]):.3f} bits")  # 1.0 bits
print(f"Biased (0.9, 0.1): {entropy([0.9, 0.1]):.3f} bits")  # 0.47 bits

# Cross-Entropy Loss
logits = torch.randn(32, 10)  # 32 samples, 10 classes
labels = torch.randint(0, 10, (32,))
ce_loss = F.cross_entropy(logits, labels)

# KL Divergence
def kl_divergence(p, q, eps=1e-10):
    """D_KL(P || Q)"""
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * np.log(p / q))

p = np.array([0.4, 0.3, 0.3])
q = np.array([0.33, 0.33, 0.34])
print(f"D_KL(P||Q) = {kl_divergence(p, q):.4f}")
print(f"D_KL(Q||P) = {kl_divergence(q, p):.4f}")  # Different!

# VAE KL Loss
def vae_kl_loss(mu, log_var):
    """D_KL(N(mu, sigma^2) || N(0, I))"""
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

mu = torch.randn(32, 64)
log_var = torch.randn(32, 64)
kl_loss = vae_kl_loss(mu, log_var)

# InfoNCE Loss
def info_nce_loss(z1, z2, temperature=0.5):
    """Contrastive loss for mutual information estimation"""
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    
    sim = F.cosine_similarity(z.unsqueeze(0), z.unsqueeze(1), dim=2)
    sim = sim / temperature
    
    labels = torch.cat([
        torch.arange(batch_size) + batch_size,
        torch.arange(batch_size)
    ])
    
    mask = torch.eye(2 * batch_size).bool()
    sim.masked_fill_(mask, float('-inf'))
    
    return F.cross_entropy(sim, labels)
```

---

## 🌍 ML Applications

| Concept | Application | Example |
|---------|-------------|---------|
| Cross-entropy | Classification loss | `nn.CrossEntropyLoss` |
| KL divergence | VAE regularization | $D\_{KL}(q(z\|x) \| p(z))$ |
| KL divergence | Knowledge distillation | Soft target matching |
| Mutual information | Contrastive learning | InfoNCE (SimCLR, CLIP) |
| Entropy | Exploration in RL | Max-entropy RL (SAC) |
| Entropy | Decision trees | Information gain (ID3) |

---

## 🔥 Why Cross-Entropy is THE Loss Function

```
Classification: Given true label y, predict P(y|x)

Cross-entropy loss:
L = -log P(y=true|x)

Why this works:
1. Minimizing L = maximizing log P(y|x) = MLE!
2. L = H(p_true, p_model) where p_true is one-hot
3. Gradients are simple: ∂L/∂logit = p_model - p_true
4. Well-calibrated probabilities (not just rankings)

Compare to MSE for classification:
• MSE: (1 - p)² when correct → saturates
• CE:  -log(p) when correct → never saturates
• CE penalizes confident wrong predictions more!
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Elements of Information Theory | Cover & Thomas |
| 📖 | Information Theory, Inference | [MacKay](http://www.inference.org.uk/itprnn/book.pdf) |
| 🎥 | Visual Information Theory | [3Blue1Brown](https://www.youtube.com/watch?v=v68zYyaEmEA) |
| 📄 | Variational Inference | [arXiv](https://arxiv.org/abs/1601.00670) |
| 🇨🇳 | 信息论基础 | [知乎](https://zhuanlan.zhihu.com/p/35379531) |
| 🇨🇳 | 交叉熵与KL散度 | [CSDN](https://blog.csdn.net/tsyccnh/article/details/79163834) |

---

## 🔗 Where Information Theory Is Used

| Topic | How It's Used |
|-------|---------------|
| **Cross-Entropy Loss** | Classification (nn.CrossEntropyLoss) |
| **VAE** | KL divergence regularization |
| **Knowledge Distillation** | KL between teacher-student |
| **RLHF** | KL penalty in PPO objective |
| **DPO** | Implicit KL in preference loss |
| **Contrastive Learning** | InfoNCE loss (mutual information) |
| **Max-Entropy RL** | Entropy bonus for exploration |
| **Language Models** | Perplexity = exp(cross-entropy) |
| **Compression** | Entropy = minimum bits |
| **Variational Inference** | ELBO = reconstruction - KL |

---

⬅️ [Back: Multivariate](../02_multivariate/) | ➡️ [Next: Estimation](../04_estimation/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
