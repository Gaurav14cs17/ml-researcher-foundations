<!-- Navigation -->
<p align="center">
  <a href="../">⬅️ Back: ML Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../01_learning_theory/">Next: Learning Theory ➡️</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Learning%20Frameworks&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/learning-paradigms.svg" width="100%">

*Caption: The four main learning paradigms in ML. The modern training pipeline (GPT, Claude, LLaMA) combines self-supervised pretraining, supervised fine-tuning, and RLHF for alignment.*

---

## 📐 Mathematical Foundations

### Supervised Learning

**Problem Setting:**
Given a training set \(\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}\) where \(x_i \in \mathcal{X}\) and \(y_i \in \mathcal{Y}\), we aim to learn a function \(f: \mathcal{X} \rightarrow \mathcal{Y}\).

**Objective Function:**

```math
\min_{f \in \mathcal{F}} \mathcal{L}(f) = \min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^{N} \ell(f(x_i), y_i)
```

**Common Loss Functions:**

| Task | Loss Function | Formula |
|------|---------------|---------|
| **Classification** | Cross-Entropy | \(\ell(f(x), y) = -\sum_{c=1}^{C} y_c \log(f(x)_c)\) |
| **Binary Classification** | Binary Cross-Entropy | \(\ell(f(x), y) = -[y\log(f(x)) + (1-y)\log(1-f(x))]\) |
| **Regression** | Mean Squared Error | \(\ell(f(x), y) = (f(x) - y)^2\) |
| **Regression** | Mean Absolute Error | \(\ell(f(x), y) = |f(x) - y|\) |

---

### Self-Supervised Learning

**Key Insight:** Create supervision signal from the data itself.

**1. Masked Language Modeling (BERT):**

```math
\mathcal{L}_{\text{MLM}} = -\mathbb{E}_{x \sim \mathcal{D}} \left[ \sum_{i \in \mathcal{M}} \log P(x_i | x_{\backslash \mathcal{M}}; \theta) \right]
```

where \(\mathcal{M}\) is the set of masked token positions.

**2. Autoregressive (GPT):**

```math
\mathcal{L}_{\text{AR}} = -\sum_{t=1}^{T} \log P(x_t | x_1, x_2, \ldots, x_{t-1}; \theta)
```

**3. Contrastive Learning (SimCLR):**

For positive pair \((z_i, z_j)\) from the same image:

```math
\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)}
```

where \(\text{sim}(u, v) = \frac{u^\top v}{\|u\| \|v\|}\) is cosine similarity and \(\tau\) is temperature.

**Proof: Why Contrastive Loss Works**

The InfoNCE loss is a lower bound on mutual information:

```math
I(X; Y) \geq \log(N) - \mathcal{L}_{\text{NCE}}
```

Maximizing \(-\mathcal{L}_{\text{NCE}}\) maximizes a lower bound on \(I(X; Y)\), learning representations that capture shared information between views.

---

### Unsupervised Learning

**Problem Setting:** Given \(\mathcal{D} = \{x_i\}_{i=1}^{N}\), discover structure without labels.

**1. Clustering (k-Means):**

```math
\min_{\mu_1, \ldots, \mu_K} \sum_{i=1}^{N} \min_{k} \|x_i - \mu_k\|^2
```

**2. Dimensionality Reduction (PCA):**

```math
\max_{W \in \mathbb{R}^{d \times k}} \text{Var}(Wx) \quad \text{s.t.} \quad W^\top W = I
```

**Solution:** \(W\) = top-k eigenvectors of covariance matrix \(\Sigma = \frac{1}{N}\sum_i (x_i - \bar{x})(x_i - \bar{x})^\top\)

**3. Variational Autoencoder (VAE):**

```math
\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{\text{KL}}(q(z|x) \| p(z))
```

---

## 📊 Empirical Risk Minimization (ERM)

### Definition

**True Risk (Population Risk):**

```math
R(h) = \mathbb{E}_{(x,y) \sim P}[\ell(h(x), y)] = \int \ell(h(x), y) \, dP(x, y)
```

**Empirical Risk:**

```math
\hat{R}(h) = \frac{1}{n} \sum_{i=1}^{n} \ell(h(x_i), y_i)
```

**ERM Principle:**

```math
\hat{h}_{\text{ERM}} = \arg\min_{h \in \mathcal{H}} \hat{R}(h)
```

### Theoretical Justification

**Theorem (Law of Large Numbers):**
For fixed hypothesis \(h\):

```math
\hat{R}(h) \xrightarrow{p} R(h) \quad \text{as } n \to \infty
```

**Theorem (Uniform Convergence):**
For hypothesis class \(\mathcal{H}\) with finite VC dimension \(d\):

```math
\Pr\left[\sup_{h \in \mathcal{H}} |R(h) - \hat{R}(h)| > \epsilon\right] \leq 4 \cdot m_{\mathcal{H}}(2n) \cdot e^{-n\epsilon^2/8}
```

where \(m_{\mathcal{H}}(n)\) is the growth function.

### Generalization Bound

**Theorem:** With probability \(\geq 1 - \delta\):

```math
R(\hat{h}) \leq \hat{R}(\hat{h}) + \sqrt{\frac{2d \log(en/d) + 2\log(2/\delta)}{n}}
```

**Proof Sketch:**
1. Apply Hoeffding's inequality to bound deviation for single \(h\)
2. Use union bound over effective hypotheses (bounded by growth function)
3. Apply Sauer's lemma: \(m_{\mathcal{H}}(n) \leq \left(\frac{en}{d}\right)^d\)

### The Overfitting Problem

```math
R(\hat{h}) = \underbrace{\hat{R}(\hat{h})}_{\text{training error}} + \underbrace{(R(\hat{h}) - \hat{R}(\hat{h}))}_{\text{generalization gap}}
```

**Solutions:**

| Method | Modification | Effect |
|--------|-------------|--------|
| **Regularized ERM** | \(\min \hat{R}(h) + \lambda\Omega(h)\) | Constrains hypothesis complexity |
| **Early Stopping** | Stop before convergence | Implicit regularization |
| **Cross-Validation** | Holdout set | Estimate true risk |
| **More Data** | Increase \(n\) | Tighter bound |

---

## 💻 Code Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LearningFrameworks:
    """Implementations of different learning paradigms."""
    
    @staticmethod
    def empirical_risk(model, dataloader, loss_fn):
        """Compute empirical risk (average loss).
        
        Args:
            model: Neural network model
            dataloader: PyTorch DataLoader
            loss_fn: Loss function (e.g., nn.CrossEntropyLoss())
            
        Returns:
            float: Average loss over dataset
        """
        model.eval()
        total_loss = 0.0
        n_samples = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                pred = model(x)
                loss = loss_fn(pred, y)
                total_loss += loss.item() * x.size(0)
                n_samples += x.size(0)
        
        return total_loss / n_samples
    
    @staticmethod
    def erm_training_loop(model, train_loader, optimizer, loss_fn, epochs):
        """Standard ERM training loop.
        
        The optimization:
            θ* = argmin_θ (1/n) Σ_i L(f_θ(x_i), y_i)
        """
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in train_loader:
                optimizer.zero_grad()
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(train_loader):.4f}")

class ContrastiveLoss(nn.Module):
    """SimCLR-style contrastive loss (InfoNCE)."""
    
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_i, z_j):
        """
        Compute NT-Xent loss.
        
        L = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
        
        Args:
            z_i, z_j: Representations of augmented views [batch_size, dim]
        """
        batch_size = z_i.size(0)
        
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate
        z = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, dim]
        
        # Similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # [2N, 2N]
        
        # Create labels for positive pairs
        labels = torch.cat([
            torch.arange(batch_size, 2*batch_size),
            torch.arange(batch_size)
        ]).to(z.device)
        
        # Mask out self-similarity
        mask = torch.eye(2*batch_size, dtype=torch.bool).to(z.device)
        sim_matrix.masked_fill_(mask, float('-inf'))
        
        # Cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

class MaskedLanguageModel(nn.Module):
    """BERT-style masked language modeling."""
    
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, input_ids, mask_positions):
        """
        L_MLM = -E[Σ_{i ∈ masked} log P(x_i | x_\masked)]
        """
        x = self.embedding(input_ids)
        x = self.transformer(x)
        
        # Only compute loss for masked positions
        masked_representations = x[mask_positions]
        logits = self.output(masked_representations)
        return logits

class AutoregressiveLM(nn.Module):
    """GPT-style autoregressive language model."""
    
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, input_ids):
        """
        L_AR = -Σ_t log P(x_t | x_1, ..., x_{t-1})
        """
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len).unsqueeze(0).to(input_ids.device)
        
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        
        # Causal mask: can only attend to previous tokens
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len), diagonal=1
        ).bool().to(input_ids.device)
        
        # Memory is the input itself for decoder-only
        x = self.transformer(x, x, tgt_mask=causal_mask)
        logits = self.output(x)
        return logits
```

---

## 📂 Topics

| Folder | Topic | Key Idea |
|--------|-------|----------|
| [01_self_supervised/](./01_self_supervised/) | Self-Supervised Learning | Create labels from data |
| [02_supervised/](./02_supervised/) | Supervised Learning | (x, y) pairs |
| [03_unsupervised/](./03_unsupervised/) | Unsupervised Learning | Find structure |

---

## 📊 Comparison

| Framework | Input | Output | Loss Function | Examples |
|-----------|-------|--------|---------------|----------|
| **Supervised** | (x, y) | Predictor | Cross-entropy, MSE | Classification, Regression |
| **Unsupervised** | x | Structure | Reconstruction, KL | Clustering, PCA, VAE |
| **Self-supervised** | x | Features | Contrastive, Reconstruction | BERT, GPT, SimCLR |
| **Reinforcement** | (s, a, r) | Policy | Expected return | Games, Robotics |

---

## 🔥 Modern Paradigm

```
1. Self-supervised pre-training (massive unlabeled data)
   +-- Language: GPT (autoregressive), BERT (masked LM)
   +-- Vision: MAE (masked autoencoding), SimCLR (contrastive)
   +-- Multi-modal: CLIP (image-text contrastive)
   
2. Supervised fine-tuning (small labeled data)
   +-- Task-specific adaptation (classification, QA, etc.)

3. Alignment (RLHF for safety and helpfulness)
   +-- ChatGPT, Claude, Gemini
```

---

## 🔗 Where This Topic Is Used

| Framework | Applications |
|-----------|-------------|
| **Supervised** | Classification, Regression, Object Detection, Segmentation |
| **Self-Supervised** | BERT, GPT, SimCLR, MAE, CLIP, Stable Diffusion |
| **Unsupervised** | Clustering, PCA, VAE, Anomaly Detection |
| **Semi-Supervised** | Pseudo-labeling, MixMatch, FixMatch |
| **Reinforcement** | Q-learning, PPO, RLHF |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | BERT: Pre-training of Deep Bidirectional Transformers | [arXiv](https://arxiv.org/abs/1810.04805) |
| 📄 | SimCLR: A Simple Framework for Contrastive Learning | [arXiv](https://arxiv.org/abs/2002.05709) |
| 📄 | GPT-3: Language Models are Few-Shot Learners | [arXiv](https://arxiv.org/abs/2005.14165) |
| 📖 | Deep Learning (Goodfellow et al.) | [Book](https://www.deeplearningbook.org/) |
| 📖 | Understanding Machine Learning (Shalev-Shwartz) | [Book](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../">⬅️ Back: ML Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../01_learning_theory/">Next: Learning Theory ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
