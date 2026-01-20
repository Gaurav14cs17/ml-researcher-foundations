<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Sequence-to-Sequence&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Mathematical Foundations

### Problem Definition

Map input sequence to output sequence:

```math
(x_1, x_2, ..., x_n) \rightarrow (y_1, y_2, ..., y_m)
```

Where input length $n$ and output length $m$ may differ.

### Conditional Probability

```math
P(y_1, ..., y_m | x_1, ..., x_n) = \prod_{t=1}^{m} P(y_t | y_1, ..., y_{t-1}, x_1, ..., x_n)
```

---

## 📐 Encoder-Decoder Architecture

### Encoder

Process input sequence to fixed representation:

```math
h_t = \text{RNN}(x_t, h_{t-1})
```

Final hidden state: $c = h\_n$ (context vector)

For bidirectional:

```math
h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]
```

### Decoder

Generate output sequence:

```math
s_t = \text{RNN}(y_{t-1}, s_{t-1}, c)
P(y_t | y_{1:t-1}, x) = \text{softmax}(W_o s_t)
```

### Information Bottleneck

**Problem:** Entire input compressed into single vector $c$.

Long sequences → information loss.

**Solution:** Attention mechanism.

---

## 📐 Attention Mechanism

### Bahdanau Attention (Additive)

**Alignment score:**

```math
e_{t,s} = v^T \tanh(W_a s_{t-1} + U_a h_s)
```

**Attention weights:**

```math
\alpha_{t,s} = \frac{\exp(e_{t,s})}{\sum_{k=1}^{n} \exp(e_{t,k})}
```

**Context vector:**

```math
c_t = \sum_{s=1}^{n} \alpha_{t,s} h_s
```

**Decoder update:**

```math
s_t = \text{RNN}(y_{t-1}, s_{t-1}, c_t)
```

### Luong Attention (Multiplicative)

**Score variants:**

| Type | Formula |
|------|---------|
| **Dot** | $e\_{t,s} = s\_t^T h\_s$ |
| **General** | $e\_{t,s} = s\_t^T W\_a h\_s$ |
| **Concat** | $e\_{t,s} = v^T \tanh(W\_a[s\_t; h\_s])$ |

---

## 📐 Training

### Teacher Forcing

During training, use ground truth $y\_{t-1}$ as input:

```math
s_t = \text{RNN}(y_{t-1}^{*}, s_{t-1}, c_t)
```

**Pros:** Faster convergence  
**Cons:** Exposure bias (train/test mismatch)

### Scheduled Sampling

Mix teacher forcing and free-running:

```math
\text{input}_t = \begin{cases}
y_{t-1}^{*} & \text{with probability } \epsilon \\
\hat{y}_{t-1} & \text{with probability } 1-\epsilon
\end{cases}
```

Decrease $\epsilon$ during training.

### Cross-Entropy Loss

```math
\mathcal{L} = -\sum_{t=1}^{m} \log P(y_t^* | y_{1:t-1}^*, x)
```

---

## 📐 Decoding Strategies

### Greedy Decoding

```math
\hat{y}_t = \arg\max_y P(y | y_{1:t-1}, x)
```

Fast but suboptimal.

### Beam Search

Maintain top-$k$ candidates at each step:

```math
\text{score}(y_{1:t}) = \sum_{i=1}^{t} \log P(y_i | y_{1:i-1}, x)
```

**Length normalization:**

```math
\text{score}_{norm} = \frac{\text{score}(y_{1:t})}{t^\alpha}
```

### Nucleus (Top-p) Sampling

Sample from smallest set with cumulative probability $\geq p$:

```math
V_p = \arg\min_V \sum_{y \in V} P(y) \geq p
```

---

## 💻 Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers, 
                           batch_first=True, bidirectional=True, dropout=dropout)
    
    def forward(self, src):

        # src: (batch, src_len)
        embedded = self.embedding(src)  # (batch, src_len, embed_dim)
        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs: (batch, src_len, 2*hidden_dim)
        # hidden: (2*n_layers, batch, hidden_dim)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        self.attn = nn.Linear(encoder_dim + decoder_dim, decoder_dim)
        self.v = nn.Linear(decoder_dim, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):

        # hidden: (batch, decoder_dim)
        # encoder_outputs: (batch, src_len, encoder_dim)
        
        src_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        attention = self.v(energy).squeeze(2)  # (batch, src_len)
        
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, encoder_dim, decoder_dim, n_layers=1, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(encoder_dim, decoder_dim)
        self.rnn = nn.LSTM(embed_dim + encoder_dim, decoder_dim, n_layers,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(encoder_dim + decoder_dim + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, cell, encoder_outputs):

        # input: (batch, 1) - single token
        # hidden: (n_layers, batch, decoder_dim)
        # encoder_outputs: (batch, src_len, encoder_dim)
        
        embedded = self.dropout(self.embedding(input))  # (batch, 1, embed_dim)
        
        # Attention
        attn_weights = self.attention(hidden[-1], encoder_outputs)  # (batch, src_len)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch, 1, encoder_dim)
        
        # RNN input: embedded + context
        rnn_input = torch.cat([embedded, context], dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        # Prediction
        prediction = self.fc(torch.cat([output, context, embedded], dim=2).squeeze(1))
        
        return prediction, hidden, cell, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):

        # src: (batch, src_len)
        # tgt: (batch, tgt_len)
        
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.vocab_size
        
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        # Encode
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # Combine bidirectional hidden states
        hidden = hidden.view(self.encoder.rnn.num_layers, 2, batch_size, -1)
        hidden = hidden[:, 0, :, :] + hidden[:, 1, :, :]  # Sum directions
        cell = cell.view(self.encoder.rnn.num_layers, 2, batch_size, -1)
        cell = cell[:, 0, :, :] + cell[:, 1, :, :]
        
        # First decoder input: <SOS> token
        input = tgt[:, 0:1]
        
        for t in range(1, tgt_len):
            output, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            
            # Teacher forcing
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            input = tgt[:, t:t+1] if use_teacher else output.argmax(1, keepdim=True)
        
        return outputs
    
    def generate(self, src, max_len=50, sos_idx=1, eos_idx=2):
        """Generate output sequence (inference)"""
        self.eval()
        
        with torch.no_grad():
            encoder_outputs, hidden, cell = self.encoder(src)
            
            # Combine bidirectional
            hidden = hidden.view(self.encoder.rnn.num_layers, 2, src.size(0), -1)
            hidden = hidden[:, 0, :, :] + hidden[:, 1, :, :]
            cell = cell.view(self.encoder.rnn.num_layers, 2, src.size(0), -1)
            cell = cell[:, 0, :, :] + cell[:, 1, :, :]
            
            input = torch.tensor([[sos_idx]]).to(self.device)
            outputs = []
            
            for _ in range(max_len):
                output, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs)
                predicted = output.argmax(1)
                outputs.append(predicted.item())
                
                if predicted.item() == eos_idx:
                    break
                
                input = predicted.unsqueeze(1)
        
        return outputs

# Beam Search
def beam_search(model, src, beam_width=5, max_len=50, sos_idx=1, eos_idx=2):
    """Beam search decoding"""
    model.eval()
    
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src)
        
        # Initialize beams: (score, sequence, hidden, cell)
        beams = [(0, [sos_idx], hidden, cell)]
        complete = []
        
        for _ in range(max_len):
            new_beams = []
            
            for score, seq, hidden, cell in beams:
                if seq[-1] == eos_idx:
                    complete.append((score, seq))
                    continue
                
                input = torch.tensor([[seq[-1]]]).to(src.device)
                output, new_hidden, new_cell, _ = model.decoder(
                    input, hidden, cell, encoder_outputs
                )
                
                log_probs = F.log_softmax(output, dim=1)
                topk_probs, topk_idx = log_probs.topk(beam_width)
                
                for prob, idx in zip(topk_probs[0], topk_idx[0]):
                    new_score = score + prob.item()
                    new_seq = seq + [idx.item()]
                    new_beams.append((new_score, new_seq, new_hidden, new_cell))
            
            # Keep top beams
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]
        
        # Add incomplete beams
        complete.extend([(s, seq) for s, seq, _, _ in beams])
        
        # Return best (length-normalized)
        best = max(complete, key=lambda x: x[0] / len(x[1]))
        return best[1]
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Seq2Seq Paper | [arXiv](https://arxiv.org/abs/1409.3215) |
| 📄 | Bahdanau Attention | [arXiv](https://arxiv.org/abs/1409.0473) |
| 📄 | Luong Attention | [arXiv](https://arxiv.org/abs/1508.04025) |
| 📄 | Beam Search | [arXiv](https://arxiv.org/abs/1702.01806) |
| 🇨🇳 | Seq2Seq详解 | [知乎](https://zhuanlan.zhihu.com/p/37148308) |

---

## 🔗 Applications

| Application | Example |
|-------------|---------|
| **Machine Translation** | English → French |
| **Summarization** | Document → Summary |
| **Dialogue** | Question → Response |
| **Speech Recognition** | Audio → Text |
| **Code Generation** | Natural language → Code |

---

⬅️ [Back: RNN](../09_rnn/README.md) | ➡️ [Next: Transformer](../11_transformer/README.md)

---

⬅️ [Back: Architectures](../../README.md)

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
