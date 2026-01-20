<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=120&section=header&text=Lecture%2016%20Efficient%20LLMs&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-09-1ABC9C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

# Lecture 16: Efficient Large Language Models

[← Back to Course](../) | [← Previous](../15_efficient_vision_models/) | [Next: Efficient Diffusion →](../17_efficient_diffusion_models/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09_efficient_ml/16_efficient_llms/demo.ipynb) ← **Try the code!**

---

## 📺 Video Lecture

| Resource | Link |
|----------|------|
| 🎥 **Lecture Video** | [Watch on YouTube](https://www.youtube.com/watch?v=v5CgSOL4GlM&list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&index=16) |
| 📊 **Slides** | [MIT Course Page](https://hanlab.mit.edu/courses/2023-fall-65940) |
| ⏱️ **Duration** | ~90 minutes |

### 📝 Video Summary

This lecture covers **efficient inference for large language models**:

- **LLM inference characteristics**: Memory-bound, not compute-bound

- **KV Cache optimization**: Essential for autoregressive generation

- **Speculative decoding**: Using small model to accelerate large model

- **PagedAttention**: vLLM's memory management innovation

- **Continuous batching**: Maximizing throughput

- **Serving optimization stack**: From hardware to algorithms

> 💡 *"LLM inference is fundamentally memory-bound—optimizing memory access is more important than reducing FLOPs."* — Prof. Song Han

---

![Overview](overview.png)

## LLM Inference Characteristics

LLM inference is **memory-bound**, not compute-bound:

```
Batch size 1, seq_len 2048:

- Compute: Matrix ops (fast)

- Memory: Load 7B weights from HBM (slow!)

Arithmetic Intensity = FLOPs / Bytes loaded < 1

```

---

## 📐 Mathematical Foundations & Proofs

### Autoregressive Generation Complexity

**Per-token generation:**

1. **Prefill (first token):** Process all input tokens

```math
\text{FLOPs}_{prefill} = 2 \times N_{prompt} \times N_{params}

```math

2. **Decode (subsequent tokens):** Generate one token at a time

```

\text{FLOPs}_{decode} = 2 \times N_{params}

```

**Total for generating \( T \) tokens:**

```math
\text{FLOPs}_{total} = 2 \times N_{params} \times (N_{prompt} + T)

```

---

### KV Cache Memory Analysis

**Without KV cache:**
Each new token requires recomputing all K, V:

```math
\text{FLOPs}_{no\_cache} = O(N^2 \times d \times L)

```

**With KV cache:**
Only compute K, V for new token:

```math
\text{FLOPs}_{cache} = O(N \times d \times L)

```

**Speedup:** \( N \times \) (sequence length).

**Memory cost:**

```math
M_{KV} = 2 \times L \times N \times d \times b

```

**Example (LLaMA-7B):**
- L = 32 layers, N = 4096 context, d = 4096, b = 2 (FP16)

- \( M_{KV} = 2 \times 32 \times 4096 \times 4096 \times 2 = 2.1 \text{ GB} \)

---

### Speculative Decoding

**Algorithm:**

1. Draft model generates K tokens quickly

2. Target model verifies all K in parallel

3. Accept matching prefix, resample from target if mismatch

**Acceptance probability:**

For draft probability \( p(x) \) and target probability \( q(x) \):

```math
P_{accept} = \min\left(1, \frac{q(x)}{p(x)}\right)

```

**Expected accepted tokens:**

```math
\mathbb{E}[\text{accepted}] = \sum_{k=1}^{K} P(\text{accept all } k)

```

**Speedup formula:**

```math
\text{Speedup} = \frac{\mathbb{E}[\text{accepted}] + 1}{1 + K \times \frac{t_{draft}}{t_{target}}}

```

For good draft model (\( P_{accept} \approx 0.8 \)), K=5:

```math
\text{Speedup} \approx 2-3\times

```

---

### PagedAttention Memory Management

**Problem:** KV cache with variable sequence lengths wastes memory.

**Standard allocation:**
Pre-allocate \( N_{max} \) for each request → memory waste.

**PagedAttention:**
Allocate in fixed-size pages (e.g., 16 tokens).

**Memory utilization:**

```math
\text{Utilization}_{paged} = \frac{\sum_i N_i}{K \times P}

```

where \( K \) = number of requests, \( P \) = page size.

Approaches 100% vs ~50% for standard allocation.

---

### Continuous Batching

**Static batching:**
Wait for B requests, process together, wait for all to finish.

**Continuous batching:**
As requests finish, add new ones immediately.

**Throughput improvement:**

```math
\text{Throughput}_{continuous} = \frac{N_{requests}}{T_{total}}

```

vs.

```math
\text{Throughput}_{static} = \frac{B}{\max_i T_i}

```

**Improvement:** 2-4× for variable-length requests.

---

### Arithmetic Intensity Analysis

**Matrix multiply \( Y = XW \):**

```math
I = \frac{2 \times m \times n \times k}{(m \times k + k \times n + m \times n) \times b}

```

**For batch size 1:**

```math
I = \frac{2nk}{(k + kn + n) \times 2} \approx 1

```

**Memory-bound!**

**For batch size B:**

```math
I = \frac{2Bnk}{(Bk + kn + Bn) \times 2} \approx B

```

**Compute-bound for large B.**

---

## 🧮 Key Derivations

### Token Generation Latency

**Time per token:**

```math
t_{token} = \max\left(\frac{\text{FLOPs}_{decode}}{\text{FLOPS}_{hardware}}, \frac{M_{weights} + M_{KV}}{\text{BW}_{memory}}\right)

```

For A100 (312 TFLOPS, 2TB/s BW), LLaMA-7B:

- Compute: \( 2 \times 7B / 312T = 45\mu s \)

- Memory: \( 14GB / 2TB/s = 7ms \)

**Memory-bound by 150×!**

---

### MQA/GQA Memory Savings

| Method | KV Heads | KV Cache Size |
|--------|----------|--------------|
| MHA | H | \( 2LNdH \) |
| GQA | G | \( 2LNdG \) |
| MQA | 1 | \( 2LNd \) |

**MQA reduction:** \( H \times \) smaller KV cache.

**LLaMA-2 (GQA with 8 groups, 32 query heads):**
4× KV cache reduction.

---

### Batching Efficiency

**GPU utilization:**

```math
\eta = \frac{\text{Compute time}}{\text{Total time}} = \frac{t_{compute}}{t_{compute} + t_{memory}}

```

**With batching:**

```math
\eta_B = \frac{B \times t_{compute}}{B \times t_{compute} + t_{memory}}

```

As \( B \to \infty \): \( \eta_B \to 1 \)

**Optimal batch size:** Where compute time ≈ memory time.

---

## 💻 Code Examples

### Speculative Decoding

```python
import torch
import torch.nn.functional as F

def speculative_decode(target_model, draft_model, prompt, max_tokens=100, K=5):
    """
    Speculative decoding: Use small draft model to accelerate large target model
    
    K: number of tokens to draft before verification
    """
    tokens = prompt.clone()
    
    while tokens.shape[1] < max_tokens:
        # Draft: Generate K tokens with small model
        draft_tokens = tokens.clone()
        draft_probs = []
        
        for _ in range(K):
            with torch.no_grad():
                logits = draft_model(draft_tokens)[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                draft_probs.append(probs)
                next_token = torch.multinomial(probs, 1)
                draft_tokens = torch.cat([draft_tokens, next_token], dim=1)
        
        # Verify: Run target model on all K+1 positions at once
        with torch.no_grad():
            target_logits = target_model(draft_tokens)
            target_probs = F.softmax(target_logits[:, -(K+1):-1, :], dim=-1)
        
        # Accept or reject each token
        n_accepted = 0
        for i in range(K):
            draft_token = draft_tokens[:, tokens.shape[1] + i]
            p_draft = draft_probs[i][:, draft_token].squeeze()
            p_target = target_probs[:, i, draft_token].squeeze()
            
            # Acceptance probability: min(1, p_target/p_draft)
            accept_prob = torch.minimum(
                torch.ones_like(p_target),
                p_target / (p_draft + 1e-10)
            )
            
            if torch.rand(1) < accept_prob:
                n_accepted += 1
            else:
                # Resample from adjusted distribution
                adjusted_probs = F.relu(target_probs[:, i] - draft_probs[i])
                adjusted_probs = adjusted_probs / adjusted_probs.sum(dim=-1, keepdim=True)
                new_token = torch.multinomial(adjusted_probs, 1)
                tokens = torch.cat([tokens, draft_tokens[:, tokens.shape[1]:tokens.shape[1]+n_accepted], new_token], dim=1)
                break
        else:
            # All accepted, sample one more from target
            final_probs = F.softmax(target_logits[:, -1, :], dim=-1)
            final_token = torch.multinomial(final_probs, 1)
            tokens = torch.cat([tokens, draft_tokens[:, tokens.shape[1]:], final_token], dim=1)
    
    return tokens

# PagedAttention (simplified)
class PagedKVCache:
    """
    Paged KV cache for memory-efficient LLM serving
    """
    def __init__(self, page_size=16, max_pages=1000, head_dim=128, n_heads=32, n_layers=32):
        self.page_size = page_size
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Page pool (pre-allocated memory)
        self.k_pages = torch.zeros(max_pages, n_layers, n_heads, page_size, head_dim)
        self.v_pages = torch.zeros(max_pages, n_layers, n_heads, page_size, head_dim)
        
        # Free page list
        self.free_pages = list(range(max_pages))
        
        # Page table per request: request_id -> list of page indices
        self.page_tables = {}
    
    def allocate_page(self, request_id):
        """Allocate a new page for a request"""
        if not self.free_pages:
            raise RuntimeError("Out of pages!")
        
        page_idx = self.free_pages.pop()
        
        if request_id not in self.page_tables:
            self.page_tables[request_id] = []
        self.page_tables[request_id].append(page_idx)
        
        return page_idx
    
    def free_request(self, request_id):
        """Free all pages for a completed request"""
        if request_id in self.page_tables:
            self.free_pages.extend(self.page_tables[request_id])
            del self.page_tables[request_id]
    
    def get_kv(self, request_id, layer_idx):
        """Get full KV cache for a request"""
        pages = self.page_tables.get(request_id, [])
        if not pages:
            return None, None
        
        k = torch.cat([self.k_pages[p, layer_idx] for p in pages], dim=1)
        v = torch.cat([self.v_pages[p, layer_idx] for p in pages], dim=1)
        return k, v

# Continuous Batching
class ContinuousBatcher:
    """
    Continuous batching for LLM serving
    """
    def __init__(self, model, max_batch_size=32):
        self.model = model
        self.max_batch_size = max_batch_size
        self.active_requests = {}  # request_id -> {tokens, kv_cache, ...}
        self.pending_requests = []
    
    def add_request(self, request_id, prompt_tokens):
        """Add new request to pending queue"""
        self.pending_requests.append({
            'id': request_id,
            'tokens': prompt_tokens,
            'generated': 0,
            'max_tokens': 100
        })
    
    def step(self):
        """Process one generation step for all active requests"""
        # Add pending requests if space available
        while self.pending_requests and len(self.active_requests) < self.max_batch_size:
            req = self.pending_requests.pop(0)
            self.active_requests[req['id']] = req
        
        if not self.active_requests:
            return []
        
        # Batch all active requests
        # In practice, this requires careful padding/packing
        completed = []
        
        for req_id, req in list(self.active_requests.items()):
            # Generate one token (simplified)
            with torch.no_grad():
                logits = self.model(req['tokens'])[:, -1, :]
                next_token = logits.argmax(dim=-1, keepdim=True)
                req['tokens'] = torch.cat([req['tokens'], next_token], dim=1)
                req['generated'] += 1
            
            # Check completion
            if req['generated'] >= req['max_tokens']:
                completed.append((req_id, req['tokens']))
                del self.active_requests[req_id]
        
        return completed

```

---

## 🎯 Where Used

| Concept | Applications |
|---------|-------------|
| KV Cache | All autoregressive LLMs |
| Speculative Decoding | Production LLM serving |
| MQA/GQA | LLaMA 2, Mistral, Falcon |
| PagedAttention | vLLM, TensorRT-LLM |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [← Efficient Vision Models](../15_efficient_vision_models/README.md) | [Efficient ML](../README.md) | [Efficient Diffusion →](../17_efficient_diffusion_models/README.md) |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | FlashAttention | [arXiv](https://arxiv.org/abs/2205.14135) |
| 📄 | vLLM/PagedAttention | [arXiv](https://arxiv.org/abs/2309.06180) |
| 📄 | Speculative Decoding | [arXiv](https://arxiv.org/abs/2211.17192) |
| 📄 | AWQ | [arXiv](https://arxiv.org/abs/2306.00978) |
| 🎥 | MIT 6.5940 TinyML | [Course](https://hanlab.mit.edu/courses/2024-fall-65940) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=80&section=footer" width="100%"/>
</p>
