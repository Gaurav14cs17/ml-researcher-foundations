<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Stochastic%20Gradient%20Descent%20SG&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# Stochastic Gradient Descent (SGD)

> **The workhorse of deep learning**

---

## 📐 Algorithm

```
For each step:
    Sample mini-batch B ⊂ {1, ..., n}
    g = (1/|B|) Σᵢ∈B ∇fᵢ(x)   # Stochastic gradient
    x = x - α·g

Key: E[g] = ∇f(x) (unbiased estimate)
```

---

## 📊 SGD vs GD

| Aspect | GD | SGD |
|--------|----|----|
| Per-step cost | O(n) | O(batch_size) |
| Convergence | Deterministic | Noisy |
| Generalization | Worse | Better! |
| Memory | High | Low |

---

## 🔑 Why SGD Works

```
1. Computational: Much faster per iteration
2. Implicit regularization: Noise helps generalization
3. Escaping saddles: Noise helps escape bad regions
```

---

## 💻 Code

```python
def sgd(model, data, lr=0.01, batch_size=32, epochs=10):
    for epoch in range(epochs):
        for batch in get_batches(data, batch_size):
            loss = model.forward(batch)
            grads = model.backward()
            
            for param, grad in zip(model.params, grads):
                param -= lr * grad
```

---

## 🌍 Where Used

| Model | Optimizer |
|-------|-----------|
| ResNet | SGD + Momentum |
| BERT | AdamW |
| GPT | AdamW |

---

---

⬅️ [Back: Adam](./adam.md)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
