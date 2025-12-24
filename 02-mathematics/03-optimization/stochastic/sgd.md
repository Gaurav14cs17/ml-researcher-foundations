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
