<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Prompt%20Engineering&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/prompt-engineering-complete.svg" width="100%">

*Caption: Prompt engineering optimizes how we communicate with LLMs. Techniques include few-shot learning, chain-of-thought, and structured prompts.*

---

## 📐 Key Techniques

```
Zero-shot: Direct question without examples
Few-shot: Provide examples in prompt
Chain-of-Thought: "Let's think step by step"
ReAct: Reasoning + Acting
Self-Consistency: Sample multiple times, vote

Prompt Structure:
1. System instruction
2. Context/examples
3. User query
4. Output format specification
```

---

## 💻 Examples

```python
# Zero-shot
prompt = "Classify this review as positive or negative: 'Great product!'"

# Few-shot
prompt = """
Classify reviews:
"Love it!" -> positive
"Terrible quality" -> negative
"Amazing experience" -> positive
"Great product!" -> """

# Chain-of-Thought
prompt = """
Q: If I have 5 apples and buy 3 more, how many do I have?
A: Let's think step by step.
   1. I start with 5 apples
   2. I buy 3 more
   3. 5 + 3 = 8
   Answer: 8 apples
"""
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Chain-of-Thought | [arXiv](https://arxiv.org/abs/2201.11903) |
| 📖 | Prompt Engineering Guide | [Guide](https://www.promptingguide.ai/) |
| 🇨🇳 | 提示工程详解 | [知乎](https://zhuanlan.zhihu.com/p/626895883) |

---

⬅️ [Back: 15-RAG](../15-retrieval-augmented/) | ➡️ [Back: Deep Learning](../)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
