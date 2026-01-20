<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=.&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=flat-square" alt="Updated"/>
</p>

---

<div align="center">

# 🧠 ML Researcher Foundations

### *Your Complete Journey from Zero to ML Research*

[![Stars](https://img.shields.io/badge/⭐_Star_this_repo-If_helpful!-yellow?style=for-the-badge)](.)
[![Learning](https://img.shields.io/badge/📚_Self--Paced-Learning-blue?style=for-the-badge)](.)
[![Topics](https://img.shields.io/badge/200+_Topics-Covered-green?style=for-the-badge)](.)
[![Visual](https://img.shields.io/badge/🎨_50+_Diagrams-Included-purple?style=for-the-badge)](.)

<br>

<img src="images/main-roadmap.svg" alt="ML Foundations Complete Roadmap" width="100%">

<br>

**🎯 Master the math. Understand the theory. Build the intuition.**

*No PhD required. Just curiosity and persistence.*

---

[📖 Start Learning](#-quick-start) • [🗺️ Roadmap](#️-learning-roadmap) • [🔥 Hot Topics](#-hot-topics) • [📂 Browse All](#-all-sections)

</div>

---

## 👋 Welcome, Future ML Researcher!

Whether you're a **CS graduate**, **self-learner**, or **industry practitioner** looking to level up, this repo is your **free, comprehensive guide** to the mathematical foundations of machine learning.

### ✨ What Makes This Different?

| Feature | Description |
|---------|-------------|
| 🎨 **Visual First** | Every concept has diagrams. We believe in *seeing* the math. |
| 💻 **Code Examples** | PyTorch/NumPy code for every formula. Theory meets practice. |
| 🎯 **Research-Focused** | Covers what you *actually* need for ML research papers. |
| 📈 **Progressive** | Start from basics, end at cutting-edge (Transformers, RLHF). |
| 🆓 **100% Free** | No paywalls. No sign-ups. Just learn. |

---

## 🚀 Quick Start

### 🤔 "Where should I begin?"

<table>
<tr>
<td width="33%">

### 🌱 **Beginner**
*New to ML math*

Start here 👇
1. [Mathematical Thinking](./01_foundations/01_mathematical_thinking/README.md)
2. [Linear Algebra Basics](./02_mathematics/01_linear_algebra/README.md)
3. [Probability Fundamentals](./03_probability_statistics/01_probability/README.md)

</td>
<td width="33%">

### 🌿 **Intermediate**  
*Know basics, want depth*

Jump to 👇
1. [Backpropagation](./06_deep_learning/02_backpropagation/README.md)
2. [Optimization Theory](./02_mathematics/03_optimization/README.md)
3. [Transformers](./06_deep_learning/03_architectures/06_transformer/README.md)

</td>
<td width="33%">

### 🌳 **Advanced**
*Prepping for research*

Explore 👇
1. [🔥 Flash Attention](./06_deep_learning/06_hot_topics/01_flash_attention/flash-attention.md)
2. [🔥 RLHF & DPO](./07_reinforcement_learning/06_applications/02_rlhf/README.md)
3. [🔥 LoRA](./06_deep_learning/06_hot_topics/02_lora/lora.md)

</td>
</tr>
</table>

---

## 📖 How to Read Research Papers

### 🎯 Your Guide to Understanding ML Papers

This repository includes **detailed mathematical derivations** to help you read research papers like an expert. Here's your complete strategy:

<details open>
<summary>📚 <b>Three-Pass Reading Strategy</b> (Click to expand)</summary>

### **Pass 1: The Quick Scan** (5-10 minutes)
- ✅ Read: Title, Abstract, Introduction, Section headings, Conclusion
- ❓ Ask: What problem? Is it novel? Is it relevant?
- ⏭️ Skip: All math, experiments, proofs

### **Pass 2: The Core Understanding** (1-2 hours)
- ✅ Read: Full introduction, Method section (focus on algorithms), Key figures
- ❓ Ask: What's the innovation? What's the main algorithm?
- ⏭️ Skip: Proofs (unless critical), Detailed experiments

### **Pass 3: The Deep Dive** (3-4 hours)
- ✅ Read: Everything including appendix and code
- ❓ Ask: Can I derive all equations? Can I reproduce this?
- 🎯 Goal: Full understanding and reproduction

</details>

<details>
<summary>🔑 <b>Common ML Paper Notation</b></summary>

| Symbol | Meaning | Example |
|--------|---------|---------|
| **x, X** | Input data | x ∈ ℝᵈ |
| **y, Y** | Output/label | y ∈ {0,1} |
| **θ, w** | Model parameters | θ ∈ ℝᵖ |
| **ℒ, L** | Loss function | ℒ(θ) = \|\|y - ŷ\|\|² |
| **∇_θ** | Gradient | ∇_θℒ = ∂ℒ/∂θ |
| **η, α** | Learning rate | θ ← θ - η∇ℒ |
| **Q, K, V** | Query, Key, Value (Transformer) | Q = XW_Q |
| **KL(P\|\|Q)** | KL divergence | Used in VAE, RLHF |

</details>

<details>
<summary>📊 <b>Key Topics with Research Connections</b></summary>

### 🎯 **Optimization** → Adam, SGD papers
- [Gradient Descent](./04_optimization/02_basic_methods/01_gradient_descent/) - **Complete convergence proof**
- Connections: Adam (2014), ResNet (2015), GPT-3 (2020)

### 🔄 **Backpropagation** → All neural network papers
- [Backpropagation](./06_deep_learning/02_backpropagation/) - **Step-by-step chain rule**
- Connections: AlexNet (2012), Transformer (2017)

### 📊 **KL Divergence** → VAE, RLHF papers
- [KL Divergence](./03_probability_statistics/03_information_theory/03_kl_divergence/) - **Complete proofs**
- Connections: VAE (2013), TRPO (2015), RLHF (2022)

### 🎯 **Constrained Optimization** → SVM, PCA papers
- [Lagrange Multipliers](./04_optimization/05_constrained_optimization/02_lagrange/) - **Why they work**
- Connections: SVM, PCA, Max Entropy

### 🔥 **Attention Mechanism** → Transformer papers
- [Transformer](./06_deep_learning/03_architectures/06_transformer/) - **Complete attention derivation**
- Connections: BERT (2018), GPT series, LLaMA (2023)

</details>

<details>
<summary>✅ <b>Paper Reading Checklist</b></summary>

**After reading a paper, you should be able to:**
- [ ] Explain it to a colleague in 2 minutes
- [ ] Draw the architecture from memory
- [ ] List 3 strengths and 3 weaknesses
- [ ] Describe when this method would/wouldn't work
- [ ] Identify the key innovation vs prior work
- [ ] Write pseudocode for the core algorithm
- [ ] Reproduce at least the main result

</details>

### 🚀 **Enhanced Sections for Paper Reading**

These sections now include **complete mathematical derivations** with NO steps skipped:

1. **[Gradient Descent](./04_optimization/02_basic_methods/01_gradient_descent/)** - Convergence proofs, momentum math, paper connections
2. **[Backpropagation](./06_deep_learning/02_backpropagation/)** - Complete derivation, gradient flow, numerical checking
3. **[KL Divergence](./03_probability_statistics/03_information_theory/03_kl_divergence/)** - Gibbs' inequality, VAE derivation, forward/reverse KL
4. **[Lagrange Multipliers](./04_optimization/05_constrained_optimization/02_lagrange/)** - Complete proofs, SVM connection, modern ML
5. **[Transformer Attention](./06_deep_learning/03_architectures/06_transformer/)** - Attention derivation, complexity analysis, Flash Attention

### 📚 **Recommended Paper Reading Order**

**Beginner Papers:**
1. LeNet (1998) - First CNN
2. AlexNet (2012) - Modern deep learning
3. ResNet (2015) - Skip connections

**Intermediate Papers:**
4. Adam (2014) - Optimization
5. Attention Is All You Need (2017) - Transformers
6. BERT (2018) - Bidirectional encoding

**Advanced Papers:**
7. GPT-2/3 (2019/2020) - Language modeling
8. Flash Attention (2022) - Efficient attention
9. LLaMA (2023) - Modern LLMs

---

## 🗺️ Learning Roadmap

> **Estimated Time: 18-22 weeks** (2-3 hours/day)

```
                        YOUR ML RESEARCH JOURNEY

+===========================================================================+
|                                                                           |
|  Week 1-2      Week 3-4      Week 5-6      Week 7-8      Week 9-10       |
|  +---------+   +---------+   +---------+   +---------+   +---------+     |
|  | Found-  |   | Linear  |   | Optimi- |   | Prob &  |   |   ML    |     |
|  | ations  |-->| Algebra |-->| zation  |-->| Stats   |-->| Theory  |     |
|  +---------+   +---------+   +---------+   +---------+   +---------+     |
|                                                               |          |
|                                                               ▼          |
|  Week 21-22    Week 19-20    Week 17-18    Week 15-16    Week 11-14     |
|  +---------+   +---------+   +---------+   +---------+   +-----------+  |
|  |Effic-ML |   | Deploy- |   | Compre- |   |   RL    |   |   Deep    |  |
|  |  (MIT)  |<--| ment    |<--| ssion   |<--|         |<--| Learning  |  |
|  +---------+   +---------+   +---------+   +---------+   +-----------+  |
|                                                                           |
+===========================================================================+

Learning Path:
--------------
1. Foundations ➡️ 2. Linear Algebra ➡️ 3. Optimization ➡️ 4. Probability
                                                                     |
                                                                     ▼
9. Effic-ML ⬅️ 8. Deployment ⬅️ 7. Compression ⬅️ 6. RL ⬅️ 5. Deep Learning ⬅️ ML Theory
```

---

## 📚 All Sections

### 🔢 1. Foundations — *Build Your Base*
> **Time: 2 weeks** | Prerequisites: High school math

<details>
<summary>📖 Click to expand</summary>

<img src="01_foundations/images/learning-path.svg" width="100%">

| Topic | What You'll Learn | Time |
|-------|-------------------|------|
| [🧠 Mathematical Thinking](./01_foundations/01_mathematical_thinking/README.md) | Abstraction, pattern recognition | 3h |
| [📝 Proof Techniques](./01_foundations/02_proof_techniques/README.md) | Direct proof, contradiction, induction | 4h |
| [⏱️ Asymptotic Analysis](./01_foundations/05_asymptotic_analysis/README.md) | Big-O notation, complexity | 3h |
| [💻 Numerical Computation](./01_foundations/06_numerical_computation/README.md) | Floating point, stability | 4h |

**Key Takeaway:** *"Think like a mathematician"*

</details>

---

### 📊 2. Mathematics — *The Language of ML*
> **Time: 4 weeks** | Prerequisites: Foundations

<details>
<summary>📖 Click to expand</summary>

<img src="02_mathematics/images/math-roadmap.svg" width="100%">

| Topic | What You'll Learn | Time |
|-------|-------------------|------|
| [📐 Linear Algebra](./02_mathematics/01_linear_algebra/README.md) | Vectors, matrices, SVD, eigenvalues | 12h |
| [📈 Calculus](./02_mathematics/02_calculus/README.md) | Gradients, Jacobian, Hessian, chain rule | 10h |
| [🎯 Optimization](./02_mathematics/03_optimization/README.md) | Convexity, GD, SGD, Adam | 10h |

**Key Takeaway:** *"Gradients are everything"*

</details>

---

### 📈 3. Probability & Statistics — *Embrace Uncertainty*
> **Time: 2 weeks** | Prerequisites: Calculus

<details>
<summary>📖 Click to expand</summary>

<img src="03_probability_statistics/01_probability/images/probability-ml-view.svg" width="100%">

| Topic | What You'll Learn | Time |
|-------|-------------------|------|
| [🎲 Probability Theory](./03_probability_statistics/01_probability/README.md) | Distributions, Bayes theorem | 8h |
| [📊 Multivariate Stats](./03_probability_statistics/02_multivariate/README.md) | Covariance, Gaussian | 6h |
| [📡 Information Theory](./03_probability_statistics/03_information_theory/README.md) | Entropy, KL divergence | 6h |

**Key Takeaway:** *"Cross-entropy is your loss function"*

</details>

---

### 🎯 4. Optimization — *The Engine of Learning*
> **Time: 3 weeks** | Prerequisites: Mathematics

<details>
<summary>📖 Click to expand</summary>

<img src="04_optimization/images/optimization-roadmap.svg" width="100%">

| Topic | What You'll Learn | Time |
|-------|-------------------|------|
| [🔢 Foundations](./04_optimization/01_foundations/README.md) | Calculus, gradients, Hessian | 6h |
| [📊 Basic Methods](./04_optimization/02_basic_methods/README.md) | Gradient descent, Newton's method | 8h |
| [⚡ Advanced Methods](./04_optimization/03_advanced_methods/README.md) | Conjugate gradient, Quasi-Newton | 6h |
| [📐 Convex Optimization](./04_optimization/04_convex_optimization/README.md) | Convex functions, duality | 8h |
| [🎯 Constrained Optimization](./04_optimization/05_constrained_optimization/README.md) | Lagrange multipliers, KKT | 8h |
| [📏 Linear Programming](./04_optimization/06_linear_programming/README.md) | Simplex, duality, interior point | 8h |
| [🔢 Integer Programming](./04_optimization/07_integer_programming/README.md) | MILP, branch & bound | 6h |
| [🤖 ML Optimization](./04_optimization/08_machine_learning/README.md) | **SGD**, **Adam**, momentum | 10h |
| [🧬 Metaheuristics](./04_optimization/09_metaheuristics/README.md) | Genetic algorithms, simulated annealing | 6h |

**Key Takeaway:** *"Training is optimization"*

</details>

---

### 🎯 5. ML Theory — *Understand Why Things Work*
> **Time: 2 weeks** | Prerequisites: Probability, Optimization

<details>
<summary>📖 Click to expand</summary>

<img src="05_ml_theory/images/ml-theory-roadmap.svg" width="100%">

| Topic | What You'll Learn | Time |
|-------|-------------------|------|
| [📚 Learning Frameworks](./05_ml_theory/01_learning_frameworks/README.md) | ERM, supervised vs unsupervised | 4h |
| [🎯 Generalization](./05_ml_theory/02_generalization/README.md) | Bias-variance, VC dimension | 8h |
| [🔮 Kernel Methods](./05_ml_theory/03_kernel_methods/README.md) | Kernel trick, SVM | 6h |

**Key Takeaway:** *"Generalization is the goal"*

</details>

---

### 🧬 6. Deep Learning — *The Modern Era*
> **Time: 4 weeks** | Prerequisites: ML Theory, Optimization

<details>
<summary>📖 Click to expand</summary>

<img src="06_deep_learning/images/dl-roadmap.svg" width="100%">

| Topic | What You'll Learn | Time |
|-------|-------------------|------|
| [🧠 Neural Networks](./06_deep_learning/01_neural_networks/README.md) | Neurons, layers, activations | 6h |
| [🔄 Backpropagation](./06_deep_learning/02_backpropagation/README.md) | Chain rule, autodiff | 6h |
| [🏗️ Architectures](./06_deep_learning/03_architectures/README.md) | CNN, RNN, **Transformer**, Diffusion, MoE | 15h |
| [⚙️ Training](./06_deep_learning/04_training/README.md) | Optimizers, normalization | 8h |
| [📈 Scaling](./06_deep_learning/05_scaling/README.md) | Distributed, mixed precision, efficient | 6h |
| [🔥 Flash Attention](./06_deep_learning/06_hot_topics/01_flash_attention/flash-attention.md) | 5x faster, O(n) memory | 3h |
| [🔥 LoRA](./06_deep_learning/06_hot_topics/02_lora/lora.md) | Parameter-efficient fine-tuning | 3h |

**Key Takeaway:** *"Attention is all you need"*

</details>

---

### 🎮 7. Reinforcement Learning — *Learning from Interaction*
> **Time: 4 weeks** | Prerequisites: Deep Learning

<details>
<summary>📖 Click to expand</summary>

<img src="07_reinforcement_learning/images/rl-roadmap.svg" width="100%">

| Topic | What You'll Learn | Time |
|-------|-------------------|------|
| [🎲 MDP](./07_reinforcement_learning/01_mdp/README.md) | States, actions, rewards | 4h |
| [💰 Value Methods](./07_reinforcement_learning/02_value_methods/README.md) | Bellman, Q-learning, DQN | 8h |
| [🎯 Policy Methods](./07_reinforcement_learning/03_policy_methods/README.md) | Policy gradient, PPO | 8h |
| [🔍 Exploration](./07_reinforcement_learning/04_exploration/README.md) | ε-greedy, UCB | 4h |
| [🌍 Model-Based](./07_reinforcement_learning/05_model_based/README.md) | World models, planning | 6h |
| [🔥 Applications](./07_reinforcement_learning/06_applications/README.md) | **RLHF**, DPO, robotics | 6h |

**Key Takeaway:** *"Reward is enough"*

</details>

---

### 🗜️ 8. Model Compression — *Make Models Small & Fast*
> **Time: 2 weeks** | Prerequisites: Deep Learning

<details>
<summary>📖 Click to expand</summary>

<img src="08_model_compression/images/compression-roadmap.svg" width="100%">

| Topic | What You'll Learn | Time |
|-------|-------------------|------|
| [🔢 Quantization](./08_model_compression/03_quantization/README.md) | INT8, INT4, QLoRA | 6h |
| [🔧 PEFT/LoRA](./08_model_compression/08_peft/README.md) | Parameter-efficient fine-tuning | 6h |
| [🎓 Distillation](./08_model_compression/04_knowledge_distillation/README.md) | Teacher-student training | 4h |
| [✂️ Pruning](./08_model_compression/02_parameter_reduction/pruning/README.md) | Weight removal | 4h |
| [🎯 MoE](./08_model_compression/06_sparsity/moe/README.md) | Mixture of Experts | 4h |
| [🛠️ Tools](./08_model_compression/10_tools/README.md) | bitsandbytes, PEFT, TensorRT | 4h |

**Key Takeaway:** *"Compression enables deployment"*

</details>

---

### ⚡ 9. Efficient ML — *MIT 6.5940 Course*
> **Time: 4 weeks** | Prerequisites: Deep Learning, Model Compression

<details>
<summary>📖 Click to expand</summary>

[![Open Course](https://img.shields.io/badge/MIT_6.5940-TinyML_Course-red?style=for-the-badge)](./09_efficient_ml/README.md)

| # | Lecture | What You'll Learn | Notebook |
|:-:|---------|-------------------|:--------:|
| 1-2 | [Intro & Basics](./09_efficient_ml/01_introduction/) | Why efficiency, FLOPs, roofline | [📓 demo.ipynb](./09_efficient_ml/01_introduction/demo.ipynb) |
| 3-4 | [Pruning](./09_efficient_ml/03_pruning_sparsity_1/) | Magnitude, lottery ticket | [📓 demo.ipynb](./09_efficient_ml/03_pruning_sparsity_1/demo.ipynb) |
| 5-6 | [Quantization](./09_efficient_ml/05_quantization_1/) | INT8, GPTQ, AWQ | [📓 demo.ipynb](./09_efficient_ml/05_quantization_1/demo.ipynb) |
| 7-8 | [NAS](./09_efficient_ml/07_neural_architecture_search_1/) | DARTS, Once-for-All | [📓 demo.ipynb](./09_efficient_ml/07_neural_architecture_search_1/demo.ipynb) |
| 9-10 | [Distillation & TinyML](./09_efficient_ml/09_knowledge_distillation/) | MCUNet, 256KB inference | [📓 demo.ipynb](./09_efficient_ml/09_knowledge_distillation/demo.ipynb) |
| 11-14 | [Efficient Training](./09_efficient_ml/11_efficient_transformers/) | FlashAttention, ZeRO, FSDP | [📓 demo.ipynb](./09_efficient_ml/11_efficient_transformers/demo.ipynb) |
| 15-18 | [Efficient Models](./09_efficient_ml/16_efficient_llms/) | LLMs, Diffusion, Vision | [📓 demo.ipynb](./09_efficient_ml/16_efficient_llms/demo.ipynb) |

**Key Takeaway:** *"Efficiency enables real-world deployment"*

[**📺 Watch Full Course on YouTube →**](https://www.youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB)

</details>

---

## 🔥 Hot Topics

> *What's trending in ML research right now*

<table>
<tr>
<td align="center" width="25%">

### 🤖 Transformers
<img src="06_deep_learning/03_architectures/06_transformer/images/attention.svg" width="100%">

[**Learn →**](./06_deep_learning/03_architectures/06_transformer/README.md)

*Foundation of GPT, BERT, LLaMA*

</td>
<td align="center" width="25%">

### ⚡ Flash Attention
<img src="06_deep_learning/06_hot_topics/01_flash_attention/images/flash-attention.svg" width="100%">

[**Learn →**](./06_deep_learning/06_hot_topics/01_flash_attention/flash-attention.md)

*5x faster, O(n) memory*

</td>
<td align="center" width="25%">

### 🎨 Diffusion
<img src="06_deep_learning/03_architectures/02_diffusion/images/diffusion-process.svg" width="100%">

[**Learn →**](./06_deep_learning/03_architectures/02_diffusion/README.md)

*Stable Diffusion, DALL-E*

</td>
<td align="center" width="25%">

### 🎯 RLHF
<img src="07_reinforcement_learning/06_applications/02_rlhf/images/rlhf-pipeline.svg" width="100%">

[**Learn →**](./07_reinforcement_learning/06_applications/02_rlhf/README.md)

*How ChatGPT is aligned*

</td>
</tr>
</table>

---

## 📝 Track Your Progress

> Copy this checklist to track your journey!

```markdown

## My ML Foundations Progress

### Foundations
- [ ] Mathematical Thinking
- [ ] Proof Techniques  
- [ ] Asymptotic Analysis
- [ ] Numerical Computation

### Mathematics
- [ ] Linear Algebra
- [ ] Calculus
- [ ] Optimization Theory

### Optimization
- [ ] Gradient Descent
- [ ] SGD & Adam ⭐
- [ ] Convex Optimization
- [ ] Constrained Optimization

### Probability & Statistics
- [ ] Probability Theory
- [ ] Multivariate Statistics
- [ ] Information Theory

### ML Theory
- [ ] Learning Frameworks
- [ ] Generalization
- [ ] Kernel Methods

### Deep Learning
- [ ] Neural Networks
- [ ] Backpropagation
- [ ] CNN
- [ ] RNN/LSTM
- [ ] Transformers ⭐
- [ ] Diffusion
- [ ] Training Techniques

### Reinforcement Learning
- [ ] MDP
- [ ] Value Methods
- [ ] Policy Methods
- [ ] RLHF ⭐

### Model Compression
- [ ] Quantization (INT4/INT8)
- [ ] LoRA / PEFT ⭐
- [ ] Knowledge Distillation
- [ ] Pruning
- [ ] MoE

### Efficient ML (MIT 6.5940) ⭐
- [ ] Pruning & Sparsity
- [ ] Quantization (GPTQ, AWQ)
- [ ] Neural Architecture Search
- [ ] Knowledge Distillation
- [ ] MCUNet & TinyML
- [ ] Efficient Transformers
- [ ] Distributed Training
- [ ] Efficient LLMs & Diffusion

### 🏆 Production Ready!
```

---

## 💡 Study Tips

<table>
<tr>
<td width="50%">

### ✅ Do This

- 📝 **Take notes** in your own words
- 💻 **Code every formula** you see
- 🔁 **Revisit topics** - spaced repetition works
- 🗣️ **Explain to others** (or rubber duck)
- 📊 **Draw diagrams** by hand

</td>
<td width="50%">

### ❌ Avoid This

- 🏃 **Rushing** through topics
- 📺 **Passive reading** without practice
- 🎯 **Perfectionism** - 80% is fine, move on
- 🏝️ **Isolation** - join ML communities
- 😰 **Imposter syndrome** - everyone starts somewhere

</td>
</tr>
</table>

---

## 🛠️ Recommended Tools

| Tool | Use Case | Link |
|------|----------|------|
| 🐍 Python + PyTorch | Implementing concepts | [pytorch.org](https://pytorch.org) |
| 📓 Jupyter Notebooks | Interactive learning | [jupyter.org](https://jupyter.org) |
| 📐 Desmos | Visualizing functions | [desmos.com](https://desmos.com) |
| 🧮 WolframAlpha | Checking calculations | [wolframalpha.com](https://wolframalpha.com) |
| 🎨 Excalidraw | Drawing diagrams | [excalidraw.com](https://excalidraw.com) |

---

## 📖 Companion Resources

### Books (Free Online)
- 📘 [Mathematics for Machine Learning](https://mml-book.github.io/) - Deisenroth et al.
- 📗 [Deep Learning](https://www.deeplearningbook.org/) - Goodfellow et al.
- 📙 [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html) - Sutton & Barto

### Courses
- 🎓 [Stanford CS229](https://cs229.stanford.edu/) - Machine Learning
- 🎓 [Stanford CS231n](http://cs231n.stanford.edu/) - CNNs
- 🎓 [Stanford CS224n](https://web.stanford.edu/class/cs224n/) - NLP

---

## 🤝 Community

Learning alone is hard. Join these communities:

- 💬 [r/MachineLearning](https://reddit.com/r/MachineLearning)
- 💬 [r/learnmachinelearning](https://reddit.com/r/learnmachinelearning)
- 🐦 ML Twitter/X
- 💼 LinkedIn ML groups

---

## ⭐ Support This Project

If this repo helped you:

1. **⭐ Star this repo** - It helps others find it!
2. **🔄 Share** with colleagues and friends
3. **🐛 Report issues** if you find errors
4. **💡 Suggest topics** you want covered

---

<div align="center">

### 🎯 Ready to Begin?

**[Start with Foundations →](./01_foundations/README.md)**

</div>

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

