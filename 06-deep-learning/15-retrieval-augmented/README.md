<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=15 RAG&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 📚 Retrieval-Augmented Generation

> **Enhancing LLMs with external knowledge**

---

## 🎯 Visual Overview

<img src="./images/rag-complete.svg" width="100%">

*Caption: RAG retrieves relevant documents and adds them to the LLM context. This allows using external knowledge without retraining.*

---

## 📐 RAG Pipeline

```
1. Query: User question
2. Retrieve: Find relevant documents from knowledge base
   • Embed query and documents
   • Vector similarity search
3. Augment: Add retrieved context to prompt
4. Generate: LLM generates answer using context

Key components:
• Embedding model (sentence transformers)
• Vector database (FAISS, Pinecone)
• LLM (GPT, Llama)
```

---

## 💻 Code Example

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vectorstore.as_retriever(k=3),
    chain_type="stuff"
)

# Query
answer = qa_chain.run("What is machine learning?")
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | RAG Paper | [arXiv](https://arxiv.org/abs/2005.11401) |
| 🇨🇳 | RAG详解 | [知乎](https://zhuanlan.zhihu.com/p/636117667) |

---

⬅️ [Back: 14-Continual](../14-continual-learning/) | ➡️ [Next: 16-Prompting](../16-prompt-engineering/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

