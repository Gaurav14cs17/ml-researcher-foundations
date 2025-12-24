<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Deployment&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 🚀 Deployment Considerations

> **Getting compressed models into production**

<img src="./images/deployment-visual.svg" width="100%">

---

## 📐 Mathematical Foundations

### Latency Estimation
```
Latency = compute_time + memory_time

compute_time ∝ FLOPs / peak_throughput
memory_time ∝ model_size / memory_bandwidth

For LLMs (memory bound):
Latency ≈ params × bytes_per_param / bandwidth
```

### Throughput (tokens/sec)
```
Batch processing:
throughput = batch_size / latency_per_batch

KV cache size:
cache = 2 × layers × heads × head_dim × seq_len × batch_size × bytes
```

### Memory Requirement
```
Inference memory:
= model_params + KV_cache + activations

For LLaMA-7B INT4:
= 3.5 GB + KV + activations ≈ 5-6 GB total
```

---

## 📂 Topics

| File | Topic | Platform |
|------|-------|----------|

---

## 📊 Deployment Targets

| Target | Memory | Compute | Framework |
|--------|--------|---------|-----------|
| **Mobile (iOS)** | 2-4 GB | Limited | CoreML |
| **Mobile (Android)** | 2-6 GB | Limited | TFLite |
| **Edge (IoT)** | 256 MB-1 GB | Very limited | TFLite Micro |
| **Server (CPU)** | 32+ GB | Multi-core | ONNX Runtime |
| **Server (GPU)** | 16-80 GB | High | TensorRT |
| **Cloud (TPU)** | 16+ GB | Very high | JAX |

---

## 🔥 LLM Deployment Stack

```
+-------------------------------------------------------------+
|                   LLM Deployment Options                     |
+-------------------------------------------------------------+
|                                                              |
|  Local CPU:                                                  |
|  +-- llama.cpp (GGML/GGUF quantization)                     |
|                                                              |
|  Local GPU:                                                  |
|  +-- vLLM, TensorRT-LLM, text-generation-inference         |
|                                                              |
|  Cloud:                                                      |
|  +-- OpenAI API, Anthropic API, HuggingFace Inference      |
|                                                              |
|  Serverless:                                                 |
|  +-- Modal, Replicate, Banana                               |
|                                                              |
+-------------------------------------------------------------+
```

---

## 💻 Quick Deployment Examples

### Mobile (TFLite)

```python
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # INT8
tflite_model = converter.convert()
```

### Server (TensorRT)

```bash
# Convert ONNX to TensorRT
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
```

### Local LLM (llama.cpp)

```bash
# Run quantized model
./main -m llama-7b-q4.gguf -p "Hello" -n 100
```

---

## 🔗 Where This Topic Is Used

| Deployment | Tools Used |
|------------|-----------|
| **ChatGPT-like local** | llama.cpp, Ollama |
| **Production API** | vLLM, TensorRT-LLM |
| **Mobile vision** | TFLite, CoreML |
| **Edge AI** | OpenVINO, TFLite Micro |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | TensorFlow Lite | [Docs](https://www.tensorflow.org/lite) |
| 📖 | TensorRT | [NVIDIA](https://developer.nvidia.com/tensorrt) |
| 📖 | llama.cpp | [GitHub](https://github.com/ggerganov/llama.cpp) |
| 📖 | vLLM | [GitHub](https://github.com/vllm-project/vllm) |
| 🇨🇳 | 模型部署指南 | [知乎](https://zhuanlan.zhihu.com/p/628120082) |

---

⬅️ [Back: Compression Pipelines](../) | ➡️ [Next: Workflows](../workflows/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
