# LocVec — Local Vector Retrieval Engine

> High-speed, hardware-aware vector search for local RAG on consumer-grade GPUs — no cloud dependency required.

---

## Overview

LocVec is a hardware-aware vector database and retrieval library written in Python with custom CUDA backends. It is designed for low-latency Retrieval-Augmented Generation (RAG) on consumer-grade hardware with limited VRAM.

By implementing a **dynamic memory handoff system** and **optimized clustering algorithms**, LocVec enables high-speed search across large document sets — entirely on-device.

---

## Features

- **Custom CUDA Kernels** — Accelerated matrix operations for vector similarity search
- **Dynamic VRAM Handoff** — Warm Loading cycle flushes the encoder after retrieval, freeing VRAM for LLM inference
- **IVF Indexing** — Reduces search complexity from O(N) to O(N/K) via K-Means clustered Voronoi partitioning
- **Streaming LLM Inference** — Token-by-token generation from local models (e.g. Phi-3 via Ollama)
- **PDF Ingestion** — Built-in sharding and indexing pipeline for document corpora
- **No Cloud Required** — Fully local; no API calls, no data leaving your machine

---

## System Requirements

- NVIDIA GPU (CUDA-capable)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) — version matching your GPU drivers
- Python 3.8+

Verify your CUDA installation:

```bash
nvcc --version
```

If the command is not recognized, add the CUDA `bin` directory to your system `PATH`.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/rAdvirtua/locvec.git
cd locvec

# Install required dependencies
pip install -r requirements.txt

# Compile and install the library (editable mode)
pip install -e .
```

> **Note:** Editable mode (`-e`) is required to trigger local compilation of the C/CUDA extensions.

---

## Setting Up a Local LLM with Ollama

LocVec uses [Ollama](https://ollama.com) to run LLMs locally. Follow these steps to get a model running before querying with LocVec.

### 1. Install Ollama

Download and install Ollama for your platform from [ollama.com/download](https://ollama.com/download), or via the terminal:

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
```

For Windows, use the installer from the website.

### 2. Pull a Model

Once Ollama is installed, pull a model. Phi-3 is recommended for low-VRAM setups:

```bash
# Lightweight — good for 4–6 GB VRAM
ollama pull phi3

# Alternatively, for higher quality output
ollama pull llama3
ollama pull mistral
```

Browse the full model library at [ollama.com/library](https://ollama.com/library).

### 3. Verify the Model is Running

```bash
ollama run phi3
```

You should see an interactive prompt. Type `/bye` to exit. Ollama runs as a background service automatically, so no manual server start is needed before using LocVec.

> **Tip:** Match your model choice to available VRAM. On GPUs with less than 6 GB, stick to `phi3` or `gemma:2b` to avoid OOM errors during the inference phase.

---

## Usage

The usage example requires PyMuPDF for PDF text extraction. Install it with:

```bash
pip install pymupdf
```

```python
import fitz
import time
import os
from locvec import LocalVec

def extract_and_chunk_pdf(file_path, chunk_size=300):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

engine = LocalVec()

pdf_path = "research_paper.pdf"
chunks = extract_and_chunk_pdf(pdf_path)

print(f"Extracted {len(chunks)} shards from {pdf_path}")
engine.build_full_index(chunks)

query = "Summarize the key findings of this document."
start_time = time.perf_counter()

idx, context = engine.search(query)

if idx < 0:
    print(f"Error: {context}")
    exit()

latency_ms = (time.perf_counter() - start_time) * 1000
print(f"Search completed in {latency_ms:.2f}ms")

engine.offload_encoder()

print("\nAI Response:")
for token in engine.query_llm_stream("phi3", query, context):
    print(token, end="", flush=True)
print("\n")
```

---

## How It Works

### CUDA Backend & Memory Management

Most local RAG implementations fail on consumer hardware because the embedding model and the LLM compete for the same VRAM pool. LocVec solves this with a two-phase lifecycle:

1. **Retrieval Phase** — Custom CUDA kernels handle vector similarity computations
2. **Warm Loading / VRAM Handoff** — The encoder is flushed from GPU memory immediately after retrieval, clearing space for LLM inference and preventing Out-of-Memory (OOM) errors

### IVF Indexing

LocVec transitions search complexity from linear to clustered using **Inverted File Indexing (IVF)**:

| Step | Description |
|---|---|
| **Clustering** | K-Means partitions the vector space into *K* Voronoi cells |
| **Coarse Search** | Query is matched to the nearest centroid |
| **Fine Search** | k-NN search is performed only within the relevant cluster |
| **Complexity** | Reduced from O(N) → O(N/K) |

---

## Project Structure

```
locvec/
├── src/
│   ├── cuda/        # Custom CUDA kernels (k-means training, IVF search)
│   ├── bridge/      # C wrappers interfacing Python with CUDA
│   └── locvec/      # Core Python library and high-level API
└── setup.py         # Build configuration for C/CUDA extensions
```

---

## License

See [LICENSE](LICENSE) for details.
