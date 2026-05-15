# LocVec — Local Vector Retrieval Engine

> High-speed, hardware-aware vector search for local RAG on consumer-grade GPUs — no cloud dependency required.

---

## Overview

LocVec is a hardware-aware vector database and retrieval library written in Python with custom CUDA backends. It is designed for low-latency Retrieval-Augmented Generation (RAG) on consumer-grade hardware with limited VRAM.

By implementing a **dynamic memory handoff system** and **optimized clustering algorithms**, LocVec enables high-speed search across large document sets — entirely on-device.

---

## Features

- **Custom CUDA Kernels** — Accelerated matrix operations for vector similarity search.
- **Dynamic VRAM Handoff** — Warm Loading cycle flushes the encoder after retrieval, freeing VRAM for LLM inference.
- **IVF Indexing** — Reduces search complexity from O(N) to O(N/K) via K-Means clustered Voronoi partitioning with dynamic K calculation.
- **Streaming LLM Inference** — Token-by-token generation from local models (e.g., Phi-3 via Ollama).
- **PDF Ingestion** — Built-in sharding and indexing pipeline for document corpora.
- **No Cloud Required** — Fully local; no API calls, no data leaving your machine.

---

## System Requirements

- NVIDIA GPU (CUDA-capable)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) — version matching your GPU drivers
- Python 3.10+

Verify your CUDA installation:

```bash
nvcc --version
```

> If the command is not recognized, add the CUDA `bin` directory to your system `PATH`.

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

> **Note:** Editable mode (`-e`) is required to trigger local compilation of the C/CUDA extensions via the `setup.py` build script.

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

# Explicitly initialize the engine with the preferred model
engine = LocalVec(model_name='all-MiniLM-L6-v2')

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

## PDF Chat Session

An interactive chat session over a PDF document. Maintains a short history window and enforces concise responses via a strict token limit.

```python
import os
import json
import fitz
import urllib.request
from locvec import LocalVec

class PDFChatSession:
    def __init__(self, pdf_path, db_name="chat_store"):
        self.pdf_path = pdf_path
        self.lv = LocalVec(db_name=db_name)
        self.history = []
        self.llm_endpoint = "http://localhost:11434/api/generate"
        self.model = "phi3"

    def ingest_pdf(self, shard_size=600):
        doc = fitz.open(self.pdf_path)
        full_text = "".join([page.get_text().replace('\n', ' ') for page in doc])
        shards = [full_text[i:i + shard_size] for i in range(0, len(full_text), shard_size)]
        self.lv.build_full_index(shards)

    def _call_llm_stream(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.2, "num_ctx": 4096, "num_predict": 150}
        }
        req = urllib.request.Request(
            self.llm_endpoint,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req) as response:
            full_response = ""
            for line in response:
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    token = chunk.get('response', '')
                    full_response += token
                    print(token, end="", flush=True)
                    if chunk.get('done'):
                        break
            return full_response

    def start_chat(self):
        while True:
            query = input("\nUser: ").strip()
            if query.lower() in ['exit', 'quit']:
                break

            _, context_shards = self.lv.search(query, top_k=2)
            history_str = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in self.history[-1:]])

            full_prompt = (
                f"SYSTEM: You are a concise technical assistant. "
                f"Answer using ONLY the context provided. "
                f"STRICT RULE: Your response MUST be 3-4 lines maximum.\n\n"
                f"CONTEXT: {context_shards}\n\n"
                f"HISTORY: {history_str}\n\n"
                f"USER QUERY: {query}\n\n"
                f"CONCISE ANSWER:"
            )
            print("\nAI: ", end="")
            answer = self._call_llm_stream(full_prompt)
            print()
            self.history.append({"q": query, "a": answer})

if __name__ == "__main__":
    PDF_FILE = "research_paper.pdf"
    session = PDFChatSession(PDF_FILE)
    session.ingest_pdf()
    session.start_chat()
```

---

### IVF vs. Brute-Force Benchmark

Compares LocVec's IVF-Voronoi search against a GPU flat (brute-force) baseline at scale. Useful for validating speedup on your specific hardware.

```python
import time
import gc
import fitz
import random
import numpy as np
import torch
import json
import urllib.request
from locvec import LocalVec

SOURCE_PDF = "research_paper.pdf"
QUERY = "Compare access latency and hardware characteristics of Constant vs Texture Memory in CUDA."
TARGET_SHARDS = 200000
OLLAMA_MODEL = "phi3"

def prepare_corpus(file_path, total_shards):
    doc = fitz.open(file_path)
    base_text = [p.get_text().replace('\n', ' ') for p in doc]
    multiplier = (total_shards // len(base_text)) + 1
    corpus = []
    for i in range(multiplier):
        for shard in base_text:
            if len(corpus) < total_shards:
                corpus.append(f"{shard} [id_{i}]")
    random.seed(42)
    random.shuffle(corpus)
    return corpus

def stream_inference(query, context):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"Context: {context}\n\nQuery: {query}\n\nTechnical Answer:",
        "stream": True,
        "options": {"temperature": 0.1, "num_predict": 350}
    }
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as response:
        for line in response:
            if line:
                chunk = json.loads(line.decode('utf-8'))
                yield chunk.get('response', '')
                if chunk.get('done'):
                    break

if __name__ == "__main__":
    corpus = prepare_corpus(SOURCE_PDF, TARGET_SHARDS)

    lv = LocalVec(db_name="benchmark_final")
    lv.build_full_index(corpus)

    time.sleep(20)  # Thermal recovery for laptop hardware

    embeddings_path = f"{lv.db_prefix}_offline_embeddings.bin"
    raw_data = np.fromfile(embeddings_path, dtype=np.float32).reshape(-1, lv.dims)
    all_vectors_gpu = torch.from_numpy(raw_data).to('cuda').half()
    q_vec_gpu = torch.from_numpy(lv.encoder.encode(QUERY)).to('cuda').half()

    # GPU Brute-Force
    torch.cuda.synchronize()
    _ = torch.norm(all_vectors_gpu[:100] - q_vec_gpu, dim=1)
    t0 = time.perf_counter()
    dist = torch.norm(all_vectors_gpu - q_vec_gpu, dim=1)
    flat_idx = torch.argmin(dist).item()
    torch.cuda.synchronize()
    latency_flat = (time.perf_counter() - t0) * 1000
    flat_context = corpus[flat_idx]

    # LocVec IVF
    t1 = time.perf_counter()
    lv_idx, lv_context = lv.search(QUERY, top_k=4)
    latency_lv = (time.perf_counter() - t1) * 1000

    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    lv.offload_encoder()

    del all_vectors_gpu
    gc.collect()
    torch.cuda.empty_cache()

    speedup = latency_flat / latency_lv
    print(f"Brute-Force: {latency_flat:.2f}ms | LocVec IVF: {latency_lv:.2f}ms | Speedup: {speedup:.2f}x | Peak VRAM: {peak_vram:.2f}MB")

    for token in stream_inference(QUERY, flat_context):
        print(token, end="", flush=True)
    print()

    for token in stream_inference(QUERY, lv_context):
        print(token, end="", flush=True)
    print()
```

---

## User API Reference

To use the vector engine in your own scripts, import the main class:

```python
from locvec import LocalVec
```

### `LocalVec(model_name='all-MiniLM-L6-v2')`

Initializes the vector engine and loads the embedding model into memory.

- `model_name` *(str, optional)*: Specify a different SentenceTransformers model from HuggingFace. Defaults to a fast, lightweight 384-dimension model (`'all-MiniLM-L6-v2'`).

---

### `engine.build_full_index(texts)`

Ingests a list of text shards and builds the optimized vector index on your GPU. It automatically calculates the best cluster distribution for your dataset size.

- `texts` *(list of str)*: A list of strings (e.g., your chunked PDF pages or text documents).

---

### `engine.search(query_text)`

Performs a high-speed, hardware-accelerated vector search against your indexed documents to find the most relevant context.

- `query_text` *(str)*: The question or prompt you want to search for.
- **Returns** *(tuple)*: `(idx, context)` — The integer ID of the best-matching chunk and the actual text string of that chunk.

---

### `engine.offload_encoder()`

Flushes the embedding model from your GPU's VRAM. Always call this before passing the retrieved context to a Local LLM. This guarantees your VRAM is completely free for generation, preventing out-of-memory crashes on consumer cards (4GB/6GB).

---

### `engine.query_llm_stream(model, query, context)`

Sends the user's query alongside the retrieved context to your local Ollama instance, returning a stream of the response.

- `model` *(str)*: The name of the Ollama model to use (e.g., `"phi3"` or `"llama3"`).
- `query` *(str)*: The user's original question.
- `context` *(str)*: The text payload returned by the `engine.search()` method.
- **Returns** *(generator)*: Yields individual string tokens for real-time console or UI streaming.

---

## How It Works

### CUDA Backend & Memory Management

Most local RAG implementations fail on consumer hardware because the embedding model and the LLM compete for the same VRAM pool. LocVec solves this with a two-phase lifecycle:

1. **Retrieval Phase** — Custom CUDA kernels handle vector similarity computations.
2. **Warm Loading / VRAM Handoff** — The encoder is flushed from GPU memory immediately after retrieval, clearing space for LLM inference and preventing Out-of-Memory (OOM) errors.

### IVF Indexing

LocVec transitions search complexity from linear to clustered using **Inverted File Indexing (IVF)**:

| Step | Description |
|---|---|
| **Clustering** | K-Means partitions the vector space into *K* Voronoi cells dynamically based on dataset size. |
| **Coarse Search** | Query is matched to the nearest centroid. |
| **Fine Search** | k-NN search is performed only within the relevant cluster. |
| **Complexity** | Reduced from O(N) → O(N/K). |

---

## Project Structure

```
locvec/
├── src/
│   ├── cuda/        # Custom CUDA kernels (k-means training, IVF search)
│   ├── bridge/      # C wrappers interfacing Python with CUDA memory management
│   └── locvec/      # Core Python library, CTypes bindings, and high-level API
├── setup.py         # Build configuration for C/CUDA extensions
├── testcase.py      # Reference implementation for PDF ingestion and search
└── README.md        # Documentation
```

---

## License

See [LICENSE](LICENSE) for details.
