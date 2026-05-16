import ctypes
import numpy as np
import json
import urllib.request
import os
import sys
import torch
import gc
from sentence_transformers import SentenceTransformer

class LocalVec:
    def __init__(self, db_name="default_db", model_name='all-MiniLM-L6-v2', storage_dir="."):
        os.makedirs(storage_dir, exist_ok=True)
        self.db_prefix = os.path.join(storage_dir, db_name)
        self.map_path = f"{self.db_prefix}_map.json"
        self.c_prefix = self.db_prefix.encode('utf-8')

        base_path = os.path.dirname(__file__)
        ext = ".dll" if sys.platform == "win32" else ".so"
        self.lib_path = os.path.join(base_path, f"liblocalvec{ext}")
        
        if not os.path.exists(self.lib_path):
            raise FileNotFoundError(f"LocVec Core not found at {self.lib_path}")
        self.lib = ctypes.CDLL(self.lib_path)

        self.lib.init_engine.argtypes = [ctypes.c_int, ctypes.c_char_p]
        self.lib.init_engine.restype = ctypes.c_int
        self.lib.cleanup_engine.restype = None
        self.lib.vector_search.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, ndim=1), ctypes.c_int, ctypes.c_char_p]
        self.lib.vector_search.restype = ctypes.c_int
        self.lib.train_index.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self.lib.train_index.restype = ctypes.c_int
        self.lib.build_index.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
        self.lib.build_index.restype = ctypes.c_int

        self.encoder = SentenceTransformer(model_name)
        self.dims = self.encoder.get_embedding_dimension()

        self.lib.init_engine(self.dims, self.c_prefix)
        self.db = {}
        self.refresh_map()

    def refresh_map(self):
        if os.path.exists(self.map_path):
            with open(self.map_path, "r") as f:
                self.db = json.load(f)

    def build_full_index(self, texts, max_iter=100):
        n = len(texts)
        k = max(1, int(np.sqrt(n)))
        embeddings = self.encoder.encode(texts, show_progress_bar=True).astype(np.float32)
        embeddings.tofile(f"{self.db_prefix}_offline_embeddings.bin")

        self.db = {str(i): text for i, text in enumerate(texts)}
        with open(self.map_path, "w") as f:
            json.dump(self.db, f)

        if self.lib.train_index(k, self.dims, self.c_prefix, max_iter) != 0: return False
        if self.lib.build_index(k, self.dims, self.c_prefix) != 0: return False
        
        self.lib.cleanup_engine()
        self.lib.init_engine(self.dims, self.c_prefix)
        self.refresh_map()
        return True

    def search(self, query_text, top_k=3):
        """
        Retrieves the best match from CUDA and expands the window to top_k 
        surrounding chunks to provide deep technical context.
        """
        query_vector = self.encoder.encode(query_text).astype(np.float32)
        center_idx = self.lib.vector_search(query_vector, self.dims, self.c_prefix)
        
        if center_idx < 0:
            return center_idx, "Search Core Error"

        
        context_chunks = []
        for i in range(center_idx - 1, center_idx + top_k - 1):
            chunk = self.db.get(str(i))
            if chunk:
                context_chunks.append(f"[Source Chunk {i}]: {chunk}")

        merged_context = "\n\n".join(context_chunks)
        return center_idx, merged_context

    def query_llm_stream(self, model, query, context, api_url="http://localhost:11434/api/generate", **kwargs):
        # Improved Prompt Engineering for Technical accuracy
        prompt = (
            f"### SYSTEM INSTRUCTIONS\n"
            f"You are a specialized technical assistant. Use the provided context"
            f"to answer the query accurately.\n"
            f"If the context is insufficient, state that clearly. Do not hallucinate details.\n\n"
            f"### CONTEXT\n{context}\n\n"
            f"### USER QUERY\n{query}\n\n"
            f"### TECHNICAL RESPONSE:"
        )
        
        options = {"temperature": 0.1, "num_predict": 512}
        options.update(kwargs)

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": options
        }
        
        req = urllib.request.Request(
            api_url,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                for line in response:
                    if line:
                        chunk = json.loads(line.decode('utf-8'))
                        yield chunk.get('response', '')
                        if chunk.get('done'): break
        except Exception as e:
            yield f"\n[Inference Error]: {str(e)}"

    def offload_encoder(self):
        self.encoder.to('cpu')
        torch.cuda.empty_cache()
        gc.collect()

    def __del__(self):
        try:
            self.lib.cleanup_engine()
        except:
            pass
