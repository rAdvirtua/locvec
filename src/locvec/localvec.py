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
        # 1. Dynamic Paths and Prefix Generation
        os.makedirs(storage_dir, exist_ok=True)
        self.db_prefix = os.path.join(storage_dir, db_name)
        self.map_path = f"{self.db_prefix}_map.json"
        
        # C needs byte strings, not Python strings
        self.c_prefix = self.db_prefix.encode('utf-8')

        # 2. Library Load
        base_path = os.path.dirname(__file__)
        ext = ".dll" if sys.platform == "win32" else ".so"
        self.lib_path = os.path.join(base_path, f"liblocalvec{ext}")
        
        if not os.path.exists(self.lib_path):
            raise FileNotFoundError(f"LocVec Core not found at {self.lib_path}")
        self.lib = ctypes.CDLL(self.lib_path)

        # 3. Dynamic Signatures (Now with c_char_p for string paths)
        self.lib.init_engine.argtypes = [ctypes.c_int, ctypes.c_char_p]
        self.lib.init_engine.restype = ctypes.c_int
        
        self.lib.cleanup_engine.restype = None
        
        self.lib.vector_search.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, ndim=1), ctypes.c_int, ctypes.c_char_p]
        self.lib.vector_search.restype = ctypes.c_int
        
        self.lib.train_index.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self.lib.train_index.restype = ctypes.c_int
        
        self.lib.build_index.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
        self.lib.build_index.restype = ctypes.c_int

        # 4. Model Load & Engine Init
        self.encoder = SentenceTransformer(model_name)
        self.dims = self.encoder.get_sentence_embedding_dimension()

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
        
        # Save embeddings with dynamic prefix
        embeddings.tofile(f"{self.db_prefix}_offline_embeddings.bin")

        self.db = {str(i): text for i, text in enumerate(texts)}
        with open(self.map_path, "w") as f:
            json.dump(self.db, f)

        # Pass dynamic parameters through ctypes
        if self.lib.train_index(k, self.dims, self.c_prefix, max_iter) != 0: return False
        if self.lib.build_index(k, self.dims, self.c_prefix) != 0: return False
        
        self.lib.cleanup_engine()
        self.lib.init_engine(self.dims, self.c_prefix)
        self.refresh_map()
        return True

    def search(self, query_text):
        query_vector = self.encoder.encode(query_text).astype(np.float32)
        idx = self.lib.vector_search(query_vector, self.dims, self.c_prefix)
        
        if idx < 0:
            error_msgs = {
                -1: "Search Error: No valid cluster found.",
                -2: f"IO Error: Database file {self.db_prefix}_ivf_database.bin missing.",
                -3: "Bounds Error: Resulting index out of range.",
                -404: "Engine Error: Core not initialized."
            }
            context = error_msgs.get(idx, "Unknown Core Error.")
            return idx, context

        context = self.db.get(str(idx))
        if context is None:
            return idx, f"Mapping Error: Index {idx} not found in JSON database."
            
        return idx, context

    def query_llm_stream(self, model, query, context, api_url="http://localhost:11434/api/generate", **kwargs):
        prompt = f"Context: {context}\n\nQuery: {query}\n\nAnswer concisely:"
        
        # Merge default options with any user-supplied kwargs
        options = {"temperature": 0.2}
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