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
    def __init__(self, map_path="dataset_map.json", model_name='all-MiniLM-L6-v2'):
        base_path = os.path.dirname(__file__)
        ext = ".dll" if sys.platform == "win32" else ".so"
        self.lib_path = os.path.join(base_path, f"liblocalvec{ext}")
        
        if not os.path.exists(self.lib_path):
            raise FileNotFoundError(f"LocVec Core not found at {self.lib_path}")

        self.lib = ctypes.CDLL(self.lib_path)

        self.lib.init_engine.restype = ctypes.c_int
        self.lib.cleanup_engine.restype = None
        self.lib.vector_search.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, ndim=1), ctypes.c_int]
        self.lib.vector_search.restype = ctypes.c_int
        self.lib.train_index.restype = ctypes.c_int
        self.lib.build_index.restype = ctypes.c_int

        self.lib.init_engine()

        self.encoder = SentenceTransformer(model_name)
        self.map_path = map_path
        self.db = {}
        self.refresh_map()

    def refresh_map(self):
        if os.path.exists(self.map_path):
            with open(self.map_path, "r") as f:
                self.db = json.load(f)

    def build_full_index(self, texts):
        embeddings = self.encoder.encode(texts, show_progress_bar=True).astype(np.float32)
        embeddings.tofile("offline_embeddings.bin")

        self.db = {str(i): text for i, text in enumerate(texts)}
        with open(self.map_path, "w") as f:
            json.dump(self.db, f)

        if self.lib.train_index() != 0: return False
        if self.lib.build_index() != 0: return False
        
        self.lib.cleanup_engine()
        self.lib.init_engine()
        self.refresh_map()
        return True

    def search(self, query_text):
        query_vector = self.encoder.encode(query_text).astype(np.float32)
        idx = self.lib.vector_search(query_vector, len(query_vector))
        
        if idx < 0:
            error_msgs = {
                -1: "Search Error: No valid cluster found.",
                -2: "IO Error: Database file missing.",
                -3: "Bounds Error: Resulting index out of range.",
                -404: "Engine Error: Core not initialized."
            }
            context = error_msgs.get(idx, "Unknown Core Error.")
            return idx, context

        context = self.db.get(str(idx))
        if context is None:
            return idx, f"Mapping Error: Index {idx} not found in JSON database."
            
        return idx, context

    def query_llm_stream(self, model, query, context):
        prompt = f"Context: {context}\n\nQuery: {query}\n\nAnswer concisely:"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_thread": 8, "temperature": 0.2}
        }
        
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
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
            yield f"\n[Error]: {str(e)}"

    def offload_encoder(self):
        self.encoder.to('cpu')
        torch.cuda.empty_cache()
        gc.collect()

    def __del__(self):
        try:
            self.lib.cleanup_engine()
        except:
            pass
