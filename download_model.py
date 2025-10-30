import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")  # avoid missing hf_transfer fast path
from sentence_transformers import SentenceTransformer

cache = os.environ.get("TRANSFORMERS_CACHE", "/opt/render/project/.cache")
model_name = "all-MiniLM-L6-v2"

print(f"[build] prefetching {model_name} into {cache} ...", flush=True)
SentenceTransformer(model_name, cache_folder=cache)
print("[build] done.", flush=True)
