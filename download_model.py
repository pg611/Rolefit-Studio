import os
from sentence_transformers import SentenceTransformer

cache = os.environ.get("TRANSFORMERS_CACHE", "/opt/render/project/.cache")
model_name = "all-MiniLM-L6-v2"

print(f"[build] prefetching {model_name} into {cache} ...", flush=True)
SentenceTransformer(model_name, cache_folder=cache)
print("[build] done.", flush=True)
