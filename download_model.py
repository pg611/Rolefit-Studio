import os

# Disable HF fast-transfer so we don't need the hf_transfer package on Render
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from sentence_transformers import SentenceTransformer

cache = os.environ.get("TRANSFORMERS_CACHE", "/opt/render/project/.cache")
model_name = "all-MiniLM-L6-v2"

print(f"[build] prefetching {model_name} into {cache} ...", flush=True)
SentenceTransformer(model_name, cache_folder=cache)
print("[build] done.", flush=True)

