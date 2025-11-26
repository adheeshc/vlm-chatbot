import hashlib
from pathlib import Path

import torch


class VisionEmbeddingCache:
    def __init__(self, cache_dir="./cache/vision_embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}

    def _get_cache_key(self, image_path):
        mtime = Path(image_path).stat().st_mtime
        key = f"{image_path}_{mtime}"
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, image_path):
        cache_key = self._get_cache_key(image_path)

        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]

        cache_file = self.cache_dir / f"{cache_key}.pt"
        if cache_file.exists():
            embedding = torch.load(cache_file)
            self.memory_cache[cache_key] = embedding
            return embedding

        return None

    def set(self, image_path, embedding):
        cache_key = self._get_cache_key(image_path)
        self.memory_cache[cache_key] = embedding
        cache_file = self.cache_dir / f"{cache_key}.pt"
        torch.save(embedding.cpu(), cache_file)
