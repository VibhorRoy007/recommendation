"""Build a FAISS index for creator embeddings.

- Loads normalized embeddings from ``embeddings/creator_embeddings.npy``.
- Uses ``faiss.IndexFlatIP`` (inner product) because embeddings are L2‑normalized, so IP == cosine similarity.
- Saves the index to ``indexes/creator_index.faiss``.
"""

import os
from pathlib import Path
import numpy as np
import faiss

# Paths (relative to repository root)
BASE_DIR = Path(__file__).resolve().parents[2]
EMB_DIR = BASE_DIR / "embeddings"
IDX_DIR = BASE_DIR / "indexes"

EMB_PATH = EMB_DIR / "creator_embeddings.npy"
IDX_PATH = IDX_DIR / "creator_index.faiss"

def main():
    # Ensure output directory exists
    os.makedirs(IDX_DIR, exist_ok=True)

    # Load embeddings (already normalized)
    embeddings = np.load(EMB_PATH)
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings should be a 2‑D array, got shape {embeddings.shape}")

    dim = embeddings.shape[1]
    # IndexFlatIP works with inner product (cosine similarity for normalized vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)  # type: ignore[arg-type]

    # Persist the index to disk
    faiss.write_index(index, str(IDX_PATH))
    print(f"FAISS index with {index.ntotal} vectors saved to {IDX_PATH}")

if __name__ == "__main__":
    main()
