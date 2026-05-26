"""Generate high‑quality semantic embeddings for creator biographies.

- Uses the pretrained model `sentence-transformers/all-mpnet-base-v2`.
- Reads the pre‑processed CSV produced by `preprocess.py`.
- Embeds the `biography` column in batches (default batch size 32).
- Normalizes embeddings to unit length (so inner‑product = cosine similarity).
- Saves the resulting embeddings as a NumPy ``.npy`` file.
- Stores accompanying metadata (username, biography, followersCount) as CSV.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

# Configuration
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
EMB_DIR = Path(__file__).resolve().parents[2] / "embeddings"
INPUT_CSV = DATA_DIR / "processed_creators.csv"
EMB_OUTPUT = EMB_DIR / "creator_embeddings.npy"
META_OUTPUT = EMB_DIR / "creator_metadata.csv"
BATCH_SIZE = 32
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def load_data(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def embed_biographies(texts: list, model: SentenceTransformer, batch_size: int = 32) -> np.ndarray:
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
        batch = texts[i : i + batch_size]
        # Encode returns normalized embeddings if enable_normalized_embedding=True, but we will normalize manually for clarity
        batch_emb = model.encode(batch, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
        embeddings.append(batch_emb)
    return np.vstack(embeddings)

def main():
    os.makedirs(EMB_DIR, exist_ok=True)
    df = load_data(INPUT_CSV)
    biographies = df["biography"].astype(str).tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)

    # Generate embeddings
    raw_emb = embed_biographies(biographies, model, batch_size=BATCH_SIZE)

    # Normalize to unit length (L2 norm)
    norms = np.linalg.norm(raw_emb, axis=1, keepdims=True)
    normalized_emb = raw_emb / np.maximum(norms, 1e-10)

    # Save embeddings and metadata
    np.save(EMB_OUTPUT, normalized_emb)
    df.to_csv(META_OUTPUT, index=False)
    print(f"Saved {normalized_emb.shape[0]} embeddings to {EMB_OUTPUT}")
    print(f"Saved metadata to {META_OUTPUT}")

if __name__ == "__main__":
    main()
