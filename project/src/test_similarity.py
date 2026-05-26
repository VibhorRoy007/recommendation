"""Local semantic similarity testing for creator biographies.

Usage:
    python test_similarity.py <username>

The script loads the FAISS index (inner‑product) built from normalized embeddings
and the creator metadata CSV. It retrieves the embedding for the supplied
username, performs a nearest‑neighbor search, and prints the top 10 most
similar creators (excluding the query itself).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import faiss

# ------------------------ Configuration ------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
EMB_DIR = BASE_DIR / "embeddings"
IDX_DIR = BASE_DIR / "indexes"

EMB_PATH = EMB_DIR / "creator_embeddings.npy"
META_PATH = EMB_DIR / "creator_metadata.csv"
IDX_PATH = IDX_DIR / "creator_index.faiss"
TOP_K = 10  # number of results to show (excluding the query)
# -------------------------------------------------------------

def load_resources():
    embeddings = np.load(EMB_PATH)
    metadata = pd.read_csv(META_PATH)
    # Ensure the order matches the embeddings (they were saved together)
    # The metadata CSV is written in the same order as the embeddings.
    index = faiss.read_index(str(IDX_PATH))
    return embeddings, metadata, index

def find_username_index(metadata: pd.DataFrame, username: str) -> int:
    matches = metadata.index[metadata["username"] == username].tolist()
    if not matches:
        raise ValueError(f"Username '{username}' not found in metadata")
    return int(matches[0])

def query_similar(username: str, embeddings: np.ndarray, metadata: pd.DataFrame, index: faiss.IndexFlatIP):
    query_idx = find_username_index(metadata, username)
    query_vec = embeddings[query_idx].reshape(1, -1)
    # Search for TOP_K + 1 because the query itself will appear as the nearest neighbor
    distances, indices = index.search(query_vec, TOP_K + 1)
    distances = distances.ravel()
    indices = indices.ravel()
    # Remove the query itself (if present)
    results = []
    for dist, idx in zip(distances, indices):
        if idx == query_idx:
            continue
        results.append((metadata.iloc[idx]["username"], float(dist), metadata.iloc[idx]["biography"]))
        if len(results) >= TOP_K:
            break
    return results

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_similarity.py <username>")
        sys.exit(1)
    username = sys.argv[1]
    embeddings, metadata, index = load_resources()
    try:
        results = query_similar(username, embeddings, metadata, index)
    except ValueError as e:
        print(e)
        sys.exit(1)
    print(f"Top {TOP_K} similar creators for '@{username}':")
    for rank, (usr, score, bio) in enumerate(results, start=1):
        preview = (bio[:100] + "…") if len(bio) > 100 else bio
        print(f"{rank}. @{usr} - {score:.4f}\n   {preview.encode('ascii', errors='ignore').decode('ascii')}\n")

if __name__ == "__main__":
    main()
