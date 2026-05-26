"""Live Instagram handle similarity lookup.

Workflow:
1. Fetch a single Instagram profile using the Apify actor (live_profile_fetch.fetch_profile).
2. Clean the biography with the same routine used in preprocess.py.
3. Generate a normalized embedding with ``sentence‑transformers/all‑mpnet‑base‑v2``.
4. Load the existing FAISS index (inner‑product, embeddings are L2‑normalised).
5. Retrieve the top‑k nearest neighbours (default 10) – the query handle itself is excluded.
6. Print ``username``, ``similarity score`` and a short bio preview for each neighbour.
7. (Optional) With the ``--save`` flag the new creator is appended to ``embeddings/creator_metadata.csv`` and ``embeddings/creator_embeddings.npy`` **without** rebuilding the index.

Usage::

    python test_live_query.py <instagram_handle> [--save]

Example::

    python test_live_query.py newcreator123
    python test_live_query.py newcreator123 --save
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

# Local imports
from live_profile_fetch import fetch_profile
import preprocess  # contains clean_bio function

# -------------------------------------------------
# Configuration
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
EMB_DIR = BASE_DIR / "embeddings"
IDX_DIR = BASE_DIR / "indexes"

EMB_PATH = EMB_DIR / "creator_embeddings.npy"
META_PATH = EMB_DIR / "creator_metadata.csv"
IDX_PATH = IDX_DIR / "creator_index.faiss"

TOP_K = 10
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# -------------------------------------------------

def load_resources():
    # Load FAISS index, embeddings and metadata
    index = faiss.read_index(str(IDX_PATH))
    embeddings = np.load(EMB_PATH)
    metadata = pd.read_csv(META_PATH)
    return index, embeddings, metadata

def generate_embedding(text: str, model):
    # Apply the same cleaning as the batch pipeline
    cleaned = preprocess.clean_bio(text)
    emb = model.encode([cleaned], convert_to_numpy=True, normalize_embeddings=False)
    # Normalise to unit length (so inner product = cosine)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norm, 1e-10)

def query_index(query_vec: np.ndarray, index, k: int = TOP_K + 1):
    # FAISS expects a 2‑D array
    distances, indices = index.search(query_vec, k)
    return distances.ravel(), indices.ravel()

def print_results(query_username: str, distances, indices, metadata):
    # Remove the query itself if present
    results = []
    for dist, idx in zip(distances, indices):
        if idx >= len(metadata):
            continue
        uname = metadata.iloc[idx]["username"]
        if uname == query_username:
            continue
        results.append((uname, float(dist), metadata.iloc[idx]["biography"]))
        if len(results) >= TOP_K:
            break
    print(f"Top {TOP_K} similar creators for '@{query_username}':")
    for rank, (uname, score, bio) in enumerate(results, start=1):
        preview = (bio[:120] + "…") if len(bio) > 120 else bio
        print(f"{rank}. @{uname} — {score:.4f}\n   {preview}\n")
    return results

def append_to_database(new_profile: dict, embedding: np.ndarray, metadata_df: pd.DataFrame, embeddings_arr: np.ndarray):
    # Append metadata row
    new_row = {
        "username": new_profile.get("username"),
        "biography": new_profile.get("biography", ""),
        "followersCount": new_profile.get("followersCount", 0),
    }
    metadata_df = pd.concat([metadata_df, pd.DataFrame([new_row])], ignore_index=True)
    # Append embedding
    embeddings_arr = np.vstack([embeddings_arr, embedding])
    # Save back to disk
    metadata_df.to_csv(META_PATH, index=False)
    np.save(EMB_PATH, embeddings_arr)
    print(f"Appended '{new_profile.get('username')}' to local database (metadata + embeddings). Index not rebuilt.")
    return metadata_df, embeddings_arr

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_live_query.py <instagram_handle> [--save]")
        sys.exit(1)
    handle = sys.argv[1].lstrip("@")
    save_flag = "--save" in sys.argv

    # 1️⃣ Fetch live profile
    try:
        profile = fetch_profile(handle)
    except Exception as e:
        print(f"Failed to fetch profile for '{handle}': {e}")
        sys.exit(1)

    # 2️⃣ Load model (once)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)

    # 3️⃣ Generate embedding for the fetched bio
    embedding = generate_embedding(profile.get("biography", ""), model)

    # 4️⃣ Load index & metadata
    index, embeddings, metadata = load_resources()

    # 5️⃣ Query
    dists, idxs = query_index(embedding, index)
    results = print_results(profile["username"], dists, idxs, metadata)

    # 6️⃣ Optional growth
    if save_flag:
        # Ensure we do not duplicate an existing username
        if handle in set(metadata["username"]):
            print(f"Username '{handle}' already exists in the local database – not appending.")
        else:
            metadata, embeddings = append_to_database(profile, embedding, metadata, embeddings)

if __name__ == "__main__":
    main()
