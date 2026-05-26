'''Evaluation script for the Creator Recommendation Engine.

Usage:
    python evaluate.py [--k K] [--pairs FILE]

* Loads the FAISS index, embeddings and metadata.
* Reads a CSV of labelled creator pairs (columns: creator_a, creator_b, label).
* For each pair it runs a top‑K nearest‑neighbor search for ``creator_a``.
* Computes:
    - **Hit@K** for pairs labelled ``similar`` (does ``creator_b`` appear in the top‑K?).
    - **Precision@K** – average proportion of ``similar`` items among the top‑K results.
    - **False‑Positive rate** – proportion of ``not_similar`` pairs that incorrectly appear in the top‑K.
* Prints a concise summary and writes ``evaluation_report.csv`` with per‑pair details.

The script is deliberately lightweight and re‑uses the same loading logic as
``test_similarity.py`` so the results are directly comparable to the live UI.
'''

import argparse
import pandas as pd
import numpy as np
import faiss
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration – same layout as the rest of the repo
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
EMB_DIR = BASE_DIR / "embeddings"
IDX_DIR = BASE_DIR / "indexes"

EMB_PATH = EMB_DIR / "creator_embeddings.npy"
META_PATH = EMB_DIR / "creator_metadata.csv"
IDX_PATH = IDX_DIR / "creator_index.faiss"

DEFAULT_K = 10

# ---------------------------------------------------------------------------
# Helper functions (mirroring test_similarity.py)
# ---------------------------------------------------------------------------

def load_resources():
    """Load embeddings, metadata and FAISS index."""
    embeddings = np.load(EMB_PATH)
    metadata = pd.read_csv(META_PATH)
    index = faiss.read_index(str(IDX_PATH))
    return embeddings, metadata, index


def find_username_index(metadata: pd.DataFrame, username: str) -> int:
    """Return the row index for a given ``username`` (exact match)."""
    matches = metadata.index[metadata["username"] == username].tolist()
    if not matches:
        raise ValueError(f"Username '{username}' not found in metadata")
    return int(matches[0])


def top_k_for_user(username: str, k: int, embeddings: np.ndarray, metadata: pd.DataFrame, index) -> list:
    """Return the top‑k usernames (excluding the query) for ``username``."""
    query_idx = find_username_index(metadata, username)
    query_vec = embeddings[query_idx].reshape(1, -1)
    # Search for k+1 because the query itself is the nearest neighbor
    distances, indices = index.search(query_vec, k + 1)
    distances = distances.ravel()
    indices = indices.ravel()
    results = []
    for dist, idx in zip(distances, indices):
        if idx == query_idx:
            continue
        results.append((metadata.iloc[idx]["username"], float(dist)))
        if len(results) >= k:
            break
    return results

# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def evaluate(pairs_path: Path, k: int = DEFAULT_K):
    embeddings, metadata, index = load_resources()
    pairs = pd.read_csv(pairs_path)
    # Normalise label values
    pairs["label"] = pairs["label"].str.lower().str.strip()

    # Containers for metrics
    hits = 0          # similar pairs where B is in top‑k
    total_sim = 0
    false_positives = 0
    total_not_sim = 0
    precision_acc = 0.0  # sum of precision per query (for similar items)

    # Per‑pair report rows
    report_rows = []

    # Group by creator_a so we can compute precision per query efficiently
    for _, row in pairs.iterrows():
        a = row["creator_a"].strip()
        b = row["creator_b"].strip()
        label = row["label"]
        try:
            top_k = top_k_for_user(a, k, embeddings, metadata, index)
            top_usernames = [u for u, _ in top_k]
        except ValueError as e:
            # If ``a`` is missing we skip the pair but note it in the report.
            report_rows.append({
                "creator_a": a,
                "creator_b": b,
                "label": label,
                "error": str(e),
                "hit": None,
                "rank": None,
                "precision": None,
            })
            continue

        if label == "similar":
            total_sim += 1
            if b in top_usernames:
                hits += 1
                rank = top_usernames.index(b) + 1
            else:
                rank = None
            # Precision for this query = (# of similar items in top‑k) / k
            # Since the CSV only provides one ``b`` per ``a`` we treat it as 1 if hit else 0.
            precision = 1.0 / k if b in top_usernames else 0.0
            precision_acc += precision
            report_rows.append({
                "creator_a": a,
                "creator_b": b,
                "label": label,
                "error": "",
                "hit": b in top_usernames,
                "rank": rank,
                "precision": precision,
            })
        elif label == "not_similar":
            total_not_sim += 1
            fp = b in top_usernames
            false_positives += int(fp)
            report_rows.append({
                "creator_a": a,
                "creator_b": b,
                "label": label,
                "error": "",
                "hit": not fp,
                "rank": None,
                "precision": None,
            })
        else:
            # Unknown label – skip but record.
            report_rows.append({
                "creator_a": a,
                "creator_b": b,
                "label": label,
                "error": "unknown label",
                "hit": None,
                "rank": None,
                "precision": None,
            })

    # Compute final metrics
    hit_at_k = hits / total_sim if total_sim else 0.0
    avg_precision = precision_acc / total_sim if total_sim else 0.0
    false_positive_rate = false_positives / total_not_sim if total_not_sim else 0.0

    # Summary output
    print("--- Evaluation Summary ---")
    print(f"Pairs processed          : {len(pairs)}")
    print(f"Similar pairs (label)    : {total_sim}")
    print(f"Not‑similar pairs (label): {total_not_sim}")
    print(f"Top‑{k} Hit Rate (similar)      : {hit_at_k:.3%}")
    print(f"Average Precision@{k} (similar): {avg_precision:.3%}")
    print(f"False‑Positive Rate (not similar): {false_positive_rate:.3%}")

    # Write detailed CSV for further analysis
    report_df = pd.DataFrame(report_rows)
    out_path = BASE_DIR / "evaluation_report.csv"
    report_df.to_csv(out_path, index=False)
    print(f"Detailed report written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run creator‑similarity evaluation.")
    parser.add_argument("--pairs", type=str, default=str(BASE_DIR / "evaluation_pairs.csv"),
                        help="Path to CSV containing labelled creator pairs.")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help="Number of nearest neighbours to consider (default 10).")
    args = parser.parse_args()
    evaluate(Path(args.pairs), k=args.k)
