# Creator Recommendation Engine – Methodology Memo

## Problem & Approach
We need to recommend **10** creators who are semantically similar to a given multi‑hyphenate creator.  The solution is an **embedding‑based retrieval system** that works on public Instagram bios.

## Data Pipeline
1. **Source** – Instagram public profiles scraped with Apify (hashtag discovery + follower traversal).  All JSON outputs are merged by `collect_usernames.py`.
2. **Schema** – `username`, `biography`, `followersCount`.
3. **Cleaning (`preprocess.py`)** –
   - Remove duplicate usernames and duplicate bios.
   - Drop rows with empty bios.
   - Strip whitespace and collapse multiple spaces.
   - Light emoji‑noise removal (runs > 3 emojis are stripped).
   - Output: `processed_creators.csv` (~584 creators; target is ≥ 1 000).

## Featurization
We use **Sentence‑Transformers** `all‑mpnet‑base‑v2` (384‑dim).  This model was chosen because:
- Captures semantic meaning beyond keyword overlap.
- Handles varied creator terminology (e.g., “stylist‑photographer‑vintage curator”).
- No need for a manually built vocabulary.
The embeddings are **L2‑normalized** so that inner product equals cosine similarity.

## Similarity & Ranking
- **FAISS** `IndexFlatIP` stores the normalized embeddings.
- Query embedding is computed on‑the‑fly and a **top‑10 nearest‑neighbor** search is performed.
- FAISS provides exact inner‑product search and scales far better than a brute‑force `sklearn.metrics.pairwise.cosine_similarity` loop.

## Query Interface (`app.py`)
- **Mode 1 – Instagram handle** – Live profile fetch via the Apify API (`live_profile_fetch.py`) supplies the biography for embedding.
- **Mode 2 – Raw bio** – Direct user‑provided text is embedded.
- Results are displayed in Streamlit with similarity % scores, category tags, and a short *why‑similar* explanation.

## Trade‑offs & Biases
- **Sample bias** – Current dataset is Mumbai‑centric; creators from other regions are under‑represented.
- **Language bias** – English‑dominant bios; non‑English descriptions may be poorly represented.
- **Size limitation** – Only ~584 creators; retrieval quality improves with a larger corpus.
- **Model choice** – `all‑mpnet‑base‑v2` is a general‑purpose encoder.  A domain‑specific fine‑tuned model could yield higher precision.

## Evaluation (planned)
- Hand‑label ≥ 30 creator pairs as *similar* / *not similar*.
- For each query, check if the labelled similar creators appear in the top‑10 results.
- Report **Hit‑Rate@10** and **Precision@10**.

## Next Steps
1. **Scale dataset** to > 1 000 creators (additional scrapes, other platforms).
2. **Add evaluation script** (`evaluate.py`) and populate `evaluation_pairs.csv`.
3. **Optional clustering** – run K‑means (scikit‑learn) on embeddings to surface creator archetypes.
4. **LLM‑generated explanations** for the *why‑similar* text.
5. **Documentation** – finalized README and deployment instructions.
