# Creator Recommendation Engine (Option 1)

## Overview
A semantic‑search system that, given a multi‑hyphenate creator (Instagram handle or free‑form bio), returns the **top 10** most similar creators from a curated dataset.  It powers the *"Creators like you"* feature for **bb.bio**.

## Repository layout
```
bb/
├─ data/                 # Processed CSV/Parquet of creators
├─ embeddings/           # Normalized embeddings (.npy) and metadata CSV
├─ indexes/              # FAISS index file
├─ project/
│   ├─ app.py            # Streamlit UI
│   ├─ requirements.txt  # Python dependencies
│   └─ src/              # Core pipeline scripts
│       ├─ collect_usernames.py
│       ├─ preprocess.py
│       ├─ generate_embeddings.py
│       ├─ build_faiss_index.py
│       ├─ live_profile_fetch.py
│       ├─ test_similarity.py
│       └─ test_live_query.py
├─ METHODOLOGY_MEMO.md  # One‑page design & evaluation memo
└─ apify_api_token.txt   # Apify token (keep secret – .gitignore excludes it)
```

## Prerequisites
- **Python 3.10+** (tested on 3.12)
- **Git** (optional, for version control)
- An **Apify API token** with access to the `apify/instagram-profile-scraper` actor.  Save it as `apify_api_token.txt` in the repository root (already present).  **Never commit this file.**

## Setup instructions
1. **Clone the repo** (if you haven't already) and `cd` into the project root.
   ```bash
   git clone <repo‑url>
   cd bb
   ```
2. **Create a virtual environment** (recommended) and install dependencies.
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate
   pip install -r project/requirements.txt
   ```
3. **Data preparation** – run the ingestion pipeline.
   ```bash
   # 1️⃣ Combine scraped JSON files and extract usernames
   python collect_usernames.py
   
   # 2️⃣ Clean and preprocess bios
   python project/src/preprocess.py
   
   # 3️⃣ Generate embeddings (uses a GPU if available)
   python project/src/generate_embeddings.py
   
   # 4️⃣ Build the FAISS index
   python project/src/build_faiss_index.py
   ```
   After step 4 you should see:
   - `data/processed_creators.csv` (≈ 600 rows, aim for ≥ 1 000)
   - `embeddings/creator_embeddings.npy`
   - `embeddings/creator_metadata.csv`
   - `indexes/creator_index.faiss`

4. **Run the Streamlit app**
   ```bash
   streamlit run project/app.py
   ```
   The UI opens in your browser.  Choose **Instagram handle** or **Bio input**, hit **Discover Similar Creators →**, and view the ranked results.

5. **Local similarity testing (optional)**
   ```bash
   python project/src/test_similarity.py <instagram_username>
   ```
   Shows the top‑10 similar creators for a known handle.

## Evaluation (planned)
1. Fill `evaluation_pairs.csv` with at least **30** manually labelled pairs (`creator_a,creator_b,label`).
2. Run the evaluation script (to be added):
   ```bash
   python evaluate.py
   ```
   It will report **Hit‑Rate@10** and **Precision@10**.

## Extending / Customising
- **Alternative similarity** – replace FAISS with scikit‑learn cosine similarity if you prefer a pure‑Python fallback.
- **Clustering** – run K‑means on the embeddings (see `sklearn_fallback.py`) to obtain creator archetypes.
- **More data** – add additional Instagram scrapes or other platforms (LinkedIn, YouTube) and re‑run the pipeline.

## License & Disclaimer
- This code is for internal prototyping.  The dataset contains public Instagram bios; ensure compliance with Instagram’s TOS and privacy policies before commercial use.
- The Apify token must be kept secret – add `apify_api_token.txt` to `.gitignore`.

---

