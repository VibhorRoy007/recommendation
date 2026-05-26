import sys
import os
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------
# Make the `src` folder importable
# ----------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.append(str(SRC_DIR))

from live_profile_fetch import fetch_profile
import preprocess

# ----------------------------------------------------------------------
# Helper: profile validation
# ----------------------------------------------------------------------
def validate_creator_profile(profile: dict) -> tuple[bool, str]:
    """Return (is_valid, cleaned_bio).
    A profile is valid when the ``biography`` field exists, is a string,
    and contains non‑empty text after stripping whitespace.
    """
    bio = profile.get("biography", "") or ""
    if not isinstance(bio, str):
        bio = str(bio)
    cleaned = bio.strip()
    return (len(cleaned) > 0), cleaned

# ----------------------------------------------------------------------
# Configuration / paths
# ----------------------------------------------------------------------
BASE_DIR  = Path(__file__).resolve().parent
EMB_DIR   = BASE_DIR.parent / "embeddings"
IDX_DIR   = BASE_DIR.parent / "indexes"

EMB_PATH  = EMB_DIR / "creator_embeddings.npy"
META_PATH = EMB_DIR / "creator_metadata.csv"
IDX_PATH  = IDX_DIR / "creator_index.faiss"

TOP_K      = 10
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# ----------------------------------------------------------------------
# Category / niche detection
# ----------------------------------------------------------------------
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "📸 Photography":      ["photo", "photographer", "photography", "lens", "shoot", "shot",
                             "canon", "nikon", "lightroom", "portrait", "camera"],
    "👗 Fashion":          ["fashion", "style", "stylist", "outfit", "ootd", "clothing",
                             "wear", "model", "modeling", "designer", "couture", "streetwear"],
    "🎵 Music":            ["music", "musician", "singer", "rapper", "producer", "dj",
                             "artist", "sound", "beats", "album", "track", "band", "guitar", "vocals"],
    "💼 Entrepreneurship": ["entrepreneur", "founder", "ceo", "startup", "business", "brand",
                             "venture", "investor", "growth", "scale", "hustle"],
    "🎨 Art & Design":     ["artist", "design", "designer", "creative", "illustrator", "art",
                             "draw", "paint", "visual", "graphic", "ui", "ux"],
    "🏋️ Fitness":          ["fitness", "gym", "workout", "health", "nutrition", "athlete",
                             "sport", "training", "coach", "wellness", "yoga", "bodybuilding"],
    "✈️ Travel":           ["travel", "traveler", "explorer", "adventure", "wanderlust",
                             "nomad", "journey", "world", "globe", "destination"],
    "🍕 Food":             ["food", "chef", "cook", "recipe", "foodie", "restaurant",
                             "cuisine", "baking", "gastronomy", "culinary"],
    "💻 Tech":             ["tech", "developer", "engineer", "coding", "software", "ai",
                             "machine learning", "data", "cyber", "startup", "saas"],
    "📹 Content Creation": ["content", "creator", "youtube", "youtuber", "tiktok", "reels",
                             "video", "vlog", "blogger", "influencer", "podcast"],
    "🌿 Lifestyle":        ["lifestyle", "life", "daily", "routine", "mindset", "motivation",
                             "inspiration", "positive", "self", "personal"],
    "💄 Beauty":           ["beauty", "makeup", "skincare", "cosmetics", "glow", "mua",
                             "glam", "aesthetic", "tutorial"],
}

def detect_categories(bio: str) -> list[str]:
    bio_lower = bio.lower()
    found = [cat for cat, kws in CATEGORY_KEYWORDS.items() if any(kw in bio_lower for kw in kws)]
    return found if found else ["🌐 General"]

def why_similar(query_cats: list[str], result_cats: list[str]) -> str:
    shared = list(set(query_cats) & set(result_cats))
    if not shared:
        return "Similar overall creative positioning"
    labels = [c.split(" ", 1)[-1] for c in shared]
    return f"Shared focus: {', '.join(labels[:-1])} & {labels[-1]}" if len(labels) > 1 \
           else f"Both operate in the {labels[0]} space"

# ----------------------------------------------------------------------
# Backend utilities (cached)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource(show_spinner=False)
def load_resources():
    index    = faiss.read_index(str(IDX_PATH))
    metadata = pd.read_csv(META_PATH)
    return index, metadata

def generate_embedding(text: str, model: SentenceTransformer) -> np.ndarray:
    """Generate a normalized sentence‑transformer embedding for a cleaned bio.
    The cleaning step (emoji removal, whitespace collapse) lives in ``preprocess.clean_bio``.
    """
    cleaned = preprocess.clean_bio(text)
    vec     = model.encode([cleaned], convert_to_numpy=True, normalize_embeddings=False)
    norm    = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / np.maximum(norm, 1e-10)

def query_faiss(query_vec: np.ndarray, index, k: int = TOP_K + 1):
    distances, indices = index.search(query_vec, k)
    return distances.ravel(), indices.ravel()

def format_score(score: float) -> str:
    return f"{min(int(score * 100), 100)}%"

def score_to_int(score: float) -> int:
    return min(int(score * 100), 100)

# ----------------------------------------------------------------------
# Custom CSS  —  dark editorial aesthetic
# ----------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');

:root {
    --bg:       #0a0a0f;
    --surface:  #111118;
    --surface2: #18181f;
    --border:   #2a2a38;
    --accent:   #7c6af7;
    --accent2:  #f066a8;
    --accent3:  #40e0b0;
    --text:     #e8e8f0;
    --muted:    #7070a0;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stSidebar"], #MainMenu, footer { display:none !important; visibility:hidden !important; }

.block-container { max-width:820px !important; padding:2.5rem 1.5rem 5rem !important; }

/* Hero */
.hero { text-align:center; padding:2.5rem 0 1.5rem; }
.hero-eyebrow {
    font-family:'DM Mono',monospace; font-size:.72rem; letter-spacing:.25em;
    text-transform:uppercase; color:var(--accent); margin-bottom:.75rem;
}
.hero-title {
    font-family:'Syne',sans-serif; font-size:clamp(2rem,5vw,3.2rem);
    font-weight:800; line-height:1.08;
    background:linear-gradient(135deg,#e8e8f0 0%,var(--accent) 60%,var(--accent2) 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; margin-bottom:.6rem;
}
.hero-sub { font-size:.95rem; color:var(--muted); font-weight:300; max-width:460px; margin:0 auto; line-height:1.6; }

/* Radio tabs */
.stRadio > label { display:none !important; }
div[role="radiogroup"] {
    display:flex !important; gap:.5rem !important;
    background:var(--surface) !important; border:1px solid var(--border) !important;
    border-radius:12px !important; padding:5px !important; margin-bottom:1.5rem !important;
}
div[role="radiogroup"] label {
    flex:1 !important; text-align:center !important; border-radius:8px !important;
    padding:.55rem 1rem !important; cursor:pointer !important;
    font-family:'DM Sans',sans-serif !important; font-size:.88rem !important;
    font-weight:500 !important; color:var(--muted) !important;
    transition:all .2s ease !important; border:none !important; background:transparent !important;
}
div[role="radiogroup"] label:has(input:checked) { background:var(--accent) !important; color:#fff !important; }

/* Inputs */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background:var(--surface) !important; border:1px solid var(--border) !important;
    border-radius:10px !important; color:var(--text) !important;
    font-family:'DM Sans',sans-serif !important; font-size:.95rem !important;
    padding:.75rem 1rem !important; transition:border-color .2s ease !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color:var(--accent) !important; box-shadow:0 0 0 3px rgba(124,106,247,.15) !important;
}
.stTextInput label, .stTextArea label {
    color:var(--muted) !important; font-size:.78rem !important;
    font-family:'DM Mono',monospace !important; letter-spacing:.08em !important;
    text-transform:uppercase !important;
}

/* Button */
.stButton > button {
    width:100% !important;
    background:linear-gradient(135deg,var(--accent) 0%,var(--accent2) 100%) !important;
    color:#fff !important; border:none !important; border-radius:10px !important;
    font-family:'Syne',sans-serif !important; font-weight:700 !important;
    font-size:.95rem !important; letter-spacing:.04em !important;
    padding:.75rem 1.5rem !important; transition:opacity .2s ease,transform .15s ease !important;
}
.stButton > button:hover { opacity:.88 !important; transform:translateY(-1px) !important; }

/* Query card */
.query-card {
    background:var(--surface); border:1px solid var(--border);
    border-left:3px solid var(--accent); border-radius:14px;
    padding:1.4rem 1.6rem; margin:1.8rem 0 1rem;
}
.query-label {
    font-family:'DM Mono',monospace; font-size:.68rem; letter-spacing:.2em;
    text-transform:uppercase; color:var(--accent); margin-bottom:.4rem;
}
.query-username { font-family:'Syne',sans-serif; font-size:1.25rem; font-weight:700; color:var(--text); margin-bottom:.5rem; }
.query-bio { font-size:.9rem; color:var(--muted); line-height:1.6; margin-bottom:.6rem; }

/* Tags */
.tag-row { display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.5rem; }
.tag {
    background:var(--surface2); border:1px solid var(--border); border-radius:20px;
    padding:.22rem .75rem; font-size:.75rem; font-family:'DM Sans',sans-serif;
    color:var(--text); white-space:nowrap;
}
.tag-accent { background:rgba(124,106,247,.12); border-color:rgba(124,106,247,.35); color:var(--accent); }

/* Section header */
.section-header {
    font-family:'Syne',sans-serif; font-size:.76rem; font-weight:700;
    letter-spacing:.18em; text-transform:uppercase; color:var(--muted);
    margin:1.8rem 0 .9rem; display:flex; align-items:center; gap:.6rem;
}
.section-header::after { content:''; flex:1; height:1px; background:var(--border); }

/* Result card */
.result-card {
    background:var(--surface); border:1px solid var(--border); border-radius:14px;
    padding:1.2rem 1.5rem; margin-bottom:.85rem;
    transition:border-color .2s ease,background .2s ease;
    position:relative; overflow:hidden;
}
.result-card:hover { border-color:rgba(124,106,247,.4); background:var(--surface2); }
.result-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,var(--accent),var(--accent2)); opacity:0; transition:opacity .2s ease;
}
.result-card:hover::before { opacity:1; }

.result-top { display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; margin-bottom:.5rem; }
.result-rank { font-family:'DM Mono',monospace; font-size:.7rem; color:var(--muted); min-width:22px; padding-top:2px; }
.result-username { font-family:'Syne',sans-serif; font-size:1.05rem; font-weight:700; color:var(--text); text-decoration:none; flex:1; }
.result-username:hover { color:var(--accent); }
.result-score-block { text-align:right; flex-shrink:0; }
.result-score-pct { font-family:'DM Mono',monospace; font-size:1.1rem; font-weight:500; color:var(--accent3); }
.result-score-label { font-size:.65rem; color:var(--muted); font-family:'DM Mono',monospace; letter-spacing:.08em; text-transform:uppercase; }

/* Sim bar */
.sim-bar-bg { background:var(--surface2); border-radius:4px; height:4px; margin:.45rem 0 .65rem; overflow:hidden; }
.sim-bar-fill { height:100%; border-radius:4px; background:linear-gradient(90deg,var(--accent),var(--accent3)); }

.result-bio { font-size:.85rem; color:var(--muted); line-height:1.55; margin-bottom:.7rem; }
.result-footer { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:.5rem; }
.why-similar { font-size:.75rem; color:var(--accent2); font-style:italic; font-family:'DM Sans',sans-serif; }
.ig-link {
    font-family:'DM Mono',monospace; font-size:.72rem; color:var(--accent); text-decoration:none;
    letter-spacing:.05em; border:1px solid rgba(124,106,247,.3); border-radius:6px;
    padding:.2rem .6rem; transition:background .2s;
}
.ig-link:hover { background:rgba(124,106,247,.12); }

/* Misc */
    .warning-card {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-left: 4px solid var(--accent);
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin: 1.5rem 0;
        color: var(--muted);
    }
    .warning-header {
        font-family: 'DM Sans',sans-serif;
        font-weight: 600;
        color: var(--accent);
        margin-bottom: 0.4rem;
    }
    .warning-body {
        font-size: .9rem;
        line-height: 1.4;
    }
.stAlert { border-radius:10px !important; }
.stSpinner > div { border-top-color:var(--accent) !important; }
hr { border-color:var(--border) !important; }
.empty-state { text-align:center; padding:3rem 1rem; color:var(--muted); font-size:.9rem; }
.empty-state-icon { font-size:2.5rem; margin-bottom:.6rem; }
</style>
"""

# ======================================================================
# Page setup
# ======================================================================
st.set_page_config(
    page_title="Creator Discovery · AI Engine",
    page_icon="🔭",
    layout="centered",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ======================================================================
# Hero
# ======================================================================
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">AI-Powered &nbsp;·&nbsp; Semantic Search</div>
  <div class="hero-title">Creator Discovery Engine</div>
  <div class="hero-sub">Find creators who think, create, and position like you — powered by semantic embeddings.</div>
</div>
""", unsafe_allow_html=True)

# ======================================================================
# Mode selector
# ======================================================================
mode = st.radio(
    "Discovery mode",
    ["🔎  Instagram Handle", "✍️  Bio Input"],
    horizontal=True,
    label_visibility="collapsed",
)
use_instagram = mode.startswith("🔎")

# ======================================================================
# Input fields
# ======================================================================
if use_instagram:
    username_input = st.text_input(
        "INSTAGRAM USERNAME",
        placeholder="e.g. paridafx",
        help="Enter without the @ symbol",
    )
    bio_input = ""
else:
    username_input = ""
    bio_input = st.text_area(
        "YOUR BIO / POSITIONING",
        placeholder="Fashion stylist | Photographer | Indie creative building my own lane...",
        height=110,
        help="Describe yourself as a creator. The more specific, the better the matches.",
    )

search_button = st.button("Discover Similar Creators →")

# ======================================================================
# Search logic
# ======================================================================
if search_button:

    # Validation
    if use_instagram and not username_input.strip():
        st.warning("⚠️ Please enter an Instagram username.")
        st.stop()
    if not use_instagram and not bio_input.strip():
        st.warning("⚠️ Please enter a bio or creative positioning statement.")
        st.stop()

    with st.spinner("Loading models and index…"):
        model = load_model()
        faiss_index, metadata_df = load_resources()

    # ── MODE 1: Instagram Handle ─────────────────────────────────────────
    if use_instagram:
        handle = username_input.strip().lstrip("@")
        with st.spinner(f"Fetching @{handle}'s profile…"):
            try:
                profile = fetch_profile(handle)
            except RuntimeError as e:
                st.error(f"❌ Could not fetch profile: {e}")
                st.stop()
            except Exception as e:
                st.error(f"❌ Unexpected error while fetching profile: {e}")
                st.stop()

        # Validate profile bio
        is_valid, cleaned_bio = validate_creator_profile(profile)
        if not is_valid:
            # Show warning UI and abort further processing
            st.markdown("""
            <div class='warning-card'>
              <div class='warning-header'>⚠️ No biography available</div>
              <div class='warning-body'>This Instagram profile has no bio available. Recommendations cannot be generated.<br>Try another creator or use the Bio Input mode.</div>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        query_bio = cleaned_bio
        display_handle = f"@{profile.get('username', handle)}"
        query_cats = detect_categories(query_bio)
        queried_uname = profile.get("username", handle)

        tags_html = "".join(f'<span class="tag tag-accent">{c}</span>' for c in query_cats)
        st.markdown(f"""
        <div class="query-card">
          <div class="query-label">Queried Profile</div>
          <div class="query-username">{display_handle}</div>
          <div class="query-bio">{query_bio or '<em>No bio available</em>'}</div>
          <div class="tag-row">{tags_html}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── MODE 2: Bio Input ────────────────────────────────────────────────
    else:
        query_bio      = bio_input.strip()
        display_handle = None
        queried_uname  = None
        query_cats     = detect_categories(query_bio)

        tags_html = "".join(f'<span class="tag tag-accent">{c}</span>' for c in query_cats)
        st.markdown(f"""
        <div class="query-card">
          <div class="query-label">Your Creative Positioning</div>
          <div class="query-bio">{query_bio}</div>
          <div class="tag-row">{tags_html}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Embed + search ───────────────────────────────────────────────────
    with st.spinner("Generating semantic embedding…"):
        query_vec = generate_embedding(query_bio, model)

    with st.spinner("Searching creator index…"):
        distances, indices = query_faiss(query_vec, faiss_index)

    # ── Build result list ────────────────────────────────────────────────
    results = []
    for dist, idx in zip(distances, indices):
        if idx >= len(metadata_df):
            continue
        row   = metadata_df.iloc[idx]
        uname = str(row["username"])
        if queried_uname and uname == queried_uname:
            continue
        bio_raw = str(row.get("biography", ""))
        r_cats  = detect_categories(bio_raw)
        results.append({
            "username":  uname,
            "score_str": format_score(dist),
            "score_int": score_to_int(dist),
            "bio":       bio_raw,
            "cats":      r_cats,
            "why":       why_similar(query_cats, r_cats),
        })
        if len(results) >= TOP_K:
            break

    # ── Render result cards ──────────────────────────────────────────────
    if not results:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-state-icon">🌑</div>
          No similar creators found in the index.<br>Try a different bio or username.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="section-header">Top {len(results)} Similar Creators</div>',
                    unsafe_allow_html=True)

        for i, res in enumerate(results, start=1):
            pct     = res["score_int"]
            tags    = "".join(f'<span class="tag">{c}</span>' for c in res["cats"])
            preview = (res["bio"][:160] + "…") if len(res["bio"]) > 160 else res["bio"]

            st.markdown(f"""
            <div class="result-card">
              <div class="result-top">
                <span class="result-rank">#{i:02d}</span>
                <a class="result-username"
                   href="https://instagram.com/{res['username']}"
                   target="_blank">@{res['username']}</a>
                <div class="result-score-block">
                  <div class="result-score-pct">{res['score_str']}</div>
                  <div class="result-score-label">match</div>
                </div>
              </div>
              <div class="sim-bar-bg">
                <div class="sim-bar-fill" style="width:{pct}%"></div>
              </div>
              <div class="result-bio">{preview or '<em>No bio available</em>'}</div>
              <div class="tag-row" style="margin-bottom:.7rem">{tags}</div>
              <div class="result-footer">
                <span class="why-similar">✦ {res['why']}</span>
                <a class="ig-link"
                   href="https://instagram.com/{res['username']}"
                   target="_blank">instagram ↗</a>
              </div>
            </div>
            """, unsafe_allow_html=True)