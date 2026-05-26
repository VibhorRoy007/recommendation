"""
Automatic generation of a labelled evaluation dataset for the
Creator Recommendation Engine (auto version).

- "similar"   → two creators share at least one curated category tag.
- "not_similar" → creators share no tags (or are random when a perfect
  mismatch cannot be found).

The script writes `evaluation_pairs_generated.csv` (creator_a,creator_b,label)
with at least TARGET_PAIRS rows (default 30).  This avoids overwriting any
existing `evaluation_pairs.csv` that might be locked.
"""

import csv
import random
from pathlib import Path
from typing import List, Set, Tuple

import pandas as pd

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]          # repo root
DATA_CSV = BASE_DIR / "data" / "processed_creators.csv"
OUT_CSV  = BASE_DIR / "evaluation_pairs_generated.csv"  # NEW file name

TARGET_PAIRS = 30                # total rows (similar + not_similar)
MAX_SIMILAR   = TARGET_PAIRS // 2   # aim for roughly 50/50 split
RANDOM_SEED   = 42
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Category detection – identical to the logic in app.py
# ----------------------------------------------------------------------
CATEGORY_KEYWORDS: dict[str, List[str]] = {
    "📸 Photography": [
        "photo", "photographer", "photography", "lens", "shoot", "shot",
        "canon", "nikon", "lightroom", "portrait", "camera",
    ],
    "👗 Fashion": [
        "fashion", "style", "stylist", "outfit", "ootd", "clothing",
        "wear", "model", "modeling", "designer", "couture", "streetwear",
    ],
    "🎵 Music": [
        "music", "musician", "singer", "rapper", "producer", "dj",
        "artist", "sound", "beats", "album", "track", "band", "guitar", "vocals",
    ],
    "💼 Entrepreneurship": [
        "entrepreneur", "founder", "ceo", "startup", "business", "brand",
        "venture", "investor", "growth", "scale", "hustle",
    ],
    "🎨 Art & Design": [
        "artist", "design", "designer", "creative", "illustrator", "art",
        "draw", "paint", "visual", "graphic", "ui", "ux",
    ],
    "🏋️ Fitness": [
        "fitness", "gym", "workout", "health", "nutrition", "athlete",
        "sport", "training", "coach", "wellness", "yoga", "bodybuilding",
    ],
    "✈️ Travel": [
        "travel", "traveler", "explorer", "adventure", "wanderlust",
        "nomad", "journey", "world", "globe", "destination",
    ],
    "🍕 Food": [
        "food", "chef", "cook", "recipe", "foodie", "restaurant",
        "cuisine", "baking", "gastronomy", "culinary",
    ],
    "💻 Tech": [
        "tech", "developer", "engineer", "coding", "software", "ai",
        "machine learning", "data", "cyber", "startup", "saas",
    ],
    "📹 Content Creation": [
        "content", "creator", "youtube", "youtuber", "tiktok", "reels",
        "video", "vlog", "blogger", "influencer", "podcast",
    ],
    "🌿 Lifestyle": [
        "lifestyle", "life", "daily", "routine", "mindset", "motivation",
        "inspiration", "positive", "self", "personal",
    ],
    "💄 Beauty": [
        "beauty", "makeup", "skincare", "cosmetics", "glow", "mua",
        "glam", "aesthetic", "tutorial",
    ],
}


def detect_categories(bio: str) -> Set[str]:
    """Return the set of category names that appear in a bio."""
    bio_lower = bio.lower()
    found: Set[str] = set()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in bio_lower for kw in kws):
            found.add(cat)
    return found if found else {"🌐 General"}

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def load_creators() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, usecols=["username", "biography"])
    df["categories"] = df["biography"].apply(detect_categories)
    return df

def pick_similar_pairs(df: pd.DataFrame, max_pairs: int) -> List[Tuple[str, str]]:
    similar: List[Tuple[str, str]] = []
    usernames = list(df["username"])
    random.shuffle(usernames)
    # category → usernames map
    cat_to_users: dict[str, List[str]] = {}
    for _, row in df.iterrows():
        for cat in row["categories"]:
            cat_to_users.setdefault(cat, []).append(row["username"])
    attempts = 0
    while len(similar) < max_pairs and attempts < max_pairs * 10:
        a = random.choice(usernames)
        a_cats = df.loc[df["username"] == a, "categories"].iloc[0]
        cat = random.choice(list(a_cats))
        candidates = [u for u in cat_to_users[cat] if u != a]
        if not candidates:
            attempts += 1
            continue
        b = random.choice(candidates)
        pair = tuple(sorted((a, b)))
        if pair not in similar:
            similar.append(pair)
        attempts += 1
    return similar

def pick_not_similar_pairs(
    df: pd.DataFrame, existing_pairs: Set[Tuple[str, str]], max_pairs: int
) -> List[Tuple[str, str]]:
    not_sim: List[Tuple[str, str]] = []
    usernames = list(df["username"])
    attempts = 0
    while len(not_sim) < max_pairs and attempts < max_pairs * 20:
        a, b = random.sample(usernames, 2)
        pair = tuple(sorted((a, b)))
        if pair in existing_pairs:
            attempts += 1
            continue
        a_cats = df.loc[df["username"] == a, "categories"].iloc[0]
        b_cats = df.loc[df["username"] == b, "categories"].iloc[0]
        if a_cats.isdisjoint(b_cats):
            not_sim.append(pair)
        attempts += 1
    return not_sim

# ----------------------------------------------------------------------
# Main – writes evaluation_pairs_generated.csv
# ----------------------------------------------------------------------
def main() -> None:
    random.seed(RANDOM_SEED)
    df = load_creators()
    if df.empty:
        raise RuntimeError("processed_creators.csv is empty or missing")

    similar_target = min(MAX_SIMILAR, len(df) // 2)
    not_sim_target = TARGET_PAIRS - similar_target

    similar_pairs = pick_similar_pairs(df, similar_target)
    similar_set = {tuple(sorted(p)) for p in similar_pairs}
    not_sim_pairs = pick_not_similar_pairs(df, similar_set, not_sim_target)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["creator_a", "creator_b", "label"])
        for a, b in similar_pairs:
            writer.writerow([a, b, "similar"])
        for a, b in not_sim_pairs:
            writer.writerow([a, b, "not_similar"])
    total = len(similar_pairs) + len(not_sim_pairs)
    print(f"✅ Generated {total} rows → {OUT_CSV}")
    print("   • similar      :", len(similar_pairs))
    print("   • not_similar  :", len(not_sim_pairs))
    print("\nRun the evaluation script against this file:")
    print("   python project/src/evaluate.py")
    print("(If you need the exact name `evaluation_pairs.csv`, rename the file.)")

if __name__ == "__main__":
    main()
