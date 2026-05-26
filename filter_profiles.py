"""
Step 2 of data collection: take the raw output from Apify's Instagram
Profile Scraper and filter it down to quality multi-hyphenate creative
profiles that are actually useful for the recommendation engine.

HOW TO USE:
    python filter_profiles.py  --input  dataset_instagram-profile-scraper_*.json
                                --output creators_clean.json

Or just run it and edit the INPUT_FILES list below.
"""

import json
import glob
import argparse
import re
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────────────

# Edit these or pass via --input flag
INPUT_FILES = [
    "dataset_instagram-profile-scraper_2026-05-21_15-32-44-670.json",
    # Add more files here as you scrape more batches:
    # "dataset_instagram-profile-scraper_batch2.json",
]

DEFAULT_OUTPUT = "creators_clean.json"

# Follower range for real creators (not bots, not mega-celebs)
MIN_FOLLOWERS = 200
MAX_FOLLOWERS = 500_000

# Bio must be longer than this (character count)
MIN_BIO_LENGTH = 10

# Must have posted at least this many times
MIN_POSTS = 10

# ── KEYWORD LISTS ─────────────────────────────────────────────────────────────

CREATIVE_KEYWORDS = [
    # Visual arts
    "photographer", "photography", "photo", "cinematographer",
    "stylist", "styling", "fashion", "wardrobe",
    "designer", "design", "graphic", "illustrator", "illustration",
    "artist", "art", "painter", "sketch", "drawing",
    "architect", "interior", "space design",
    # Film & media
    "filmmaker", "film", "director", "producer", "editor", "cinemat",
    "documentary", "screenwriter", "screenplay",
    # Writing
    "writer", "author", "poet", "journalist", "blogger",
    "copywriter", "content creator", "storyteller", "editor",
    # Music
    "musician", "music", "singer", "vocalist", "composer",
    "dj", "beatmaker", "sound designer", "rapper", "band",
    # Performance
    "comedian", "standup", "stand-up", "improv", "actor", "actress",
    "dancer", "choreographer", "performer",
    # Wellness & lifestyle
    "yoga", "wellness", "nutritionist", "chef", "food", "recipe",
    "baker", "barista", "mixologist",
    # Digital / modern
    "podcaster", "podcast", "creative", "curator", "brand",
    "solopreneur", "freelancer", "founder", "co-founder", "art", "model", "video", "editor", "copywriter", "strategy", "marketing", "host", "mc",
]

# Bios containing these are likely shops/brands, not individual creators
# (applied only if no strong creative signal)
BUSINESS_EXCLUSION_KEYWORDS = [
    "shipping worldwide", "order via dm", "no refunds",
    "shop now", "wholesale", "cash on delivery", "cod available"
]

# Bio separators that suggest multiple disciplines (multi-hyphenate signal)
MULTI_HYPHENATE_SEPARATORS = ["|", "+", "•", "·", "×", "&", " and ", " + "]


# ── FILTERING LOGIC ───────────────────────────────────────────────────────────

def has_creative_signal(bio: str) -> bool:
    bio_lower = bio.lower()
    return any(kw in bio_lower for kw in CREATIVE_KEYWORDS)


def is_business_account_by_bio(bio: str) -> bool:
    """True if bio strongly reads as a shop/trading business."""
    bio_lower = bio.lower()
    # Only exclude if no creative signal at all
    if has_creative_signal(bio):
        return False
    return any(kw in bio_lower for kw in BUSINESS_EXCLUSION_KEYWORDS)


def multi_hyphenate_score(bio: str) -> int:
    """
    Returns 0-3+ indicating how multi-hyphenate the bio reads.
    Higher = more likely our ICP.
    """
    score = 0
    bio_lower = bio.lower()

    # Count how many creative keywords appear
    keyword_hits = sum(1 for kw in CREATIVE_KEYWORDS if kw in bio_lower)
    if keyword_hits >= 2:
        score += 1
    if keyword_hits >= 3:
        score += 1

    # Check for separator patterns (fashion stylist | photographer)
    has_separator = any(sep in bio for sep in MULTI_HYPHENATE_SEPARATORS)
    if has_separator:
        score += 1

    return score


def filter_profile(profile: dict) -> tuple[bool, str]:
    """
    Returns (keep: bool, reason: str).
    """
    bio = (profile.get("biography") or "").strip()
    followers = profile.get("followersCount") or 0
    posts = profile.get("postsCount") or 0
    is_private = profile.get("private", False)

    if is_private:
        return False, "private account"
    if not bio:
        return False, "empty bio"
    if len(bio) < MIN_BIO_LENGTH:
        return False, f"bio too short ({len(bio)} chars)"
    if followers < MIN_FOLLOWERS:
        return False, f"too few followers ({followers})"
    if followers > MAX_FOLLOWERS:
        return False, f"too many followers ({followers}) - likely mega-celeb"
    if posts < MIN_POSTS:
        return False, f"too few posts ({posts})"
    if not has_creative_signal(bio):
        return False, "no creative keywords in bio"
    if is_business_account_by_bio(bio):
        return False, "reads as shop/business, not individual creator"

    return True, "ok"


def enrich_profile(profile: dict) -> dict:
    """Add computed fields useful for ML."""
    bio = (profile.get("biography") or "").strip()
    bio_lower = bio.lower()

    matched_keywords = [kw for kw in CREATIVE_KEYWORDS if kw in bio_lower]
    mh_score = multi_hyphenate_score(bio)

    return {
        "username": profile.get("username", ""),
        "fullName": profile.get("fullName", ""),
        "biography": bio,
        "followersCount": profile.get("followersCount", 0),
        "followsCount": profile.get("followsCount", 0),
        "postsCount": profile.get("postsCount", 0),
        "isBusinessAccount": profile.get("isBusinessAccount", False),
        "verified": profile.get("verified", False),
        # ── computed ──
        "matched_disciplines": matched_keywords,
        "discipline_count": len(matched_keywords),
        "multi_hyphenate_score": mh_score,
        "is_multi_hyphenate": mh_score >= 1,
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Filter Instagram profiles to quality creators")
    parser.add_argument("--input", nargs="+", help="Input JSON file(s)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSON file")
    args = parser.parse_args()

    input_files = args.input or INPUT_FILES

    # Expand globs
    expanded = []
    for pattern in input_files:
        matched = glob.glob(pattern)
        expanded.extend(matched if matched else [pattern])

    print(f"\nLoading {len(expanded)} file(s)...")
    all_profiles = []
    for fpath in expanded:
        if not Path(fpath).exists():
            print(f"    File not found: {fpath} - skipping")
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "items" in data:
            data = data["items"]
        print(f"    {fpath}  -> {len(data)} profiles")
        all_profiles.extend(data)

    total = len(all_profiles)
    print(f"\nTotal raw profiles: {total:,}")

    # Deduplicate by username
    seen = set()
    unique_profiles = []
    for p in all_profiles:
        u = (p.get("username") or "").lower().strip()
        if u and u not in seen:
            seen.add(u)
            unique_profiles.append(p)
    print(f"  After dedup: {len(unique_profiles):,}  ({total - len(unique_profiles):,} duplicates removed)")

    # Filter
    kept = []
    rejection_reasons: dict[str, int] = {}

    for profile in unique_profiles:
        keep, reason = filter_profile(profile)
        if keep:
            kept.append(enrich_profile(profile))
        else:
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    print(f"\nProfiles kept: {len(kept):,}  ({len(kept)/len(unique_profiles)*100:.1f}%)")
    print(f"  Profiles removed: {len(unique_profiles) - len(kept):,}")

    print(f"\n   Rejection breakdown:")
    for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
        print(f"     {count:4d}  {reason}")

    # Stats on kept profiles
    if kept:
        multi = [p for p in kept if p["is_multi_hyphenate"]]
        print(f"\nQuality stats on kept profiles:")
        print(f"     Multi-hyphenate (score >= 1): {len(multi):,}  ({len(multi)/len(kept)*100:.1f}%)")
        avg_followers = sum(p["followersCount"] for p in kept) / len(kept)
        avg_bio_len = sum(len(p["biography"]) for p in kept) / len(kept)
        print(f"     Avg followers:  {avg_followers:,.0f}")
        print(f"     Avg bio length: {avg_bio_len:.0f} chars")

        # Top disciplines
        from collections import Counter
        all_disciplines = []
        for p in kept:
            all_disciplines.extend(p["matched_disciplines"])
        top = Counter(all_disciplines).most_common(10)
        print(f"\n   Top 10 disciplines in your database:")
        for disc, count in top:
            print(f"     {count:4d}  {disc}")

    # Save
    output_path = args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(kept):,} profiles to: {output_path}")
    print(f"\nNext step: run recommend.py or app.py with --data {output_path}")

    if len(kept) < 1000:
        needed = 1000 - len(kept)
        print(f"\nYou have {len(kept)} profiles — you need {needed} more to hit the 1,000 target.")
        print(f"Scrape more hashtags or follower lists and re-run this script.")
        print(f"Add new JSON files to INPUT_FILES at the top of this script.")


if __name__ == "__main__":
    main()