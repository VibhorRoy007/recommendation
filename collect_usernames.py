"""
Step 1 of data collection: combine all your Apify scrape outputs,
extract usernames, deduplicate, and save a clean list ready for
the Instagram Profile Scraper.

HOW TO USE:
  - Put all your Apify JSON files (hashtag scraper + follower scraper outputs)
    in the same folder as this script.
  - Update FILE_PATTERNS to match your filenames.
  - Run: python collect_usernames.py
  - Output: usernames_clean.txt  (one username per line, ready for Apify)
"""

import json
import glob
import os
import re

# ── CONFIG ──────────────────────────────────────────────────────────────────────

# Glob patterns to find all your Apify output files
FILE_PATTERNS = [
    "hashtag_scrapper.json",                    # Your hashtag scraper output
    "instagram_follower_scraper.json",          # Your follower scraper output
    "dataset_*.json",                           # In case you have dataset_ files
    "*instagram*.json",                         # Catch-all for any Instagram JSON
    "*scraper*.json",                           # Catch-all for any scraper JSON
]

OUTPUT_FILE = "usernames_clean.txt"

# ── USERNAME EXTRACTION ───────────────────────────────────────────────────────

def extract_username(item: dict) -> str | None:
    """
    Apify scrapers use different field names depending on the actor.
    This tries all known ones.
    """
    for field in ["ownerUsername", "username", "Username", "handle", "owner_username"]:
        val = item.get(field)
        if val and isinstance(val, str):
            return val.strip().lstrip("@").lower()
    return None


def is_valid_username(username: str) -> bool:
    """Basic sanity check on Instagram usernames."""
    if not username:
        return False
    if len(username) < 2 or len(username) > 30:
        return False
    # Instagram usernames: letters, numbers, dots, underscores only
    if not re.match(r"^[a-z0-9._]+$", username):
        return False
    return True


def load_all_files() -> list[dict]:
    """Load all JSON files matching the patterns."""
    all_items = []
    files_found = []

    for pattern in FILE_PATTERNS:
        matched = glob.glob(pattern)
        files_found.extend(matched)

    if not files_found:
        print("No JSON files found. Make sure your Apify output files are")
        print("in the same directory as this script.")
        print(f"Looking for patterns: {FILE_PATTERNS}")
        return []

    print(f"Found {len(files_found)} file(s):")
    for f in files_found:
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            # Apify sometimes wraps in {"items": [...]}
            if isinstance(data, dict) and "items" in data:
                data = data["items"]
            print(f"    {f}  -> {len(data)} records")
            all_items.extend(data)

    return all_items


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("\nLoading Apify output files...\n")
    all_items = load_all_files()

    if not all_items:
        return

    print(f"Total raw records: {len(all_items)}")

    # Extract and clean usernames
    raw_usernames = [extract_username(item) for item in all_items]
    raw_usernames = [u for u in raw_usernames if u]
    print(f"Usernames extracted: {len(raw_usernames)}")

    # Validate and deduplicate
    valid = [u for u in raw_usernames if is_valid_username(u)]
    unique = sorted(set(valid))
    print(f"After validation + dedup: {len(unique)}")
    print(f"Duplicates removed: {len(valid) - len(unique)}")

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(unique))

    print(f"\nSaved to: {OUTPUT_FILE}")
    print(f"\nNext step:")
    print(f"Upload the {OUTPUT_FILE} file to Apify Instagram Profile Scraper")
    print(f"as the username input list.")
    print(f"\nIf you have more than 100 usernames, split into batches of 100")
    print(f"(free tier limit). Use the split helper below:")
    print(f"python collect_usernames.py --split 100")

    # Optional: split into batches
    import sys
    if "--split" in sys.argv:
        idx = sys.argv.index("--split")
        batch_size = int(sys.argv[idx + 1])
        batches = [unique[i:i+batch_size] for i in range(0, len(unique), batch_size)]
        for i, batch in enumerate(batches):
            fname = f"usernames_batch_{i+1:02d}.txt"
            with open(fname, "w") as f:
                f.write("\n".join(batch))
        print(f"\nSplit into {len(batches)} batches of {batch_size} or less saved as usernames_batch_XX.txt")


if __name__ == "__main__":
    main()