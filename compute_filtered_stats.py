import json, glob, os

# Config same as filter_profiles.py
MIN_FOLLOWERS = 500
MAX_FOLLOWERS = 500_000
MIN_BIO_LENGTH = 40
MIN_POSTS = 10

CREATIVE_KEYWORDS = [
    "photographer", "photography", "photo", "cinematographer",
    "stylist", "styling", "fashion", "wardrobe",
    "designer", "design", "graphic", "illustrator", "illustration",
    "artist", "art", "painter", "sketch", "drawing",
    "architect", "interior", "space design",
    "filmmaker", "film", "director", "producer", "editor", "cinemat",
    "documentary", "screenwriter", "screenplay",
    "writer", "author", "poet", "journalist", "blogger",
    "copywriter", "content creator", "storyteller", "editor",
    "musician", "music", "singer", "vocalist", "composer",
    "dj", "beatmaker", "sound designer", "rapper", "band",
    "comedian", "standup", "stand-up", "improv", "actor", "actress",
    "dancer", "choreographer", "performer",
    "yoga", "wellness", "nutritionist", "chef", "food", "recipe",
    "baker", "barista", "mixologist",
    "podcaster", "podcast", "creative", "curator", "brand",
    "solopreneur", "freelancer",
]

BUSINESS_EXCLUSION_KEYWORDS = [
    "distributor", "wholesaler", "retailer", "manufacturer",
    "export", "import", "trading", "pvt ltd", "private limited",
    "shop", "store", "order via", "cash on delivery", "cod",
    "free delivery", "delivery across", "bulk order",
    "steel", "cement", "tiles", "chemicals", "pharma",
]

def has_creative_signal(bio: str) -> bool:
    bio_lower = bio.lower()
    return any(kw in bio_lower for kw in CREATIVE_KEYWORDS)

def is_business_account_by_bio(bio: str) -> bool:
    bio_lower = bio.lower()
    if has_creative_signal(bio):
        return False
    return any(kw in bio_lower for kw in BUSINESS_EXCLUSION_KEYWORDS)

def filter_profile(profile: dict):
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
        return False, "too many followers"
    if posts < MIN_POSTS:
        return False, f"too few posts ({posts})"
    if not has_creative_signal(bio):
        return False, "no creative keywords in bio"
    if is_business_account_by_bio(bio):
        return False, "business account"
    return True, "ok"

input_files = [
    "C:/Users/ASUS/Desktop/bb/dataset_instagram-profile-scraper_2026-05-21_15-32-44-670.json",
    "C:/Users/ASUS/Desktop/bb/dataset_instagram-profile-scraper_2026-05-21_20-09-46-812.json",
    "C:/Users/ASUS/Desktop/bb/dataset_instagram-profile-scraper_2026-05-21_20-17-51-805.json",
]

all_profiles = []
for fpath in input_files:
    if not os.path.exists(fpath):
        continue
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "items" in data:
        data = data["items"]
    all_profiles.extend(data)

# deduplicate
seen = set()
unique = []
for p in all_profiles:
    u = (p.get("username") or "").lower().strip()
    if u and u not in seen:
        seen.add(u)
        unique.append(p)

filtered_out = 0
filtered_followers_sum = 0
for p in unique:
    keep, _ = filter_profile(p)
    if not keep:
        filtered_out += 1
        filtered_followers_sum += p.get("followersCount", 0)

print(f"Total unique profiles: {len(unique)}")
print(f"Filtered out profiles: {filtered_out}")
print(f"Total followers of filtered-out profiles: {filtered_followers_sum}")
