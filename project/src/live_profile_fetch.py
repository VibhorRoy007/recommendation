"""Live Instagram profile fetcher using Apify API.

- Reads the Apify API token from ``apify_api_token.txt`` located at the project root.
- Calls the public Apify actor ``apify/instagram-profile`` to retrieve a single profile.
- Returns a dictionary with ``username``, ``biography``, ``followersCount`` and any other metadata
  returned by the actor (e.g., ``profilePicUrl``).  Errors are raised as ``RuntimeError``.

Usage example::

    from live_profile_fetch import fetch_profile
    data = fetch_profile('some_handle')
    print(data)
"""

import json
import os
from pathlib import Path
import time
import requests

# -------------------------------------------------
# Helper to read the Apify token (plain text file)
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKEN_PATH = PROJECT_ROOT / "apify_api_token.txt"

def _load_token() -> str:
    if not TOKEN_PATH.is_file():
        raise RuntimeError(f"Apify token file not found at {TOKEN_PATH}")
    token = TOKEN_PATH.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("Apify token file is empty")
    return token

# -------------------------------------------------
# Fetch a single Instagram profile via Apify
# -------------------------------------------------
APIFY_ACTOR = "apify~instagram-profile-scraper"
APIFY_RUN_URL = f"https://api.apify.com/v2/acts/{APIFY_ACTOR}/runs"

def fetch_profile(username: str, wait_seconds: int = 5, max_tries: int = 12) -> dict:
    """Retrieve an Instagram profile using the Apify actor.

    Parameters
    ----------
    username: str
        Instagram handle without the leading '@'.
    wait_seconds: int, optional
        Seconds to wait between status polls (default 5).
    max_tries: int, optional
        Maximum number of status polls before giving up (default 12 → ~1 min).

    Returns
    -------
    dict
        Profile data containing at least ``username``, ``biography`` and ``followersCount``.
    """
    token = _load_token()
    # Start a run with the desired input
    payload = {
        "usernames": [username]
    }
    params = {
        "token": token,
        "waitForFinish": 0  # start async, we will poll ourselves
    }
    response = requests.post(APIFY_RUN_URL, params=params, json=payload)
    if response.status_code != 201:
        raise RuntimeError(f"Failed to start Apify run: {response.status_code} {response.text}")
    run_info = response.json()
    run_id = run_info["data"]["id"]
    # Poll for completion
    status_url = f"{APIFY_RUN_URL}/{run_id}"
    for _ in range(max_tries):
        status_resp = requests.get(status_url, params={"token": token})
        if status_resp.status_code != 200:
            raise RuntimeError(f"Error checking run status: {status_resp.status_code}")
        status_data = status_resp.json()["data"]
        if status_data["status"] == "SUCCEEDED":
            break
        if status_data["status"] in ("FAILED", "ABORTED"):
            raise RuntimeError(f"Apify run failed with status {status_data['status']}")
        time.sleep(wait_seconds)
    else:
        raise RuntimeError("Apify run did not finish within the expected time")

    # Once succeeded, fetch the result dataset (the default output is a JSON with a single item)
    # The run provides a ``defaultDatasetId`` we can query.
    dataset_id = status_data.get("defaultDatasetId")
    if not dataset_id:
        raise RuntimeError("No dataset produced by the Apify run")
    dataset_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
    ds_resp = requests.get(dataset_url, params={"token": token})
    if ds_resp.status_code != 200:
        raise RuntimeError(f"Failed to retrieve dataset items: {ds_resp.status_code}")
    items = ds_resp.json()
    if not items:
        raise RuntimeError("Apify returned an empty result set for the profile")
    # The actor returns a list, usually with a single dict representing the profile
    profile = items[0]
    # Normalise keys to match our stored schema
    result = {
        "username": profile.get("username", username),
        "biography": profile.get("biography", ""),
        "followersCount": profile.get("followersCount", 0),
        "profilePicUrl": profile.get("profilePicUrl"),
        "isBusinessAccount": profile.get("isBusinessAccount", False),
        "verified": profile.get("verified", False),
    }
    return result

# ---------------------------------------------------------------------------
# Small __main__ helper for quick manual testing (not used in the pipeline)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python live_profile_fetch.py <instagram_username>")
        sys.exit(1)
    handle = sys.argv[1].lstrip("@")
    try:
        prof = fetch_profile(handle)
        print(json.dumps(prof, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")
