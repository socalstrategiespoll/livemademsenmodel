"""
civicAPI live feed for the Massachusetts Senate Democratic Primary (Markey vs.
Moulton), built for the 2-candidate election_model_template.py architecture.

Endpoint:  https://civicapi.org/api/v2/race/{race_id}
Race:      87556 (2026 Massachusetts U.S. Senate Democratic Primary — confirmed
           via https://civicapi.org/results/elections/87556). Election day is
           September 1, 2026 -- polls close 8:00 PM ET. Two other Democrats
           (William Gates, Alexander Rikleen) are also on the ballot; they
           aren't tracked individually and will fall into
           state_dropped_other_candidates / dropped_other_candidates per town,
           not into either major candidate's total, per the two-candidate
           framing Wilson specified.
Auth:      none, per the WI/MI precedent. Attribution required: credit
           civicapi.org anywhere this output is published.

Deductive (not vote-mode) client, same reasoning as Wisconsin's: no theta, no
mode gap, no sub-city feeds. Just Markey/Moulton votes per town, handed to
ElectionModel.update_county().
"""

import re
import time
import unicodedata

try:
    import requests
except ImportError:
    requests = None

API_BASE = "https://civicapi.org/api/v2"
MA_SENATE_DEM_PRIMARY = 87556

# Substring match keys -- VERIFY against the actual payload once reachable.
MARKEY_KEYS = ("markey",)
MOULTON_KEYS = ("moulton",)

REQUEST_TIMEOUT = 15
MAX_RETRIES = 4


def normalize_town(name: str) -> str:
    """Reduce a MA town/city name to a matching key. Handles hyphenated names
    (Manchester-by-the-Sea) and the handful of towns that could plausibly
    arrive with 'Town of' / 'City of' prefixes from a feed."""
    if name is None:
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"\b(town|city) of\b", " ", text)
    text = text.replace("-", " ").replace(".", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def build_town_lookup(town_names) -> dict:
    return {normalize_town(t): t for t in town_names}


def fetch_race(race_id=MA_SENATE_DEM_PRIMARY, timeout: int = REQUEST_TIMEOUT,
               max_retries: int = MAX_RETRIES, session=None) -> dict:
    """GET a race payload, retrying on transient failure with backoff.
    Raises on exhaustion -- callers should catch and keep the last good snapshot."""
    if requests is None:
        raise RuntimeError("requests is not installed: pip install requests")

    url = "{}/race/{}".format(API_BASE, race_id)
    getter = session.get if session is not None else requests.get
    last_error = None

    for attempt in range(max_retries):
        try:
            response = getter(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    raise RuntimeError("civicAPI fetch failed after {} attempts: {}".format(
        max_retries, last_error))


def _match_candidate(name: str, keys: tuple) -> bool:
    lowered = str(name).lower()
    return any(k in lowered for k in keys)


def extract_two_way(candidate_list: list) -> tuple:
    """Pull Markey and Moulton votes out of a candidate array. If any other
    candidate qualifies for the 2026 ballot and starts polling non-trivially,
    add a third 'other' bucket here AND to CANDIDATES in ma_senate_model.py --
    until then, anyone else on the ballot is dropped rather than silently
    folded into either major candidate's total.
    Returns (markey, moulton, matched_names, dropped_votes)."""
    markey = moulton = dropped = 0
    matched = {"markey": None, "moulton": None}

    for entry in candidate_list or []:
        name = entry.get("name", "")
        votes = int(entry.get("votes") or 0)
        if _match_candidate(name, MARKEY_KEYS):
            markey += votes
            matched["markey"] = name
        elif _match_candidate(name, MOULTON_KEYS):
            moulton += votes
            matched["moulton"] = name
        else:
            dropped += votes

    return markey, moulton, matched, dropped


def parse_payload(payload: dict, town_names) -> dict:
    """Turn a civicAPI race payload into town-level Markey/Moulton vote counts.
    UNVERIFIED against a real payload -- see module docstring."""
    lookup = build_town_lookup(town_names)

    state_markey, state_moulton, matched_names, state_dropped = extract_two_way(
        payload.get("candidates"))

    records = {}
    unmatched = []

    for _slug, region in (payload.get("region_results") or {}).items():
        region_type = str(region.get("type", "")).lower()
        if region_type not in ("town", "city", "municipality", ""):
            continue
        raw_name = region.get("name", _slug)
        key = normalize_town(raw_name)
        town = lookup.get(key)
        if town is None:
            unmatched.append(raw_name)
            continue

        markey, moulton, _, dropped = extract_two_way(region.get("candidates"))
        total = markey + moulton
        if total <= 0:
            continue

        records[town] = {
            "markey": markey,
            "moulton": moulton,
            "dropped_other_candidates": dropped,
            "percent_precincts": region.get("percent_reporting"),
        }

    return {
        "election_name": payload.get("election_name"),
        "last_updated": payload.get("last_updated"),
        "percent_precincts_statewide": payload.get("percent_reporting"),
        "state_markey": state_markey,
        "state_moulton": state_moulton,
        "state_dropped_other_candidates": state_dropped,
        "candidate_names": matched_names,
        "towns": records,
        "unmatched": unmatched,
    }
