"""
fetch_and_predict.py
--------------------
The daily job. Pulls today's fixtures from a tennis API, matches each
player to our Elo ratings, computes win probabilities, writes
predictions.json.

  ratings.json  +  today's fixtures  ->  predictions.json

The API key is read from the RAPIDAPI_KEY environment variable, which in
GitHub Actions comes from a repo Secret -- it is NEVER committed to the
repo. Only the ONE function `fetch_todays_fixtures()` is provider-specific;
swap its innards for whichever tennis API you subscribe to and the rest
stays identical.
"""
import os
import re
import json
import unicodedata
import datetime as dt
from difflib import get_close_matches
from urllib.request import Request, urlopen

from predict import win_prob   # reuse the Elo logic from predict.py

RATINGS_FILE = "ratings.json"
ALIAS_FILE = "aliases.json"        # manual name overrides you top up over time
OUT_FILE = "predictions.json"


# ---------------------------------------------------------------------------
# 1. PROVIDER-SPECIFIC: today's fixtures. This is the only bit you edit.
# ---------------------------------------------------------------------------
def _get_page(host, key, tour, date, page):
    """One page of the matchstat fixtures endpoint. Surface comes from the
    nested tournament.court relation, so we must request it via `include`.
    Doubles are filtered out server-side via PlayerGroup:singles."""
    from urllib.error import HTTPError
    url = (f"https://{host}/tennis/v2/{tour}/fixtures/{date}"
           f"?include=tournament.court"
           f"&filter=PlayerGroup:singles"
           f"&pageSize=100&pageNo={page}")
    req = Request(url, headers={
        "x-rapidapi-key": key,
        "x-rapidapi-host": host,
    })
    try:
        with urlopen(req, timeout=30) as r:
            return json.load(r)
    except HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:500]
        # RapidAPI puts the real reason here, e.g.
        #   403 -> "You are not subscribed to this API."
        #   401 -> "Invalid API key"
        #   429 -> "You have exceeded ... requests"
        raise SystemExit(
            f"\nAPI request failed: HTTP {e.code} on {url}\n"
            f"Server said: {body}\n"
            f"(403 = key valid but not subscribed to this API on RapidAPI; "
            f"401 = bad key; 429 = quota exceeded.)\n")


def _surface_of(m):
    """Pull surface name out of the nested tournament.court object.
    Falls back to 'Hard' if the relation wasn't returned."""
    tour = m.get("tournament") or {}
    court = (tour.get("court") or {}) if isinstance(tour, dict) else {}
    return court.get("name") or "Hard"


def fetch_todays_fixtures():
    """Return a list of {"p1","p2","surface","tournament"} for today's ATP
    singles matches, wired to the matchstat "Tennis API (ATP, WTA, ITF)".

    Only ATP is fetched because ratings.json is built from ATP history. To
    add women's matches, also fetch tour="wta" AND build a wta ratings file.
    """
    key = os.environ["RAPIDAPI_KEY"]
    host = os.environ.get("RAPIDAPI_HOST", "tennis-api-atp-wta-itf.p.rapidapi.com")
    date = dt.date.today().isoformat()          # -> "2026-07-22"

    fixtures, page = [], 1
    while True:
        payload = _get_page(host, key, "atp", date, page)
        for m in payload.get("data", []):
            p1 = (m.get("player1") or {}).get("name")
            p2 = (m.get("player2") or {}).get("name")
            if not p1 or not p2 or "/" in p1 or "/" in p2:   # skip doubles / blanks
                continue
            tour = m.get("tournament") or {}
            fixtures.append({
                "p1": p1, "p2": p2,
                "surface": _surface_of(m),
                "tournament": tour.get("name", "") if isinstance(tour, dict) else "",
            })
        if not payload.get("hasNextPage") or page >= 20:     # safety cap
            break
        page += 1
    return fixtures


# ---------------------------------------------------------------------------
# 2. Name matching: API names -> ratings.json names (provider-agnostic)
# ---------------------------------------------------------------------------
def _norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.replace("-", " ")
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


class NameMatcher:
    def __init__(self, ratings, aliases):
        self.full = {_norm(n): n for n in ratings}
        self.aliases = {_norm(k): v for k, v in aliases.items()}
        self.li, self.last = {}, {}
        for n in ratings:
            parts = _norm(n).split()
            if len(parts) >= 2:
                self.li.setdefault(f"{parts[-1]} {parts[0][0]}", []).append(n)
                self.last.setdefault(parts[-1], []).append(n)

    def match(self, api_name):
        q = _norm(api_name)
        if q in self.aliases:
            return self.aliases[q]
        if q in self.full:
            return self.full[q]
        parts = q.split()
        if len(parts) >= 2:
            for last, fi in [(parts[-1], parts[0][0]), (parts[0], parts[-1][0])]:
                hit = self.li.get(f"{last} {fi}")
                if hit and len(hit) == 1:
                    return hit[0]
            for p in parts:
                hit = self.last.get(p)
                if hit and len(hit) == 1:
                    return hit[0]
        fz = get_close_matches(q, list(self.full), n=1, cutoff=0.86)
        return self.full[fz[0]] if fz else None


# ---------------------------------------------------------------------------
# 3. Orchestrate
# ---------------------------------------------------------------------------
def main():
    ratings = json.load(open(RATINGS_FILE))
    aliases = json.load(open(ALIAS_FILE)) if os.path.exists(ALIAS_FILE) else {}
    matcher = NameMatcher(ratings, aliases)

    fixtures = fetch_todays_fixtures()
    predictions, unmatched = [], []

    for f in fixtures:
        m1, m2 = matcher.match(f["p1"]), matcher.match(f["p2"])
        if not m1 or not m2:
            unmatched.append(f["p1"] if not m1 else f["p2"])
            continue
        p = win_prob(m1, m2, f["surface"], ratings)
        predictions.append({
            "player1": m1, "player2": m2, "surface": f["surface"],
            "tournament": f.get("tournament", ""),
            "p1_win_prob": round(p, 3),
            "favourite": m1 if p >= 0.5 else m2,
            "confidence": round(abs(p - 0.5) * 2, 3),
        })

    out = {
        "generated": dt.datetime.utcnow().isoformat() + "Z",
        "count": len(predictions),
        "predictions": predictions,
    }
    json.dump(out, open(OUT_FILE, "w"), indent=1)
    print(f"wrote {len(predictions)} predictions to {OUT_FILE}")
    if unmatched:
        # These need entries in aliases.json so they resolve next run.
        print("UNMATCHED (add to aliases.json):", sorted(set(unmatched)))


if __name__ == "__main__":
    main()
