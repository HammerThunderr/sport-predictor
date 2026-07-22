"""
update_ratings.py
-----------------
Replays the full match history chronologically and writes ratings.json:
the current overall + per-surface Elo for every active player.

This is the piece your daily GitHub Actions job runs. It's deterministic
and cheap (a few seconds over 20 years of matches), so you just re-run it
whenever new results land, then feed ratings.json to predict.py.
"""
import pandas as pd
import json
from collections import defaultdict

START = 1500.0
ACTIVE_SINCE = 20240101   # only export players seen on/after this date
MIN_MATCHES = 20


def k_factor(n):
    return 250.0 / ((n + 5) ** 0.4)


def expected(a, b):
    return 1.0 / (1.0 + 10 ** ((b - a) / 400.0))


def build_ratings(raw):
    raw = raw.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
    elo = defaultdict(lambda: START)
    selo = defaultdict(lambda: START)
    n = defaultdict(int)
    ns = defaultdict(int)
    name, last = {}, {}

    for r in raw.itertuples(index=False):
        w, l = r.winner_id, r.loser_id
        s = r.surface if isinstance(r.surface, str) else "Unknown"
        name[w], name[l] = r.winner_name, r.loser_name
        ew, el = elo[w], elo[l]
        ex = expected(ew, el)
        elo[w] = ew + k_factor(n[w]) * (1 - ex)
        elo[l] = el + k_factor(n[l]) * -(1 - ex)
        sw, sl = selo[(s, w)], selo[(s, l)]
        exs = expected(sw, sl)
        selo[(s, w)] = sw + k_factor(ns[(s, w)]) * (1 - exs)
        selo[(s, l)] = sl + k_factor(ns[(s, l)]) * -(1 - exs)
        n[w] += 1; n[l] += 1; ns[(s, w)] += 1; ns[(s, l)] += 1
        last[w] = last[l] = r.tourney_date

    ratings = {}
    for pid, nm in name.items():
        if last.get(pid, 0) >= ACTIVE_SINCE and n[pid] >= MIN_MATCHES:
            ratings[nm] = {
                "elo": round(elo[pid], 1),
                "hard": round(selo[("Hard", pid)], 1),
                "clay": round(selo[("Clay", pid)], 1),
                "grass": round(selo[("Grass", pid)], 1),
                "matches": n[pid],
            }
    return ratings


if __name__ == "__main__":
    raw = pd.read_csv("raw_all.csv")
    ratings = build_ratings(raw)
    json.dump(ratings, open("ratings.json", "w"), indent=1)
    print(f"wrote ratings.json with {len(ratings)} active players")
