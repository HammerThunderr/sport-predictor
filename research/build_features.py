"""
build_features.py
-----------------
Turns raw ATP match rows (winner/loser format) into a leakage-free,
match-level training table where every feature is known BEFORE the match.

Core ideas:
  * One chronological pass over all matches.
  * Maintain running Elo (global + per-surface), head-to-head, and recent
    form. For each match we RECORD the pre-match state, THEN update it.
  * Rows are winner-centric during the pass, then randomly flipped so the
    target isn't always "player 1 wins" (that would be trivial leakage).
"""
import pandas as pd
import numpy as np
from collections import defaultdict, deque

RNG = np.random.default_rng(42)
START_ELO = 1500.0
FORM_WINDOW = 20  # matches used for recent win-rate


def k_factor(n_matches):
    # Decaying K: volatile for newcomers, stable for veterans (538-style).
    return 250.0 / ((n_matches + 5) ** 0.4)


def expected(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def build(raw):
    elo = defaultdict(lambda: START_ELO)
    surf_elo = defaultdict(lambda: START_ELO)     # key (surface, player)
    n_played = defaultdict(int)
    n_played_surf = defaultdict(int)              # key (surface, player)
    h2h = defaultdict(int)                          # key (a,b) = a's wins over b
    form = defaultdict(lambda: deque(maxlen=FORM_WINDOW))  # 1=win 0=loss

    rows = []
    for r in raw.itertuples(index=False):
        w, l = r.winner_id, r.loser_id
        surf = r.surface if isinstance(r.surface, str) else "Unknown"

        # ---- snapshot pre-match state (winner-centric) ----
        we, le = elo[w], elo[l]
        wse, lse = surf_elo[(surf, w)], surf_elo[(surf, l)]
        w_form = np.mean(form[w]) if form[w] else 0.5
        l_form = np.mean(form[l]) if form[l] else 0.5
        h2h_w = h2h[(w, l)] - h2h[(l, w)]  # winner's net h2h edge

        def num(x):
            return x if pd.notna(x) else np.nan

        feat = dict(
            tourney_date=r.tourney_date,
            surface=surf,
            best_of=num(getattr(r, "best_of", np.nan)),
            # winner-minus-loser diffs (positive => winner was favoured)
            elo_diff=we - le,
            surf_elo_diff=wse - lse,
            rank_diff=(num(r.loser_rank) - num(r.winner_rank)),        # +ve => winner higher-ranked
            rank_pts_diff=(num(r.winner_rank_points) - num(r.loser_rank_points)),
            age_diff=(num(r.winner_age) - num(r.loser_age)),
            ht_diff=(num(r.winner_ht) - num(r.loser_ht)),
            form_diff=w_form - l_form,
            h2h_diff=h2h_w,
            exp_diff=n_played[w] - n_played[l],  # experience (matches played)
        )
        rows.append(feat)

        # ---- update state AFTER recording ----
        k_w = k_factor(n_played[w])
        k_l = k_factor(n_played[l])
        exp_w = expected(we, le)
        elo[w] = we + k_w * (1 - exp_w)
        elo[l] = le + k_l * (0 - (1 - exp_w))

        ks_w = k_factor(n_played_surf[(surf, w)])
        ks_l = k_factor(n_played_surf[(surf, l)])
        exp_ws = expected(wse, lse)
        surf_elo[(surf, w)] = wse + ks_w * (1 - exp_ws)
        surf_elo[(surf, l)] = lse + ks_l * (0 - (1 - exp_ws))

        n_played[w] += 1; n_played[l] += 1
        n_played_surf[(surf, w)] += 1; n_played_surf[(surf, l)] += 1
        h2h[(w, l)] += 1
        form[w].append(1); form[l].append(0)

    df = pd.DataFrame(rows)

    # ---- randomly flip so target is balanced, not "winner is always p1" ----
    flip = RNG.random(len(df)) < 0.5
    diff_cols = ["elo_diff", "surf_elo_diff", "rank_diff", "rank_pts_diff",
                 "age_diff", "ht_diff", "form_diff", "h2h_diff", "exp_diff"]
    df.loc[flip, diff_cols] *= -1
    df["target"] = np.where(flip, 0, 1)  # 1 = player1 (the +diff side) won
    return df


if __name__ == "__main__":
    raw = pd.read_csv("raw_all.csv")
    raw = raw.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
    df = build(raw)
    df.to_csv("features.csv", index=False)
    print("feature rows:", len(df))
    print(df[["elo_diff", "surf_elo_diff", "form_diff", "h2h_diff", "target"]].describe().round(2))
