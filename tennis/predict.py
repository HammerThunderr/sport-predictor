"""
predict.py
----------
Turns the current ratings table into a win probability for any matchup.

The evaluation in train.py showed Elo + surface-Elo carry ~75% of all
predictive signal, and a pure-Elo baseline already hits ~63.6% vs the
full model's 64.8%. So the shippable v1 predictor is just the Elo
formula, blended 60/40 between surface-specific and overall Elo. No ML
library needed at inference time -- ideal for a GitHub Actions job.

Upgrade path: load model.pkl and feed the full feature vector (form,
h2h, rank) if you want the extra ~1 point of accuracy.
"""
import json
import sys

SURFACE_WEIGHT = 0.6  # weight on surface Elo vs overall


def win_prob(p1, p2, surface, ratings):
    a, b = ratings[p1], ratings[p2]
    s = surface.lower()
    e1 = SURFACE_WEIGHT * a.get(s, a["elo"]) + (1 - SURFACE_WEIGHT) * a["elo"]
    e2 = SURFACE_WEIGHT * b.get(s, b["elo"]) + (1 - SURFACE_WEIGHT) * b["elo"]
    return 1.0 / (1.0 + 10 ** ((e2 - e1) / 400.0))


def predict_fixtures(fixtures, ratings):
    """fixtures: list of {p1, p2, surface}. Returns predictions list."""
    out = []
    for f in fixtures:
        if f["p1"] not in ratings or f["p2"] not in ratings:
            out.append({**f, "error": "player not in ratings"})
            continue
        p = win_prob(f["p1"], f["p2"], f["surface"], ratings)
        fav = f["p1"] if p >= 0.5 else f["p2"]
        out.append({**f, "p1_win_prob": round(p, 3),
                    "favourite": fav, "confidence": round(abs(p - 0.5) * 2, 3)})
    return out


if __name__ == "__main__":
    ratings = json.load(open("ratings.json"))
    demo = [
        {"p1": "Carlos Alcaraz", "p2": "Jannik Sinner", "surface": "Clay"},
        {"p1": "Carlos Alcaraz", "p2": "Jannik Sinner", "surface": "Hard"},
        {"p1": "Novak Djokovic", "p2": "Alex de Minaur", "surface": "Hard"},
        {"p1": "Rafael Nadal", "p2": "Alexander Zverev", "surface": "Clay"},
    ]
    for r in predict_fixtures(demo, ratings):
        print(f"{r['p1']} vs {r['p2']} ({r['surface']}): "
              f"{r['p1']} wins {r['p1_win_prob']*100:.1f}%  -> fav {r['favourite']}")
