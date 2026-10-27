"""How well does each brain read a NEW player in the first rounds? Leave-one-player-out."""
import sys
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline

df = pd.read_csv(sys.argv[1]).sort_values(["id", "period"])
M = {"r": 0, "p": 1, "s": 2}; RES = {"win": 0, "lose": 1, "tie": 2}
df["c"] = df.choice.map(M); df["o"] = df["other.choice"].map(M); df["res"] = df.result.map(RES)
g = df.groupby("id")
df["c1"] = g.c.shift(1); df["o1"] = g.o.shift(1); df["res1"] = g.res.shift(1)
df["c2"] = g.c.shift(2); df["res2"] = g.res.shift(2)
print("first-round choice share:", df[df.period == 1].choice.value_counts(normalize=True).round(3).to_dict())
d1 = df.dropna(subset=["c1"]).copy()
for col in ["c1", "o1", "res1"]: d1[col] = d1[col].astype(int)
d1["y_rel"] = (d1.c - d1.c1) % 3; d1["o_rel"] = (d1.o1 - d1.c1) % 3

def personal(x, prior, w):
    counts = np.ones((3, 3, 3)) if prior is None else 1 + w * prior
    hits = []
    for r, orl, yr in zip(x.res1, x.o_rel, x.y_rel):
        hits.append(counts[r, orl].argmax() == yr); counts[r, orl, yr] += 1
    return np.array(hits, float)

def pop_prior(train):
    t = np.zeros((3, 3, 3))
    np.add.at(t, (train.res1.values, train.o_rel.values, train.y_rel.values), 1)
    return t / t.sum(axis=2, keepdims=True).clip(min=1)  # conditional probabilities

rows = {k: [] for k in ["uniform", "warm w=10", "warm w=30", "warm w=100", "global DT (2-round feats)", "always counter-stay"]}
d2 = d1.dropna(subset=["c2"]).copy(); d2["c2"] = d2.c2.astype(int); d2["prev_shift"] = (d2.c1 - d2.c2) % 3
ids = d1.id.unique()
for pid in ids:
    tr1, te1 = d1[d1.id != pid], d1[d1.id == pid]
    pr = pop_prior(tr1)
    rows["uniform"].append(personal(te1, None, 0)[:49])
    for w in (10, 30, 100):
        rows[f"warm w={w}"].append(personal(te1, pr, w)[:49])
    rows["always counter-stay"].append((te1.y_rel == 0).values[:49].astype(float))
for k in range(0, len(ids), 12):  # global DT, 6 folds of 12 players
    test_ids = ids[k:k + 12]
    tr, te = d2[~d2.id.isin(test_ids)], d2[d2.id.isin(test_ids)]
    feats = ["res1", "o_rel", "prev_shift", "res2"]
    m = make_pipeline(OneHotEncoder(handle_unknown="ignore"), DecisionTreeClassifier(max_depth=6, random_state=0))
    m.fit(tr[feats].astype(str), tr.y_rel)
    for pid in test_ids:
        x = te[te.id == pid]
        h = (m.predict(x[feats].astype(str)) == x.y_rel.values).astype(float)
        rows["global DT (2-round feats)"].append(np.r_[np.nan, h][:49])  # round 2 has no 2-round features
for name, arrs in rows.items():
    A = np.array(arrs)
    # column j = predicting round j+2 (round 1 has no history)
    print(f"{name:28s} rounds 2-10: {np.nanmean(A[:, :9]):.3f}  rounds 2-20: {np.nanmean(A[:, :19]):.3f}  rounds 2-50: {np.nanmean(A[:, :49]):.3f}")
after_loss = d1[d1.res1 == 1]
print("rule 'after a loss, predict repeat' accuracy on post-loss rounds:", round((after_loss.y_rel == 0).mean(), 3), "n=", len(after_loss))
