"""Quick feasibility check: can lab algorithms predict real humans' next RPS move?"""
import sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv(sys.argv[1]).sort_values(["id", "game", "period"])
print("players", df.id.nunique(), "| games", df.game.nunique(), "| rows/player", df.groupby("id").size().describe()[["min", "max"]].tolist())
print("choice share", df.choice.value_counts(normalize=True).round(3).to_dict())

R = {"r": 0, "p": 1, "s": 2}
df["c"] = df.choice.map(R); df["o"] = df["other.choice"].map(R)
g = df.groupby(["id", "game"])
for k in (1, 2):
    df[f"c{k}"] = g.c.shift(k); df[f"o{k}"] = g.o.shift(k); df[f"res{k}"] = g.result.shift(k)
d = df.dropna(subset=["c2"]).copy()
for col in ["c1", "o1", "c2", "o2"]:
    d[col] = d[col].astype(int)
# relative coding: 0 = repeat, 1 = shift to the move that beats your last, 2 = shift the other way
d["y_rel"] = (d.c - d.c1) % 3
d["o_rel"] = (d.o1 - d.c1) % 3
d["prev_shift"] = (d.c1 - d.c2) % 3
stay = d.groupby("res1").apply(lambda x: (x.y_rel.value_counts(normalize=True).sort_index().round(3)).to_dict())
print("next move vs last own move, by last result (0 stay, 1 up, 2 down):", stay.to_dict())

sets = {
    "absolute (last 2 rounds)": (["c1", "o1", "res1", "c2", "o2", "res2"], "c"),
    "relative (shifts)": (["res1", "o_rel", "prev_shift", "res2"], "y_rel"),
}
models = {
    "Random/most-frequent": DummyClassifier(strategy="most_frequent"),
    "Naive Bayes": CategoricalNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=0),
    "Logistic Reg.": LogisticRegression(max_iter=1000),
    "Perceptron": Perceptron(random_state=0),
    "KNN": KNeighborsClassifier(50),
    "SVM (rbf)": SVC(),
}
cv = GroupKFold(5)
for name, (feats, target) in sets.items():
    X = d[feats].astype(str); y = d[target]
    print(f"\n== {name}: n={len(d)} target={target}")
    for mname, m in models.items():
        t = time.time()
        enc = OneHotEncoder(handle_unknown="ignore") if mname != "Naive Bayes" else __import__("sklearn.preprocessing", fromlist=["OrdinalEncoder"]).OrdinalEncoder()
        s = cross_val_score(make_pipeline(enc, m), X, y, cv=cv, groups=d.id)
        print(f"  {mname:22s} acc={s.mean():.3f} ±{s.std():.3f}  ({time.time()-t:.1f}s)")

# personal adaptation: per-player online Markov (counts of shift given last result)
acc = []
for pid, x in d.groupby("id"):
    counts = np.ones((3, 3, 3))  # res(3) x o_rel(3) -> y_rel(3)
    resmap = {"win": 0, "lose": 1, "tie": 2}
    hits = []
    for r, orl, yr in zip(x.res1.map(resmap), x.o_rel, x.y_rel):
        hits.append(counts[r, orl].argmax() == yr)
        counts[r, orl, yr] += 1
    acc.append(hits)
L = min(map(len, acc)); A = np.array([a[:L] for a in acc], float)
for lo, hi in [(0, 50), (50, 150), (150, L)]:
    print(f"online per-player predictor, rounds {lo}-{hi}: acc={A[:, lo:hi].mean():.3f}")
