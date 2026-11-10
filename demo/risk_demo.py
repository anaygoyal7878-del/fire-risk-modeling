"""Runnable demo of the fire risk pipeline on fully synthetic data.

No real records are used. It builds fake buildings and fire years, makes
history features from earlier years only, trains on the past, tests on the
future, and compares the model with simple baselines.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


def make_data(n=20000, years=range(2010, 2026), seed=0):
    rng = np.random.default_rng(seed)
    age = rng.gamma(4, 7, n)
    size = rng.lognormal(7.5, 0.6, n)
    use = rng.integers(0, 6, n)
    vuln = rng.uniform(0, 1, n)
    hydrant = rng.gamma(2, 400, n)
    base = -8.0 + 0.03 * age + 0.00015 * size + 0.8 * (use == 3) + 1.5 * vuln + 0.0005 * hydrant
    rows = []
    for y in years:
        p = 1 / (1 + np.exp(-(base + rng.normal(0, 0.3, n))))
        rows.append(pd.DataFrame({"id": np.arange(n), "year": y, "fire": rng.random(n) < p}))
    fires = pd.concat(rows)
    static = pd.DataFrame({"id": np.arange(n), "age": age, "size": size, "use": use, "vuln": vuln, "hydrant": hydrant})
    return static, fires


def frame(static, fires, year, window=10):
    """One row per building for `year`. History uses only fires BEFORE `year`."""
    past = fires[(fires.year < year) & (fires.year >= year - window)]
    hist = past.groupby("id").fire.sum().rename("hist_fires")
    df = static.merge(hist, on="id", how="left").fillna({"hist_fires": 0})
    label = fires[fires.year == year].set_index("id").fire
    df["y"] = df.id.map(label).astype(int)
    return df


FEATURES = ["age", "size", "use", "vuln", "hydrant", "hist_fires"]


def top_share(y, s, frac):
    k = max(1, int(len(y) * frac))
    order = np.argsort(-s)[:k]
    return y[order].sum() / max(1, y.sum())


def run(seed=0):
    static, fires = make_data(seed=seed)
    train = pd.concat([frame(static, fires, y) for y in range(2012, 2023)])
    test = pd.concat([frame(static, fires, y) for y in (2024, 2025)])
    model = HistGradientBoostingClassifier(class_weight="balanced", max_depth=4, learning_rate=0.06, max_iter=250, random_state=seed)
    model.fit(train[FEATURES], train.y)
    s = model.predict_proba(test[FEATURES])[:, 1]
    y = test.y.values
    out = {
        "model_auc": roc_auc_score(y, s),
        "size_baseline_auc": roc_auc_score(y, test["size"].values),
        "model_avg_precision": average_precision_score(y, s),
        "model_top5": top_share(y, s, 0.05),
        "size_top5": top_share(y, test["size"].values, 0.05),
        "positives": int(y.sum()),
    }
    return out


if __name__ == "__main__":
    for k, v in run().items():
        print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
