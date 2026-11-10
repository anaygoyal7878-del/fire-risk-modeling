import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "demo"))
import numpy as np
import risk_demo as d


def test_history_never_uses_the_future():
    static, fires = d.make_data(n=500, seed=1)
    df = d.frame(static, fires, 2020, window=10)
    past = fires[(fires.year < 2020) & (fires.year >= 2010)].groupby("id").fire.sum()
    assert (df.set_index("id").hist_fires == past.reindex(df.id).fillna(0).values).all()
    future_only = fires[fires.year >= 2020].groupby("id").fire.sum().reindex(df.id).fillna(0).values
    assert not np.array_equal(df.hist_fires.values, future_only)


def test_model_beats_random_and_size_baseline():
    r = d.run(seed=0)
    assert r["model_auc"] > 0.6
    assert r["model_auc"] > r["size_baseline_auc"]


def test_top_share_bounds():
    y = np.array([1, 0, 0, 1, 0]); s = np.array([.9, .1, .2, .8, .3])
    assert d.top_share(y, s, 0.4) == 1.0
