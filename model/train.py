import pandas as pd, numpy as np, sys, json
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
D = sys.argv[1]
B = pd.read_pickle(f'{D}/B.pkl'); H = pd.read_pickle(f'{D}/H.pkl')
rng = np.random.default_rng(7)

def xy(lat, lon):
    return np.c_[lon * np.cos(np.radians(33.2)) * 111320, lat * 110574]

has = B.lat.notna() & B.lon.notna()
P = xy(B.lat.fillna(0).values, B.lon.fillna(0).values)
pid_index = {p: i for i, p in enumerate(B.pid)}

def history(y0, y1):
    """fire history features using only fires in [y0, y1]"""
    e = H[(H.year >= y0) & (H.year <= y1)]
    at = np.zeros(len(B)); at_s = np.zeros(len(B))
    for p, s in zip(e.pid, e.structure):
        i = pid_index.get(p)
        if i is not None:
            at[i] += 1; at_s[i] += s
    eg = e[e.lat.notna()]
    out = {'hist_fires_here': at, 'hist_struct_here': at_s}
    if len(eg):
        T = cKDTree(xy(eg.lat.values, eg.lon.values))
        st = eg.structure.values
        for r, name in [(305, '1000ft'), (805, 'half_mile')]:
            nb = T.query_ball_point(P, r)
            out['hist_fires_' + name] = np.array([len(n) for n in nb], float)
            out['hist_struct_' + name] = np.array([st[n].sum() if n else 0 for n in nb], float)
    for k in out:
        out[k][~has.values] = np.nan if 'here' not in k else out[k][~has.values]
    return pd.DataFrame(out)

CAT = ['zip', 'district', 'btype', 'rcp', 'usecode', 'owner_type', 'mail', 'landcover']
NUM = ['age', 'log_sqft', 'units', 'log_mkt', 'log_bval', 'log_lval', 'acres', 'pool', 'homestead', 'alcohol_ever',
       'mi_station', 'drive_min', 'ft_hydrant', 'ft_nearest_bldg', 'n300', 'n1000', 'tree_pct', 'grass_pct',
       'hazmat_hwy', 'rail_buf', 'gas_buf', 'ft_tier2', 'svi', 'lat', 'lon']
HIST = ['hist_fires_here', 'hist_struct_here', 'hist_fires_1000ft', 'hist_struct_1000ft', 'hist_fires_half_mile', 'hist_struct_half_mile']

def frame(end_year, hist_from, hist_to):
    X = B[CAT + [c for c in NUM if c != 'age']].copy()
    X['age'] = end_year - B.year_built
    hf = history(hist_from, hist_to)
    for c in HIST: X[c] = hf[c].values
    return X

# periods: history window = the 10 years before the label period starts
Xtr = frame(2020, 2000, 2009); Xva = frame(2023, 2013, 2022); Xte = frame(2025, 2014, 2023)
pop_tr = (B.year_built.fillna(0) <= 2019).values
pop_va = (B.year_built.fillna(0) <= 2022).values
pop_te = (B.year_built.fillna(0) <= 2024).values
ytr, yva, yte = B.y_train.values, B.y_val.values, B.y_test.values
FEATS = CAT + NUM + HIST

def hgb(params):
    Xc = Xtr.copy()
    cat_mask = [c in CAT for c in FEATS]
    m = HistGradientBoostingClassifier(categorical_features=cat_mask, class_weight='balanced', random_state=0, **params)
    return m, cat_mask

def enc(X):
    X = X[FEATS].copy()
    for c in CAT: X[c] = X[c].astype('category')
    return X

# consistent category levels
for c in CAT:
    cats = pd.Categorical(B[c]).categories
    for X in (Xtr, Xva, Xte): X[c] = pd.Categorical(X[c], categories=cats)

def capture(y, s, pcts=(0.01, 0.05, 0.10, 0.20)):
    o = np.argsort(-s); n = len(s); tot = y.sum()
    return {f'top{int(p*100)}pct': round(float(y[o[:int(n*p)]].sum() / tot), 3) for p in pcts}

def score(y, s):
    r = {'auc': round(float(roc_auc_score(y, s)), 3), 'avg_precision': round(float(average_precision_score(y, s)), 4)}
    r.update(capture(y, s)); return r

results = {}
# baselines on test
te_idx = pop_te
base = {
    'Random order': rng.random(te_idx.sum()),
    'Oldest buildings first': -B.year_built.fillna(2026).values[te_idx],
    'Biggest buildings first': B.sqft.fillna(0).values[te_idx],
    'Past fires nearby first (half mile, 10 yrs)': Xte.hist_fires_half_mile.fillna(0).values[te_idx] + 0.001 * B.sqft.fillna(0).values[te_idx] / 1e6,
}
for k, s in base.items(): results['baseline: ' + k] = score(yte[te_idx], s)

# tune HGB on validation
best = None
for lr in [0.03, 0.06]:
    for leaf in [15, 31]:
        for l2 in [1.0, 10.0]:
            p = dict(learning_rate=lr, max_leaf_nodes=leaf, l2_regularization=l2, max_iter=400, min_samples_leaf=40,
                     early_stopping=True, validation_fraction=0.15, n_iter_no_change=30)
            m = HistGradientBoostingClassifier(categorical_features=[c in CAT for c in FEATS], class_weight='balanced', random_state=0, **p)
            m.fit(Xtr[FEATS][pop_tr], ytr[pop_tr])
            s = m.predict_proba(Xva[FEATS][pop_va])[:, 1]
            ap = average_precision_score(yva[pop_va], s); auc = roc_auc_score(yva[pop_va], s)
            key = auc + 5 * ap
            if best is None or key > best[0]: best = (key, p, auc, ap)
print('best val', best)
P_BEST = best[1]
m = HistGradientBoostingClassifier(categorical_features=[c in CAT for c in FEATS], class_weight='balanced', random_state=0, **P_BEST)
m.fit(Xtr[FEATS][pop_tr], ytr[pop_tr])
results['gradient boosting'] = score(yte[te_idx], m.predict_proba(Xte[FEATS][te_idx])[:, 1])
results['gradient boosting (validation 2023)'] = score(yva[pop_va], m.predict_proba(Xva[FEATS][pop_va])[:, 1])

# logistic regression (explainable)
pre = ColumnTransformer([('c', OneHotEncoder(handle_unknown='ignore', min_frequency=30), CAT),
                         ('n', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), NUM + HIST)])
lr = make_pipeline(pre, LogisticRegression(C=0.3, class_weight='balanced', max_iter=3000))
lr.fit(Xtr[FEATS][pop_tr].astype({c: str for c in CAT}), ytr[pop_tr])
s_lr = lr.predict_proba(Xte[FEATS][te_idx].astype({c: str for c in CAT}))[:, 1]
results['logistic regression'] = score(yte[te_idx], s_lr)
s_lr_va = lr.predict_proba(Xva[FEATS][pop_va].astype({c: str for c in CAT}))[:, 1]
results['logistic regression (validation 2023)'] = score(yva[pop_va], s_lr_va)

# permutation importance on test (grouped, AUC drop) for gradient boosting
from sklearn.inspection import permutation_importance
Xt = Xte[FEATS][te_idx]
pi = permutation_importance(m, Xt, yte[te_idx], scoring='roc_auc', n_repeats=5, random_state=0)
imp = sorted(zip(FEATS, pi.importances_mean), key=lambda t: -t[1])
results['importance'] = [(k, round(float(v), 4)) for k, v in imp]
results['n'] = {'train_buildings': int(pop_tr.sum()), 'train_pos': int(ytr[pop_tr].sum()), 'val_buildings': int(pop_va.sum()), 'val_pos': int(yva[pop_va].sum()), 'test_buildings': int(te_idx.sum()), 'test_pos': int(yte[te_idx].sum())}
results['best_params'] = P_BEST
json.dump(results, open(f'{D}/results.json', 'w'), indent=1)
for k, v in results.items():
    if k != 'importance': print(k, v)
print(results['importance'][:20])
