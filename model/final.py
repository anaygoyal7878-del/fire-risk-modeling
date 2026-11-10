import sys; sys.argv = ['x', 'data']
exec(open('model/train.py').read().split('results = {}')[0])
import json
PRM = dict(learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=40, max_iter=300, l2_regularization=1.0, random_state=0)
cm = [c in CAT for c in FEATS]

# 1) evaluated model: train 2010 to 2020, test 2024 to 2025
ev = HistGradientBoostingClassifier(categorical_features=cm, **PRM).fit(Xtr[FEATS][pop_tr], ytr[pop_tr])
s_te = ev.predict_proba(Xte[FEATS][pop_te])[:, 1]
s_va = ev.predict_proba(Xva[FEATS][pop_va])[:, 1]
EVAL = {'test': score(yte[pop_te], s_te), 'val': score(yva[pop_va], s_va)}
# tier hit rates on the test years
pct_te = pd.Series(s_te).rank(pct=True).values
tiers = [('Very high', 0.98), ('High', 0.90), ('Elevated', 0.75), ('Lower', 0.0)]
def tier_of(p):
    for name, cut in tiers:
        if p >= cut: return name
yt = yte[pop_te]; nt = B.n_test.fillna(0).values[pop_te]
tier_te = np.array([tier_of(p) for p in pct_te])
EVAL['tiers_test'] = {t: {'buildings': int((tier_te == t).sum()), 'buildings_with_fire': int(yt[tier_te == t].sum()),
                          'share_of_fires': round(float(nt[tier_te == t].sum() / nt.sum()), 3)} for t, _ in tiers}
from sklearn.inspection import permutation_importance
base = {
    'Random order': rng.random(pop_te.sum()),
    'Oldest buildings first': -B.year_built.fillna(2026).values[pop_te],
    'Biggest buildings first': B.sqft.fillna(0).values[pop_te],
    'Most past fires nearby first': Xte.hist_fires_half_mile.fillna(0).values[pop_te] + B.sqft.fillna(0).values[pop_te] / 1e9,
}
EVAL['baselines'] = {k: score(yte[pop_te], v) for k, v in base.items()}
EVAL['n'] = {'train_buildings': int(pop_tr.sum()), 'train_fire_buildings': int(ytr[pop_tr].sum()), 'test_buildings': int(pop_te.sum()), 'test_fire_buildings': int(yt.sum()), 'test_fires': int(nt.sum())}
pi = permutation_importance(ev, Xte[FEATS][pop_te], yt, scoring='roc_auc', n_repeats=8, random_state=0)
EVAL['importance'] = sorted([(k, round(float(v), 4)) for k, v in zip(FEATS, pi.importances_mean)], key=lambda t: -t[1])
print(json.dumps({k: v for k, v in EVAL.items() if k != 'importance'}, indent=1)); print(EVAL['importance'][:15])

# 2) final model: all labeled periods stacked, then score 2026 to 2027 with history 2016 to 2025
Xs = pd.concat([Xtr[FEATS][pop_tr], Xva[FEATS][pop_va], Xte[FEATS][pop_te]])
ys = np.r_[ytr[pop_tr], yva[pop_va], yte[pop_te]]
fin = HistGradientBoostingClassifier(categorical_features=cm, **PRM).fit(Xs, ys)
Xnow = frame(2027, 2016, 2025)
for c in CAT: Xnow[c] = pd.Categorical(Xnow[c], categories=Xtr[c].cat.categories)
s_now = fin.predict_proba(Xnow[FEATS])[:, 1]
pct = pd.Series(s_now).rank(pct=True).values

# 3) plain language reasons: swap each factor to the typical value and see how much risk drops
REASONS = {
 'age': ('Older building', lambda X: X.assign(age=np.nanmedian(X.age))),
 'size': ('Large building', lambda X: X.assign(log_sqft=np.nanmedian(X.log_sqft), units=0)),
 'use': ('Higher risk use or building type', lambda X: X.assign(usecode=X.usecode.mode()[0], btype=X.btype.mode()[0], rcp=X.rcp.mode()[0])),
 'value': ('Low property value', lambda X: X.assign(log_mkt=np.nanmedian(X.log_mkt), log_bval=np.nanmedian(X.log_bval), log_lval=np.nanmedian(X.log_lval))),
 'owner': ('Owner does not live here', lambda X: X.assign(homestead=1, owner_type=X.owner_type.mode()[0], mail=X.mail.mode()[0])),
 'history': ('Past fires here or nearby', lambda X: X.assign(**{c: 0 for c in HIST})),
 'access': ('Far from a fire station', lambda X: X.assign(mi_station=np.nanmedian(X.mi_station), drive_min=np.nanmedian(X.drive_min), ft_hydrant=np.nanmedian(X.ft_hydrant))),
 'hazard': ('Near hazardous chemicals or hazard routes', lambda X: X.assign(ft_tier2=np.nanmedian(X.ft_tier2), hazmat_hwy=0, rail_buf=0, gas_buf=0)),
 'area': ('Higher need neighborhood', lambda X: X.assign(svi=np.nanmedian(X.svi))),
 'density': ('Crowded with nearby buildings', lambda X: X.assign(n300=np.nanmedian(X.n300), n1000=np.nanmedian(X.n1000), ft_nearest_bldg=np.nanmedian(X.ft_nearest_bldg))),
 'veg': ('Vegetation nearby', lambda X: X.assign(tree_pct=0, grass_pct=0, landcover=X.landcover.mode()[0])),
}
base_s = s_now
drops = {}
for k, (lab, fn) in REASONS.items():
    Xm = fn(Xnow[FEATS].copy())
    for c in CAT: Xm[c] = pd.Categorical(Xm[c], categories=Xtr[c].cat.categories)
    drops[k] = base_s - fin.predict_proba(Xm[FEATS])[:, 1]
Dm = pd.DataFrame(drops)
rel = Dm.div(base_s, axis=0)
reasons = []
for i in range(len(B)):
    r = rel.iloc[i]; r = r[r > 0.08].sort_values(ascending=False)[:3]
    reasons.append('; '.join(REASONS[k][0] for k in r.index))

out = B[['pid', 'address', 'lat', 'lon', 'district', 'btype', 'rcp', 'bname', 'year_built', 'sqft', 'units', 'drive_min', 'station', 'ft_hydrant']].copy()
out['score'] = s_now; out['percentile'] = np.round(pct * 100, 1)
out['tier'] = [tier_of(p) for p in pct]
out['reasons'] = reasons
out['fires_2010_2025'] = (B.n_train.fillna(0) + B.n_val.fillna(0) + B.n_test.fillna(0)).astype(int)
out.to_csv(f'{D}/buildings_scored.csv', index=False)
Dm.insert(0, "pid", B.pid.values); Dm["score"] = base_s; Dm.to_csv(f"{D}/factor_drops.csv", index=False)
json.dump(EVAL, open(f'{D}/eval.json', 'w'), indent=1)
print(out.tier.value_counts().to_dict()); print(out.sort_values('score', ascending=False).head(15)[['address', 'btype', 'bname', 'percentile', 'reasons']].to_string())
