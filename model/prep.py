import pandas as pd, numpy as np, re, sys
D = sys.argv[1]
b = pd.read_csv(f'{D}/buildings.csv', header=2, dtype=str)
f = pd.read_csv(f'{D}/features.csv', header=2, dtype=str)
h = pd.read_csv(f'{D}/firehistory.csv', header=2, dtype=str)
assert (b.iloc[:, 0].values == f.iloc[:, 0].values).all()

def num(s):
    return pd.to_numeric(s.astype(str).str.replace(r'[$,%]', '', regex=True).replace({'nan': np.nan, '': np.nan}), errors='coerce')

B = pd.DataFrame({'pid': b.iloc[:, 0]})
B['address'] = b.iloc[:, 2]; B['zip'] = b.iloc[:, 3].fillna('NA'); B['district'] = b.iloc[:, 4].fillna('NA')
B['btype'] = b.iloc[:, 5].fillna('NA'); B['rcp'] = b.iloc[:, 6].fillna('NA'); B['bname'] = b.iloc[:, 7].fillna('')
uc = b.iloc[:, 8].fillna('NA'); top = uc.value_counts().index[:30]; B['usecode'] = uc.where(uc.isin(top), 'OTHER')
B['year_built'] = num(b.iloc[:, 10]); B['sqft'] = num(b.iloc[:, 11]); B['units'] = num(b.iloc[:, 12]).fillna(0)
B['mkt'] = num(b.iloc[:, 13]); B['bval'] = num(b.iloc[:, 14]); B['lval'] = num(b.iloc[:, 15])
B['mi_station'] = num(b.iloc[:, 24]); B['ft_hydrant'] = num(b.iloc[:, 26])
B['lat'] = num(b.iloc[:, 28]); B['lon'] = num(b.iloc[:, 29])
B['owner_type'] = b.iloc[:, 35].fillna('NA'); B['mail'] = b.iloc[:, 36].fillna('NA'); B['homestead'] = (b.iloc[:, 37] == 'Yes').astype(int)
B['pool'] = (b.iloc[:, 40] == 'Yes').astype(int); B['acres'] = num(b.iloc[:, 41])
B['alcohol_ever'] = num(b.iloc[:, 46]).fillna(0)
B['ft_nearest_bldg'] = num(b.iloc[:, 47]); B['n300'] = num(b.iloc[:, 48]); B['n1000'] = num(b.iloc[:, 49])
B['drive_min'] = num(b.iloc[:, 50]); B['station'] = b.iloc[:, 52].fillna('NA')
B['landcover'] = b.iloc[:, 53].fillna('NA'); B['tree_pct'] = num(b.iloc[:, 54]); B['grass_pct'] = num(b.iloc[:, 55])
B['hazmat_hwy'] = (f.iloc[:, 12].fillna('No') != 'No').astype(int)
B['rail_buf'] = (f.iloc[:, 13] == 'Yes').astype(int); B['gas_buf'] = (f.iloc[:, 14] == 'Yes').astype(int)
B['ft_tier2'] = num(f.iloc[:, 15]); B['svi'] = num(f.iloc[:, 16])
B['y_train'] = (num(f.iloc[:, 17]) > 0).astype(int); B['y_val'] = (num(f.iloc[:, 18]) > 0).astype(int); B['y_test'] = (num(f.iloc[:, 19]) > 0).astype(int)
B['n_train'] = num(f.iloc[:, 17]); B['n_val'] = num(f.iloc[:, 18]); B['n_test'] = num(f.iloc[:, 19])
for c in ['sqft', 'mkt', 'bval', 'lval']:
    B['log_' + c] = np.log1p(B[c].clip(lower=0))

# fire history events with coordinates (matched buildings only)
H = pd.DataFrame({'year': num(h.iloc[:, 0]), 'code': num(h.iloc[:, 4]), 'pid': h.iloc[:, 10], 'date': h.iloc[:, 1], 'type': h.iloc[:, 5]})
H = H.merge(B[['pid', 'lat', 'lon', 'address', 'btype']], on='pid', how='left')
H['structure'] = H.code.between(111, 123).astype(int)
B.to_pickle(f'{D}/B.pkl'); H.to_pickle(f'{D}/H.pkl')
print(B.shape, H.shape, B[['y_train', 'y_val', 'y_test']].sum().to_dict(), H.lat.notna().sum())
