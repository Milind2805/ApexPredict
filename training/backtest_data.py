import json
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, 'f1_multiyear.csv')
OUT = os.path.join(ROOT, 'frontend', 'backtest.json')
SEASON = 'Year'

df = pd.read_csv(CSV)
df['Winner'] = (df['Position'] == 1).astype(int)
df = df.dropna(subset=['GridPosition', 'QualiPosition', 'Rainfall', 'AirTemp',
                       'TrackTemp', 'StandingPoints', 'HomeRace'])
df['race_id'] = df[SEASON].astype(str) + '|' + df['Race']
df['Driver'] = LabelEncoder().fit_transform(df['Abbreviation'])
df['Team'] = LabelEncoder().fit_transform(df['TeamName'])
df['Circuit'] = LabelEncoder().fit_transform(df['Race'])

FEATURES = ['Driver', 'Team', 'Circuit', 'GridPosition', 'QualiPosition',
            'Rainfall', 'AirTemp', 'TrackTemp', 'StandingPoints', 'HomeRace']

test_year = int(df[SEASON].max())
train = df[df[SEASON] < test_year]
test = df[df[SEASON] == test_year].copy()

# same setup as the deployed model: SMOTE on the training seasons only
X, y = SMOTE(random_state=42).fit_resample(train[FEATURES], train['Winner'])
model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X, y)
test['p'] = model.predict_proba(test[FEATURES])[:, 1]

races = []
for _, g in test.groupby('race_id', sort=False):      # CSV order = race order
    if g['Winner'].sum() != 1:
        continue
    g = g.sort_values('p', ascending=False)
    winner = g[g['Winner'] == 1].iloc[0]
    pole = g.loc[g['QualiPosition'].idxmin()]
    pick = g.iloc[0]
    rank = list(g['Abbreviation']).index(winner['Abbreviation']) + 1
    races.append({
        'race': str(g['Race'].iloc[0]),
        'model_pick': str(pick['Abbreviation']),
        'winner': str(winner['Abbreviation']),
        'pole': str(pole['Abbreviation']),
        'winner_rank': int(rank),
        'model_ok': bool(pick['Abbreviation'] == winner['Abbreviation']),
        'pole_ok': bool(pole['Abbreviation'] == winner['Abbreviation']),
    })

summary = {
    'races': len(races),
    'model_hits': sum(r['model_ok'] for r in races),
    'pole_hits': sum(r['pole_ok'] for r in races),
    'top3_hits': sum(r['winner_rank'] <= 3 for r in races),
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump({'season': test_year, 'summary': summary, 'races': races}, f, ensure_ascii=False, indent=1)
print(summary)
print('saved', OUT)