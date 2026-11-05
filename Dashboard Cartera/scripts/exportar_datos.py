"""Actualiza la instantánea del dashboard desde el CSV del proyecto."""
import ast
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parents[1] / 'datos'
DEST.mkdir(exist_ok=True, parents=True)
source = ROOT / 'Datos/Transformados/dowjones_diario_2016_2025.csv'
df = pd.read_csv(source, parse_dates=['date'])
assert not df.duplicated(['symbol', 'date']).any()
assert df['close'].gt(0).all() and df[['date', 'symbol', 'close']].notna().all().all()
prices = df.pivot(index='date', columns='symbol', values='close').sort_index()
returns = np.log(prices / prices.shift(1)).dropna(how='all')
notebook = json.loads((ROOT / '04_Seleccion_cartera.ipynb').read_text(encoding='utf-8'))
constants = {}
for cell in notebook['cells']:
    if cell['cell_type'] != 'code':
        continue
    for node in ast.parse(''.join(cell['source'])).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ['sector', 'excluidas']:
                    constants[target.id] = ast.literal_eval(node.value)
sector, excluded = constants['sector'], constants['excluidas']
mu, sigma = returns.mean() * 252, returns.std() * np.sqrt(252)
sharpe = (mu - .01) / sigma
corr = returns.corr()
candidates = [s for s in prices if s not in excluded and returns[s].count() == returns.count().max() and mu[s] > 0 and sharpe[s] > 0]
ranking = []
for trio in itertools.combinations(candidates, 3):
    pairs = [abs(corr.loc[a, b]) for a, b in itertools.combinations(trio, 2)]
    if len({sector[s] for s in trio}) != 3 or max(pairs) >= .6:
        continue
    mean_sharpe = float(sharpe[list(trio)].mean())
    ranking.append(dict(tickers=list(trio), sharpe=mean_sharpe, correlation=float(np.mean(pairs)), maxCorrelation=float(max(pairs)), score=mean_sharpe-float(np.mean(pairs))))
ranking.sort(key=lambda r: r['score'], reverse=True)
assert ranking[0]['tickers'] == ['CAT', 'MRK', 'MSFT'] and len(ranking) == 295
chosen = ['CAT', 'MRK', 'MSFT']
parameters = json.loads((ROOT / 'outputs/parametros_montecarlo.json').read_text())
assert np.allclose(sigma[chosen], np.array(parameters['sigma']).ravel(), atol=0.000001)
assert np.allclose(corr.loc[chosen, chosen], parameters['rho'], atol=0.000001)
cov = returns[chosen].cov().to_numpy() * 252
rng = np.random.default_rng(42)
weights = rng.dirichlet(np.ones(3), size=10000)
annual = weights @ mu[chosen].to_numpy()
vol = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov, weights))
ratios = (annual-.01)/vol
frontier = np.column_stack([vol, annual, ratios])
best, safest = int(np.argmax(ratios)), int(np.argmin(vol))
assets = [dict(ticker=s, sector=sector[s], mu=float(mu[s]), sigma=float(sigma[s]), sharpe=float(sharpe[s]), sessions=int(returns[s].count()), eligible=s in candidates, exclusion=excluded.get(s, ''), complete=bool(returns[s].count()==returns.count().max())) for s in prices]
payload = dict(meta=dict(start=str(prices.index.min().date()), end=str(prices.index.max().date()), rows=len(df), sessions=len(prices), candidates=len(candidates), source=source.name, sha256=hashlib.sha256(source.read_bytes()).hexdigest(), annualDays=252, riskFree=.01), assets=assets, dates=prices.index.strftime('%Y-%m-%d').tolist(), prices={s:[None if pd.isna(v) else round(float(v),6) for v in prices[s]] for s in prices}, correlations=corr.round(9).values.tolist(), ranking=ranking, frontier=frontier.round(8).tolist(), presets=dict(equal=[1/3]*3, maxSharpe=weights[best].tolist(), minRisk=weights[safest].tolist()))
(DEST/'proyecto.json').write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')), encoding='utf-8')
print(f'{len(df)} filas; {len(candidates)} elegibles; {len(ranking)} ternas; score ganador {ranking[0]["score"]:.9f}')
print('Volatilidad:', sigma[chosen].round(6).to_dict())
print('JSON:', (DEST/'proyecto.json').stat().st_size, 'bytes')
