import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
import cvxportfolio as cvx
from typing import Dict, List
import yfinance as yf
from scipy.stats.mstats import winsorize
from typing import List


def xsec_z(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score by date."""
    m, s = df.mean(1), df.std(1).replace(0, 1)
    return df.sub(m, 0).div(s, 0)

def cs_demean(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional demean by date."""
    return df.sub(df.mean(1), 0)

def make_panel(features: Dict[str, pd.DataFrame], target: pd.DataFrame) -> pd.DataFrame:
    """Wide (date×asset) → long panel with features + target."""
    X = pd.concat(features, axis=1)  # MultiIndex columns: (feat, asset)
    X = X.stack().rename_axis(['date','asset']).reset_index()
    Y = target.stack().rename('y').reset_index()
    Y.columns = ['date','asset','y']
    return X.merge(Y, on=['date','asset']).dropna()

class ReturnsFromDF:
    """Forecaster wrapper for cvxportfolio (date × asset DataFrame)."""
    def __init__(self, df: pd.DataFrame): self.df = df
    def __call__(self, t, h, universe, **k):
        # Robust to missing dates (e.g., holidays) and assets
        if t not in self.df.index:
            return pd.Series(0.0, index=universe, dtype=float)
        return self.df.loc[t].reindex(universe).fillna(0.0)

assets = \
    ['AAPL', 'ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'AMAT', 'AMD', 'AMGN',
    'AMZN', 'APP', 'ARM', 'ASML', 'AVGO', 'AXON', 'AZN', 'BIIB', 'BKNG', 'BKR',
    'CCEP', 'CDNS', 'CDW', 'CEG', 'CHTR', 'CMCSA', 'COST', 'CPRT', 'CRWD', 'CSCO',
    'CSGP', 'CSX', 'CTAS', 'CTSH', 'DASH', 'DDOG', 'DXCM', 'EA', 'EXC', 'FANG',
    'FAST', 'FTNT', 'GEHC', 'GFS', 'GILD', 'GOOG', 'GOOGL', 'HON', 'IDXX', 'INTC',
    'INTU', 'ISRG', 'KDP', 'KHC', 'KLAC', 'LIN', 'LRCX', 'LULU', 'MAR', 'MCHP',
    'MDLZ', 'MELI', 'META', 'MNST', 'MRVL', 'MSFT', 'MSTR', 'MU', 'NFLX', 'NVDA',
    'NXPI', 'ODFL', 'ON', 'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD', 'PEP', 'PLTR',
    'PYPL', 'QCOM', 'REGN', 'ROP', 'ROST', 'SBUX', 'SHOP', 'SNPS', 'TEAM', 'TMUS',
    'TRI', 'TSLA', 'TTD', 'TTWO', 'TXN', 'VRSK', 'VRTX', 'WBD', 'WDAY', 'XEL',
    'ZS']

# Try loading from cached CSV first
try:
    data = pd.read_csv('hf_evolution_data.csv', header=[0,1], index_col=0, parse_dates=True)
    prices = data['Close'].dropna(how='all')
except FileNotFoundError:
    # Prices (Adj Close)
    data = yf.download(assets, start="2015-01-01", group_by='column', progress=False)
    data.to_csv('hf_evolution_data.csv')  # cache for next time

prices = data['Close'].dropna(how='all')

# Example price-only signals:
mom = prices.pct_change(60)             # 3m momentum (approx)
rev = -prices.pct_change(5)             # 1w reversal proxy
qual = -prices.pct_change().rolling(60).std()  # "quality" proxy = low vol

# Map to your earlier variable names (so slides run unchanged)
btp  = (prices.rolling(252).mean() / prices)  # crude "value" proxy
roa  = qual

val  = btp[assets]  # value
qual = roa[assets]  # quality
rev  = rev[assets]  # reversal

# Cross-sectional z-scoring
mom_z, val_z, qual_z, rev_z = map(xsec_z, [mom, val, qual, rev])

# We have daily data; target = forward 5d return, demeaned
# First, rolling cumulative return over next 5 days
r5 = prices[assets].pct_change().rolling(5).sum().shift(-5)
y_cs = cs_demean(r5)

# Index alignment (avoid silent misalignment)
idx = (mom_z.index
  .intersection(val_z.index)
  .intersection(qual_z.index)
  .intersection(rev_z.index)
  .intersection(y_cs.index))

mom_z, val_z, qual_z, rev_z, y_cs = \
  mom_z.loc[idx], val_z.loc[idx], qual_z.loc[idx], rev_z.loc[idx], y_cs.loc[idx]

# Feature dict (single source of truth for names/order)
FEATS = {'mom': mom_z, 'val': val_z, 'qual': qual_z, 'rev': rev_z}

# Winsorize features (5th–95th percentile per date)
for k in FEATS:
    FEATS[k] = FEATS[k].apply(lambda row:
        pd.Series(winsorize(row, limits=[0.05, 0.05]), index=row.index),
        axis=1
    )

panel = make_panel(FEATS, y_cs)
qtiles = panel.copy()

factors = ["mom", "val", "qual", "rev"]

# Apply quintile assignment per day for each factor
for f in factors:
    qtiles[f"{f}_q"] = panel.groupby("date")[f].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") + 1
    )

# Factors with quintiles
q_factors = ["mom_q", "val_q", "qual_q", "rev_q"]

avg_returns = {}
for q in q_factors:
    avg_returns[q] = qtiles.groupby(q)["y"].mean()

# Put into a dataframe for easier comparison
avg_returns_df = pd.DataFrame(avg_returns)

print("Average Returns by Quintile:")
print(avg_returns_df)




def walk_forward_oof_blocks(panel: pd.DataFrame,
                            feature_cols: List[str],
                            assets: List[str],
                            warm: int = 60,
                            inc: int = 5,
                            fill_missing: float | None = None) -> pd.DataFrame:
    """
    Expanding fit with block predictions:
      - Sort unique dates.
      - For k = warm, warm+inc, warm+2*inc, ...:
          * Train on all rows with date < dates[k]  (expanding)
          * Predict for dates in dates[k : k+inc]   (block), strictly OOS
      - Returns alpha[date, asset] with predictions; leaves others NaN unless fill_missing is set.

    Args:
        panel: long panel with columns ['date','asset', *feature_cols, 'y'] (y not used at predict time).
        feature_cols: features to fit on.
        assets: full list of assets (columns of output).
        warm: number of *distinct* dates to include before first prediction block.
        inc: block size and refit cadence (must be >= 1).
        fill_missing: if not None, fill remaining NaNs with this value (e.g., 0.0).

    """
    if inc < 1:
        raise ValueError("inc must be >= 1")

    # Ensure dates are sorted and comparable
    dates = np.sort(panel['date'].unique())
    n = len(dates)

    alpha = pd.DataFrame(index=dates, columns=assets, dtype=float)

    if n <= max(warm, 1):
        return alpha.fillna(fill_missing) if fill_missing is not None else alpha

    model = LinearRegression()

    # k is the *start index* of each prediction block
    for k in range(warm, n, inc):
        t_start = dates[k]
        t_end_idx = min(k + inc, n)  # exclusive end index for prediction block

        # Expanding train set: all rows with date < t_start
        train = panel[panel['date'] < t_start]
        if train.empty:
            continue

        model.fit(train[feature_cols], train['y'])

        # Predict for the next inc dates: dates[k : t_end_idx]
        pred_block_dates = dates[k:t_end_idx]
        block = panel[panel['date'].isin(pred_block_dates)]

        if block.empty:
            continue

        # Predict and write into alpha for each date in the block
        preds = model.predict(block[feature_cols])
        tmp = block[['date', 'asset']].copy()
        tmp['pred'] = preds

        # Pivot to match alpha layout (date index, asset columns)
        block_alpha = tmp.pivot(index='date', columns='asset', values='pred')

        # Align to full asset list (keep only requested assets as columns)
        block_alpha = block_alpha.reindex(index=pred_block_dates, columns=assets)

        # Write into result
        alpha.loc[pred_block_dates, assets] = block_alpha.values

    if fill_missing is not None:
        alpha = alpha.fillna(fill_missing)

    return alpha

alpha = walk_forward_oof_blocks(panel, list(FEATS.keys()), assets, warm=60)

# Stabilize: winsorize (5th–95th percentile per date)
alpha = alpha.apply(lambda row: 
    pd.Series(winsorize(row, limits=[0.05, 0.05]), index=row.index),
    axis=1
)

# alpha is now ready for use in cvxportfolio
# the shape of alpha is (dates, assets)

rf = ReturnsFromDF(alpha)
gamma = 3.0
kappa = 0.05

obj = (cvx.ReturnsForecast(rf)
  - gamma * (cvx.FullCovariance() + kappa * cvx.RiskForecastError())
  - cvx.StocksTransactionCost()
)

constraints = [cvx.LeverageLimit(1), cvx.LongOnly(), cvx.NoCash(), cvx.MaxWeights(0.33)]
#policy = cvx.MultiPeriodOptimization(obj, constraints, planning_horizon=2)
policy = cvx.SinglePeriodOptimization(obj, constraints)

start = str(alpha.index.min().date()) if len(alpha.index) else '2020-01-01'
sim = cvx.StockMarketSimulator(assets)
result = sim.backtest(policy, start_time=start)

