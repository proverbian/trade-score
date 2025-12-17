# scoring.py
import pandas as pd
import numpy as np

def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def pair_momentum_score(df: pd.DataFrame, short=5, long=20) -> float:
    close = df['Close']
    e_short = ema(close, short)
    e_long = ema(close, long)

    diff = e_short - e_long
    val = diff.iloc[-1]

    rng = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
    if rng == 0 or np.isnan(rng):
        return 0.0

    score = val / rng

    # Intraday clamp (tight)
    score = max(min(score, 1.5), -1.5)
    return float(score)

def build_currency_strength(pair_scores: dict):
    """
    pair_scores: dict(pair -> score)
    returns: dict(currency -> aggregated score)
    Logic: each pair contributes +score to base, -score to quote.
    """
    strength = {}
    for pair, score in pair_scores.items():
        base = pair[:3]
        quote = pair[3:]
        strength.setdefault(base, 0.0)
        strength.setdefault(quote, 0.0)
        strength[base] += score
        strength[quote] -= score
    return strength

def normalize_strength(strength: dict):
    # scale so values are comparable; map to integers like -6..+6 for display
    vals = np.array(list(strength.values()), dtype=float)
    if vals.std() == 0:
        return {k: 0 for k in strength}
    scaled = (vals - vals.mean()) / (vals.std())
    # map to -6..6
    scaled = np.clip(np.round(scaled * 2), -6, 6).astype(int)
    return {k: int(v) for k, v in zip(strength.keys(), scaled)}
