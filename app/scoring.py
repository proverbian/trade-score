# scoring.py
import pandas as pd
import numpy as np

def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def pair_momentum_score(df: pd.DataFrame, short=10, long=50) -> float:
    """
    Simple momentum: sign of (shortEMA - longEMA) for last close; magnitude scaled to %.
    Returns a float in roughly [-1, +1]
    """
    close = df['Close']
    e_short = ema(close, short)
    e_long = ema(close, long)
    diff = e_short - e_long
    val = diff.iloc[-1]
    # normalize by ATR-like range to avoid blowups
    rng = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
    if rng == 0 or np.isnan(rng):
        return 0.0
    score = float(val / rng)
    # clamp
    score = max(min(score, 3.0), -3.0)
    return score

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
