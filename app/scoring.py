# scoring.py
import pandas as pd
import numpy as np

def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()


def pair_performance_score(df: pd.DataFrame, lookback_bars: int = 96) -> float:
    """MarketMilk-style pair performance.

    Computes percent change over a lookback window.

    Returns a signed float in *percent* (e.g., +0.25 means +0.25%).
    """
    if df is None or df.empty:
        return 0.0

    close = df.get('Close')
    if close is None or len(close) < 2:
        return 0.0

    lb = int(lookback_bars) if lookback_bars is not None else 1
    if lb < 1:
        lb = 1

    # If insufficient bars, fall back to first available close.
    idx0 = max(0, len(close) - 1 - lb)
    c0 = float(close.iloc[idx0])
    c1 = float(close.iloc[-1])
    if c0 == 0 or np.isnan(c0) or np.isnan(c1):
        return 0.0

    return float((c1 / c0 - 1.0) * 100.0)

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


def find_strongest_weakest(strength: dict):
    """Return (strongest_currency, weakest_currency) based on strength values."""
    if not strength:
        return None, None
    items = sorted(strength.items(), key=lambda kv: kv[1])
    weakest = items[0][0]
    strongest = items[-1][0]
    return strongest, weakest


def pair_gap_from_strength(pair: str, strength: dict) -> float:
    """Compute base-minus-quote gap for a given pair using a currency strength map."""
    if not pair or len(pair) != 6 or not strength:
        return 0.0
    base, quote = pair[:3], pair[3:]
    return float(strength.get(base, 0.0) - strength.get(quote, 0.0))


def pick_pair_from_extremes(strongest: str, weakest: str, available_pairs: list[str]):
    """Pick a tradable pair that expresses strongest vs weakest.

    If strongest+weakest exists (e.g., USDJPY), bias is BUY.
    If weakest+strongest exists (e.g., JPYUSD), bias is SELL on that symbol.

    Returns: (pair, bias) or (None, None)
    """
    if not strongest or not weakest or not available_pairs:
        return None, None

    direct = f"{strongest}{weakest}"
    inverse = f"{weakest}{strongest}"
    if direct in available_pairs:
        return direct, 'BUY'
    if inverse in available_pairs:
        return inverse, 'SELL'
    return None, None

def normalize_strength(strength: dict):
    # scale so values are comparable; map to integers like -6..+6 for display
    vals = np.array(list(strength.values()), dtype=float)
    if vals.std() == 0:
        return {k: 0 for k in strength}
    scaled = (vals - vals.mean()) / (vals.std())
    # map to -6..6
    scaled = np.clip(np.round(scaled * 2), -6, 6).astype(int)
    return {k: int(v) for k, v in zip(strength.keys(), scaled)}
