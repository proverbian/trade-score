# s_r.py
import numpy as np
import pandas as pd


def find_pivots(close: pd.Series, window: int = 2):
    """Return pivot highs/lows with their index.

    Output is a list of dicts sorted by index:
    { 'type': 'H'|'L', 'i': int, 'price': float }
    """
    pivots: list[dict] = []
    if close is None or len(close) < (window * 2 + 1):
        return pivots
    n = len(close)
    for i in range(window, n - window):
        seg = close.iloc[i - window:i + window + 1]
        c = float(close.iloc[i])
        if c == float(seg.max()):
            pivots.append({'type': 'H', 'i': int(i), 'price': c})
        if c == float(seg.min()):
            pivots.append({'type': 'L', 'i': int(i), 'price': c})
    pivots.sort(key=lambda x: x['i'])
    return pivots


def infer_structure(pivots: list[dict]):
    """Infer basic market structure from pivots.

    Returns:
      {
        'trend': 'UP'|'DOWN'|'RANGE'|'UNKNOWN',
        'last_high': float|None,
        'prev_high': float|None,
        'last_low': float|None,
        'prev_low': float|None,
        'labels': {'high': 'HH'|'LH'|None, 'low': 'HL'|'LL'|None}
      }
    """
    highs = [p for p in pivots if p.get('type') == 'H']
    lows = [p for p in pivots if p.get('type') == 'L']

    last_high = highs[-1]['price'] if len(highs) >= 1 else None
    prev_high = highs[-2]['price'] if len(highs) >= 2 else None
    last_low = lows[-1]['price'] if len(lows) >= 1 else None
    prev_low = lows[-2]['price'] if len(lows) >= 2 else None

    high_label = None
    low_label = None
    if last_high is not None and prev_high is not None:
        high_label = 'HH' if last_high > prev_high else 'LH'
    if last_low is not None and prev_low is not None:
        low_label = 'HL' if last_low > prev_low else 'LL'

    if high_label == 'HH' and low_label == 'HL':
        trend = 'UP'
    elif high_label == 'LH' and low_label == 'LL':
        trend = 'DOWN'
    elif high_label is None and low_label is None:
        trend = 'UNKNOWN'
    else:
        trend = 'RANGE'

    return {
        'trend': trend,
        'last_high': last_high,
        'prev_high': prev_high,
        'last_low': last_low,
        'prev_low': prev_low,
        'labels': {'high': high_label, 'low': low_label},
    }


def detect_fvgs(df: pd.DataFrame):
    """Detect classic 3-candle Fair Value Gaps (ICT-style).

    Bullish FVG at i when High[i] < Low[i+2]; gap = (High[i], Low[i+2])
    Bearish FVG at i when Low[i] > High[i+2]; gap = (High[i+2], Low[i])

    Returns list of dicts:
      {
        'type': 'BULL'|'BEAR',
        'i': int,            # left candle index
        'lower': float,      # lower bound of gap
        'upper': float,      # upper bound of gap
      }
    """
    fvgs: list[dict] = []
    if df is None or df.empty:
        return fvgs
    if not {'High', 'Low'}.issubset(df.columns):
        return fvgs
    high = df['High'].astype(float).reset_index(drop=True)
    low = df['Low'].astype(float).reset_index(drop=True)
    n = len(df)
    if n < 3:
        return fvgs

    for i in range(0, n - 2):
        h0 = float(high.iloc[i])
        l0 = float(low.iloc[i])
        h2 = float(high.iloc[i + 2])
        l2 = float(low.iloc[i + 2])

        # bullish imbalance (gap below price after displacement up)
        if h0 < l2:
            fvgs.append({'type': 'BULL', 'i': int(i), 'lower': h0, 'upper': l2})
        # bearish imbalance (gap above price after displacement down)
        if l0 > h2:
            fvgs.append({'type': 'BEAR', 'i': int(i), 'lower': h2, 'upper': l0})

    return fvgs


def filter_unfilled_fvgs(fvgs: list[dict], df: pd.DataFrame):
    """Keep only FVGS that have not been fully traded through after creation."""
    if not fvgs or df is None or df.empty:
        return []
    if not {'High', 'Low'}.issubset(df.columns):
        return []

    high = df['High'].astype(float).reset_index(drop=True)
    low = df['Low'].astype(float).reset_index(drop=True)
    n = len(df)

    out: list[dict] = []
    for f in fvgs:
        i = int(f.get('i', 0))
        if i < 0 or i >= n:
            continue
        lower = float(f.get('lower'))
        upper = float(f.get('upper'))
        ftype = f.get('type')
        if upper <= lower:
            continue

        # After the FVG forms (at i+2), check if it was fully filled.
        start = min(n, i + 3)
        if start >= n:
            out.append(f)
            continue

        if ftype == 'BULL':
            # fully filled if price later trades down to/below the lower bound
            min_low = float(low.iloc[start:].min())
            if min_low > lower:
                out.append(f)
        elif ftype == 'BEAR':
            # fully filled if price later trades up to/above the upper bound
            max_high = float(high.iloc[start:].max())
            if max_high < upper:
                out.append(f)
    return out


def breakout_retest_setup(
    df: pd.DataFrame,
    pivots: list[dict],
    atr: float,
    direction: str,
    max_bars_since_breakout: int = 40,
    buffer_atr_mult: float = 0.10,
    retest_atr_mult: float = 0.15,
):
    """Detect a breakout + retest setup (educational).

    For BUY:
      - level = most recent pivot high
      - breakout if a close breaks above level + buffer
      - retest if later candle's low touches near level and closes back above level

    For SELL:
      - level = most recent pivot low
      - breakout if a close breaks below level - buffer
      - retest if later candle's high touches near level and closes back below level

    Returns a dict when found:
      {
        'name': 'Breakout + retest',
        'level': float,
        'breakout_i': int,
        'retest_i': int,
        'entry': float,
        'sl': float,
        'tp': float,
        'why': str,
      }
    else None.
    """
    if df is None or df.empty or atr is None:
        return None
    if not {'Open', 'High', 'Low', 'Close'}.issubset(df.columns):
        return None
    if not pivots:
        return None

    try:
        atr = float(atr)
    except Exception:
        return None
    if not np.isfinite(atr) or atr <= 0:
        return None

    direction = (direction or '').upper()
    close = df['Close'].astype(float).reset_index(drop=True)
    high = df['High'].astype(float).reset_index(drop=True)
    low = df['Low'].astype(float).reset_index(drop=True)
    n = len(df)
    if n < 10:
        return None

    buffer = buffer_atr_mult * atr
    tol = retest_atr_mult * atr
    max_bars_since_breakout = int(max_bars_since_breakout)
    if max_bars_since_breakout < 5:
        max_bars_since_breakout = 5

    if direction == 'BUY':
        highs = [p for p in pivots if p.get('type') == 'H']
        if not highs:
            return None

        # Try a few recent swing highs; the very last pivot high can be too "fresh"
        for pivot in reversed(highs[-6:]):
            level = float(pivot['price'])
            pivot_i = int(pivot['i'])
            if pivot_i >= n - 6:
                continue

            breakout_i = None
            for i in range(max(pivot_i + 1, 0), n):
                if float(close.iloc[i]) > level + buffer:
                    breakout_i = i
                    break
            if breakout_i is None:
                continue

            end = min(n, breakout_i + 1 + max_bars_since_breakout)
            for j in range(breakout_i + 1, end):
                touched = float(low.iloc[j]) <= level + tol
                close_j = float(close.iloc[j])
                open_j = float(df['Open'].astype(float).reset_index(drop=True).iloc[j])
                closed_back = close_j >= level  # confirm reclaim, buffer not required
                bullish = close_j >= open_j
                if touched and closed_back and bullish:
                    entry = level  # buy limit at the retest level after confirmation
                    sl = level - 1.0 * atr
                    tp = entry + 2.0 * atr
                    return {
                        'name': 'Breakout + retest',
                        'level': level,
                        'breakout_i': int(breakout_i),
                        'retest_i': int(j),
                        'entry': float(entry),
                        'sl': float(sl),
                        'tp': float(tp),
                        'why': f"Breakout close above swing high {level:.5f} (+{buffer_atr_mult:.2f} ATR buffer) then retest/hold within {retest_atr_mult:.2f} ATR.",
                    }
        return None

    if direction == 'SELL':
        lows = [p for p in pivots if p.get('type') == 'L']
        if not lows:
            return None

        for pivot in reversed(lows[-6:]):
            level = float(pivot['price'])
            pivot_i = int(pivot['i'])
            if pivot_i >= n - 6:
                continue

            breakout_i = None
            for i in range(max(pivot_i + 1, 0), n):
                if float(close.iloc[i]) < level - buffer:
                    breakout_i = i
                    break
            if breakout_i is None:
                continue

            end = min(n, breakout_i + 1 + max_bars_since_breakout)
            for j in range(breakout_i + 1, end):
                touched = float(high.iloc[j]) >= level - tol
                close_j = float(close.iloc[j])
                open_j = float(df['Open'].astype(float).reset_index(drop=True).iloc[j])
                closed_back = close_j <= level
                bearish = close_j <= open_j
                if touched and closed_back and bearish:
                    entry = level
                    sl = level + 1.0 * atr
                    tp = entry - 2.0 * atr
                    return {
                        'name': 'Breakout + retest',
                        'level': level,
                        'breakout_i': int(breakout_i),
                        'retest_i': int(j),
                        'entry': float(entry),
                        'sl': float(sl),
                        'tp': float(tp),
                        'why': f"Breakout close below swing low {level:.5f} (-{buffer_atr_mult:.2f} ATR buffer) then retest/reject within {retest_atr_mult:.2f} ATR.",
                    }
        return None

    return None


def detect_breakout(
    df: pd.DataFrame,
    pivots: list[dict],
    atr: float,
    direction: str,
    lookback_pivots: int = 6,
    buffer_atr_mult: float = 0.10,
):
    """Detect a breakout beyond a recent swing level (no retest requirement).

    Returns dict:
      {
        'direction': 'BUY'|'SELL',
        'level': float,
        'pivot_i': int,
        'breakout_i': int,
        'breakout_time': pandas.Timestamp|None,
      }
    or None.
    """
    if df is None or df.empty or not pivots:
        return None
    if not {'Close'}.issubset(df.columns):
        return None
    try:
        atr = float(atr)
    except Exception:
        return None
    if not np.isfinite(atr) or atr <= 0:
        return None

    direction = (direction or '').upper()
    close = df['Close'].astype(float).reset_index(drop=True)
    n = len(df)
    buffer = buffer_atr_mult * atr

    if direction == 'BUY':
        highs = [p for p in pivots if p.get('type') == 'H']
        for pivot in reversed(highs[-lookback_pivots:]):
            level = float(pivot['price'])
            pivot_i = int(pivot['i'])
            for i in range(max(pivot_i + 1, 0), n):
                if float(close.iloc[i]) > level + buffer:
                    t = None
                    try:
                        t = df.index[i]
                    except Exception:
                        t = None
                    return {
                        'direction': 'BUY',
                        'level': level,
                        'pivot_i': pivot_i,
                        'breakout_i': int(i),
                        'breakout_time': t,
                    }
        return None

    if direction == 'SELL':
        lows = [p for p in pivots if p.get('type') == 'L']
        for pivot in reversed(lows[-lookback_pivots:]):
            level = float(pivot['price'])
            pivot_i = int(pivot['i'])
            for i in range(max(pivot_i + 1, 0), n):
                if float(close.iloc[i]) < level - buffer:
                    t = None
                    try:
                        t = df.index[i]
                    except Exception:
                        t = None
                    return {
                        'direction': 'SELL',
                        'level': level,
                        'pivot_i': pivot_i,
                        'breakout_i': int(i),
                        'breakout_time': t,
                    }
        return None

    return None


def confirm_retest(
    df: pd.DataFrame,
    level: float,
    atr: float,
    direction: str,
    start_time=None,
    retest_atr_mult: float = 0.15,
    require_candle_color: bool = True,
):
    """Confirm a retest/hold of a level after a given start_time.

    For BUY: low touches <= level + tol and close >= level (and optionally bullish candle)
    For SELL: high touches >= level - tol and close <= level (and optionally bearish candle)

    Returns dict:
      { 'retest_time': Timestamp|None, 'entry': float, 'note': str }
    or None.
    """
    if df is None or df.empty:
        return None
    if not {'Open', 'High', 'Low', 'Close'}.issubset(df.columns):
        return None
    try:
        level = float(level)
        atr = float(atr)
    except Exception:
        return None
    if not np.isfinite(level) or not np.isfinite(atr) or atr <= 0:
        return None

    direction = (direction or '').upper()
    tol = retest_atr_mult * atr

    df2 = df
    if start_time is not None:
        try:
            df2 = df.loc[df.index >= start_time]
        except Exception:
            df2 = df
    if df2.empty:
        return None

    o = df2['Open'].astype(float)
    h = df2['High'].astype(float)
    l = df2['Low'].astype(float)
    c = df2['Close'].astype(float)

    for t, open_, high_, low_, close_ in zip(df2.index, o.values, h.values, l.values, c.values):
        if direction == 'BUY':
            touched = float(low_) <= level + tol
            held = float(close_) >= level
            ok = True
            if require_candle_color:
                ok = float(close_) >= float(open_)
            if touched and held and ok:
                return {'retest_time': t, 'entry': float(level), 'note': f"M5 retest/hold within {retest_atr_mult:.2f} ATR"}
        elif direction == 'SELL':
            touched = float(high_) >= level - tol
            held = float(close_) <= level
            ok = True
            if require_candle_color:
                ok = float(close_) <= float(open_)
            if touched and held and ok:
                return {'retest_time': t, 'entry': float(level), 'note': f"M5 retest/reject within {retest_atr_mult:.2f} ATR"}
    return None

def find_swing_points(close: pd.Series, window=2):
    highs, lows = [], []
    n = len(close)
    for i in range(window, n - window):
        seg = close.iloc[i-window:i+window+1]
        if close.iloc[i] == seg.max():
            highs.append(float(close.iloc[i]))
        if close.iloc[i] == seg.min():
            lows.append(float(close.iloc[i]))
    return highs, lows

def pick_zones(highs, lows, top_n=2):
    # Keep closest levels only
    res = sorted(set(highs))[-top_n:]
    sup = sorted(set(lows))[:top_n]
    return res, sup

