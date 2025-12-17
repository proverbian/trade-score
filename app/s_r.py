# s_r.py
import numpy as np
import pandas as pd

def find_swing_points(close: pd.Series, window=5):
    """
    returns two lists: swing_high_indices, swing_low_indices
    A swing high: value is max in window centered at i
    """
    highs, lows = [], []
    n = len(close)
    for i in range(window, n - window):
        seg = close.iloc[i-window:i+window+1]
        if seg.idxmax() == close.index[i]:
            highs.append((close.index[i], float(close.iloc[i])))
        if seg.idxmin() == close.index[i]:
            lows.append((close.index[i], float(close.iloc[i])))
    return highs, lows

def pick_zones(highs, lows, top_n=3):
    # Sort by recency (most recent first), then take top_n unique levels
    highs_sorted = sorted(set(h[1] for h in highs), reverse=True)[:top_n]  # Descending for resistances
    lows_sorted = sorted(set(l[1] for l in lows))[:top_n]  # Ascending for supports
    return highs_sorted, lows_sorted
