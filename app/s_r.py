# s_r.py
import numpy as np
import pandas as pd

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

