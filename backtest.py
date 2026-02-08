import yaml
import yfinance as yf
import pandas as pd
import numpy as np
import argparse
from dataclasses import dataclass
from datetime import timedelta, timezone

from app import s_r, scoring


# Optional yfinance symbol overrides (populated in main() from config)
YF_SYMBOL_OVERRIDES: dict[str, str] = {}


def fetch_fx(pair: str, interval: str, period: str) -> pd.DataFrame:
    p = str(pair).strip().upper()
    symbol = (YF_SYMBOL_OVERRIDES.get(p) or f"{p[:3]}{p[3:]}=X")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    # Ensure tz-aware UTC index if available
    try:
        if getattr(df.index, 'tz', None) is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
    except Exception:
        pass
    return df


def atr_hilo(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return (df['High'] - df['Low']).rolling(n).mean()


def pair_perf_at_time(df: pd.DataFrame, t, lookback_bars: int) -> float:
    """Percent performance ending at the last bar <= t."""
    if df is None or df.empty:
        return 0.0
    try:
        d = df.loc[df.index <= t]
    except Exception:
        d = df
    if d.empty:
        return 0.0
    close = d['Close']
    lb = int(lookback_bars) if lookback_bars is not None else 1
    if lb < 1:
        lb = 1
    if len(close) < 2:
        return 0.0
    idx0 = max(0, len(close) - 1 - lb)
    c0 = float(close.iloc[idx0])
    c1 = float(close.iloc[-1])
    if c0 == 0 or np.isnan(c0) or np.isnan(c1):
        return 0.0
    return float((c1 / c0 - 1.0) * 100.0)


def strength_snapshot(
    t,
    pairs: list[str],
    df_m15_by_pair: dict[str, pd.DataFrame],
    df_m5_by_pair: dict[str, pd.DataFrame],
    lookback_m15: int,
    lookback_m5: int,
    weights: dict,
    strength_pairs: list[str] | None = None,
):
    """Compute normalized currency strength per TF and weighted pair gap scores at time t."""
    # pair performance per timeframe
    spairs = strength_pairs if strength_pairs is not None else pairs
    pair_scores_m15 = {}
    pair_scores_m5 = {}
    for p in spairs:
        pair_scores_m15[p] = pair_perf_at_time(df_m15_by_pair.get(p), t, lookback_m15)
        pair_scores_m5[p] = pair_perf_at_time(df_m5_by_pair.get(p), t, lookback_m5)

    s15 = scoring.normalize_strength(scoring.build_currency_strength(pair_scores_m15))
    s5 = scoring.normalize_strength(scoring.build_currency_strength(pair_scores_m5))

    # weighted currency strength (using normalized strengths per TF)
    w15 = float(weights.get('M15', 0.7))
    w5 = float(weights.get('M5', 0.3))
    currencies = sorted(set(list(s15.keys()) + list(s5.keys())))
    w_strength = {c: float(s15.get(c, 0)) * w15 + float(s5.get(c, 0)) * w5 for c in currencies}
    ranked = sorted(w_strength.items(), key=lambda kv: kv[1], reverse=True)
    rank_of = {cur: i for i, (cur, _) in enumerate(ranked)}

    # weighted gap per pair
    gap = {}
    for p in pairs:
        base, quote = p[:3], p[3:]
        gap[p] = (s15.get(base, 0) - s15.get(quote, 0)) * w15 + (s5.get(base, 0) - s5.get(quote, 0)) * w5
    return {
        'strength': {'M15': s15, 'M5': s5},
        'gap': gap,
        'weighted_strength': w_strength,
        'rank_of': rank_of,
    }


def _simulate_exit(df_m5: pd.DataFrame, entry_time, direction: str, entry: float, sl: float, tp: float):
    """Simulate next-touch exit after entry_time using OHLC.

    Conservative tie-break: if both SL and TP are inside same bar, assume SL hit first.
    Returns dict with exit_time, exit_price, outcome ('TP'|'SL'|'NONE').
    """
    if df_m5.empty:
        return None
    direction = direction.upper()
    post = df_m5.loc[df_m5.index > entry_time]
    if post.empty:
        return None

    for t, row in post.iterrows():
        hi = float(row['High'])
        lo = float(row['Low'])

        if direction == 'BUY':
            sl_hit = lo <= sl
            tp_hit = hi >= tp
            if sl_hit and tp_hit:
                return {'exit_time': t, 'exit_price': sl, 'outcome': 'SL'}
            if sl_hit:
                return {'exit_time': t, 'exit_price': sl, 'outcome': 'SL'}
            if tp_hit:
                return {'exit_time': t, 'exit_price': tp, 'outcome': 'TP'}
        else:
            sl_hit = hi >= sl
            tp_hit = lo <= tp
            if sl_hit and tp_hit:
                return {'exit_time': t, 'exit_price': sl, 'outcome': 'SL'}
            if sl_hit:
                return {'exit_time': t, 'exit_price': sl, 'outcome': 'SL'}
            if tp_hit:
                return {'exit_time': t, 'exit_price': tp, 'outcome': 'TP'}

    return {'exit_time': post.index[-1], 'exit_price': float(post['Close'].iloc[-1]), 'outcome': 'NONE'}


def _is_engulfing(df: pd.DataFrame, t, direction: str) -> bool:
    """Simple engulfing check on the candle at time t vs previous candle.

    BUY: bullish engulfing (current body engulfs prior body)
    SELL: bearish engulfing
    """
    if df is None or df.empty:
        return False
    if not {'Open', 'Close'}.issubset(df.columns):
        return False
    direction = (direction or '').upper()
    try:
        if t not in df.index:
            # if exact index missing, find nearest <= t
            d = df.loc[df.index <= t]
            if d.empty:
                return False
            t = d.index[-1]
        idx = df.index.get_loc(t)
    except Exception:
        return False

    if isinstance(idx, slice) or isinstance(idx, (np.ndarray, list)):
        try:
            idx = int(np.atleast_1d(idx)[0])
        except Exception:
            return False
    if idx is None or idx <= 0:
        return False

    o0 = float(df['Open'].iloc[idx - 1])
    c0 = float(df['Close'].iloc[idx - 1])
    o1 = float(df['Open'].iloc[idx])
    c1 = float(df['Close'].iloc[idx])

    prev_hi = max(o0, c0)
    prev_lo = min(o0, c0)
    cur_hi = max(o1, c1)
    cur_lo = min(o1, c1)

    if direction == 'BUY':
        bullish = c1 > o1
        engulfs = (cur_lo <= prev_lo) and (cur_hi >= prev_hi)
        return bool(bullish and engulfs)
    if direction == 'SELL':
        bearish = c1 < o1
        engulfs = (cur_lo <= prev_lo) and (cur_hi >= prev_hi)
        return bool(bearish and engulfs)
    return False


def _structure_asof(df_h1: pd.DataFrame, window: int, asof_i: int) -> dict:
    """Compute pivot structure using only data up to asof_i (avoid lookahead)."""
    if df_h1 is None or df_h1.empty:
        return {'trend': 'UNKNOWN'}
    asof_i = int(asof_i)
    if asof_i < 10:
        return {'trend': 'UNKNOWN'}
    sub = df_h1.iloc[:asof_i + 1]
    close = sub['Close']
    piv = s_r.find_pivots(close, window=window)
    return s_r.infer_structure(piv)


def _fvg_confluence_h1(df_h1: pd.DataFrame, level: float, breakout_i: int, direction: str):
    """Check whether an unfilled H1 FVG (formed before breakout) intersects the retest level.

    Returns (has_fvg, fvg_dict|None).
    """
    if df_h1 is None or df_h1.empty:
        return False, None
    if not {'High', 'Low'}.issubset(df_h1.columns):
        return False, None
    direction = (direction or '').upper()
    breakout_i = int(breakout_i)
    if breakout_i < 3:
        return False, None

    # Only allow FVGs whose i+2 <= breakout_i (formed fully before breakout candle)
    sub = df_h1.iloc[:breakout_i + 1]
    fvgs = s_r.detect_fvgs(sub)
    fvgs = [f for f in fvgs if int(f.get('i', -999)) + 2 <= breakout_i]
    unfilled = s_r.filter_unfilled_fvgs(fvgs, sub)
    if not unfilled:
        return False, None

    # Prefer the most recent FVG of matching type that intersects the level
    want = 'BULL' if direction == 'BUY' else 'BEAR' if direction == 'SELL' else None
    cand = [f for f in unfilled if (want is None or f.get('type') == want)]
    cand.sort(key=lambda x: int(x.get('i', 0)), reverse=True)
    lvl = float(level)
    for f in cand:
        lo = float(f.get('lower'))
        hi = float(f.get('upper'))
        if lo <= lvl <= hi:
            # attach a timestamp for convenience
            try:
                f = dict(f)
                f['time'] = sub.index[int(f.get('i', 0))]
            except Exception:
                pass
            return True, f
    return False, None


def backtest_pair(
    pair: str,
    days: str = '7d',
    pivot_window: int = 2,
    buffer_atr_mult: float = 0.10,
    retest_atr_mult: float = 0.15,
    sl_atr_mult: float = 1.0,
    tp_atr_mult: float = 3.0,
    allow_overlap: bool = False,
    require_candle_color: bool = True,
    # optional bias filter (strength-gap replay)
    apply_bias_filter: bool = False,
    require_extremes: bool = False,
    extremes_top_n: int = 1,
    gap_threshold: float = 2.0,
    pairs_for_strength: list[str] | None = None,
    df_m15_by_pair: dict[str, pd.DataFrame] | None = None,
    df_m5_by_pair: dict[str, pd.DataFrame] | None = None,
    lookback_m15: int = 96,
    lookback_m5: int = 288,
    weights: dict | None = None,
    test_start_time=None,
    exact_level: bool = True,
    exclude_from_strength: list[str] | None = None,
):
    """Backtest H1 breakout + M5 retest for the last `days`.

    If apply_bias_filter=True, the trade is only taken if the strength-gap bias
    at breakout_time supports the direction.
    """
    # Use extra H1 history for pivots/levels.
    df_h1 = fetch_fx(pair, '60m', '30d')
    df_m5 = fetch_fx(pair, '5m', days)

    if df_h1.empty or df_m5.empty or len(df_h1) < 80 or len(df_m5) < 200:
        return {'pair': pair, 'trades': [], 'note': 'Insufficient data'}

    atr = atr_hilo(df_h1, 14)
    close_h1 = df_h1['Close']
    pivots = s_r.find_pivots(close_h1, window=pivot_window)

    # Build candidate breakouts from pivots (BUY from pivot highs, SELL from pivot lows)
    # Then enforce non-overlapping trades.
    trades = []

    def pip_size(p: str) -> float:
        p = str(p)
        if p == 'BTCUSD':
            # Use $1 moves as the reporting unit ("point")
            return 1.0
        if p == 'XAUUSD':
            return 0.1
        return 0.01 if p.endswith('JPY') else 0.0001

    last_trade_exit_time = None

    # Define test window start (last 7 days ending at last available m5 candle)
    end_time = df_m5.index[-1]
    if test_start_time is None:
        try:
            test_start_time = end_time - pd.Timedelta(days=7)
        except Exception:
            test_start_time = None

    # Use pivot events chronologically
    for pv in pivots:
        pv_i = int(pv['i'])
        if pv_i < 20 or pv_i >= len(df_h1) - 5:
            continue

        pv_time = df_h1.index[pv_i]
        if (not allow_overlap) and last_trade_exit_time is not None and pv_time <= last_trade_exit_time:
            continue

        a = float(atr.iloc[pv_i]) if not pd.isna(atr.iloc[pv_i]) else None
        if a is None or a <= 0:
            continue

        direction = 'BUY' if pv['type'] == 'H' else 'SELL'
        level = float(pv['price'])
        buffer = buffer_atr_mult * a

        # Find breakout close after pivot
        breakout_i = None
        for i in range(pv_i + 1, len(df_h1)):
            c = float(df_h1['Close'].iloc[i])
            if direction == 'BUY' and c > level + buffer:
                breakout_i = i
                break
            if direction == 'SELL' and c < level - buffer:
                breakout_i = i
                break
        if breakout_i is None:
            continue

        breakout_time = df_h1.index[breakout_i]
        if test_start_time is not None and breakout_time < test_start_time:
            continue
        if (not allow_overlap) and last_trade_exit_time is not None and breakout_time <= last_trade_exit_time:
            continue

        # Optional: apply strength-gap bias filter at breakout time
        bias_note = None
        if apply_bias_filter:
            if not pairs_for_strength or df_m15_by_pair is None or df_m5_by_pair is None:
                return {'pair': pair, 'trades': [], 'note': 'Missing strength data for bias filter'}
            ex = set(str(x).strip().upper() for x in (exclude_from_strength or []) if x)
            strength_pairs = [pp for pp in (pairs_for_strength or []) if str(pp).strip().upper() not in ex]
            snap = strength_snapshot(
                t=breakout_time,
                pairs=pairs_for_strength,
                df_m15_by_pair=df_m15_by_pair,
                df_m5_by_pair=df_m5_by_pair,
                lookback_m15=lookback_m15,
                lookback_m5=lookback_m5,
                weights=weights or {'M15': 0.7, 'M5': 0.3},
                strength_pairs=strength_pairs,
            )
            g = float(snap['gap'].get(pair, 0.0))
            bias = 'NEUTRAL'
            if g > float(gap_threshold):
                bias = 'BUY'
            elif g < -float(gap_threshold):
                bias = 'SELL'
            bias_note = f"gap={g:+.2f} bias={bias}"

            if require_extremes and bias in {'BUY', 'SELL'}:
                rank_of = snap.get('rank_of', {}) or {}
                rb = rank_of.get(pair[:3])
                rq = rank_of.get(pair[3:])
                ncur = max(1, len(rank_of))
                top_n = int(extremes_top_n)
                if top_n < 1:
                    top_n = 1
                if rb is None or rq is None:
                    continue
                top_base = rb <= (top_n - 1)
                bot_quote = rq >= (ncur - top_n)
                top_quote = rq <= (top_n - 1)
                bot_base = rb >= (ncur - top_n)

                ok = False
                if bias == 'BUY':
                    ok = top_base and bot_quote
                elif bias == 'SELL':
                    ok = bot_base and top_quote
                bias_note += f" extremes_ok={ok}"
                if not ok:
                    continue

            if bias != direction:
                continue

        # Confirm retest on M5 after breakout_time
        # For backtesting realism, many users want an *exact* touch of the level.
        # If exact_level=True we set the tolerance to 0 so High/Low must reach the level.
        use_retest_mult = 0.0 if exact_level else float(retest_atr_mult)
        ret = s_r.confirm_retest(
            df=df_m5,
            level=level,
            atr=a,
            direction=direction,
            start_time=breakout_time,
            retest_atr_mult=use_retest_mult,
            require_candle_color=bool(require_candle_color),
        )
        if not ret:
            continue

        entry_time = ret['retest_time']
        # Strategy uses a "level" entry; note that the retest condition allows a tolerance band.
        entry_limit = float(ret['entry'])

        # Capture the actual retest candle OHLC for manual chart verification.
        ret_bar = None
        try:
            ret_bar = df_m5.loc[entry_time]
            # if duplicate index, take the last
            if isinstance(ret_bar, pd.DataFrame):
                ret_bar = ret_bar.iloc[-1]
        except Exception:
            ret_bar = None

        ret_o = ret_h = ret_l = ret_c = None
        if ret_bar is not None:
            try:
                ret_o = float(ret_bar['Open'])
                ret_h = float(ret_bar['High'])
                ret_l = float(ret_bar['Low'])
                ret_c = float(ret_bar['Close'])
            except Exception:
                ret_o = ret_h = ret_l = ret_c = None

        tol = float(use_retest_mult) * float(a)
        # Would a limit order at the exact level have filled on the retest candle?
        limit_fillable = None
        try:
            eps = 1e-12
            if direction == 'BUY' and ret_l is not None:
                limit_fillable = bool(ret_l <= (entry_limit + eps))
            elif direction == 'SELL' and ret_h is not None:
                limit_fillable = bool(ret_h >= (entry_limit - eps))
        except Exception:
            limit_fillable = None

        # If exact_level is required, only keep trades where a limit at the level
        # would actually have filled on the retest candle.
        if exact_level and limit_fillable is not True:
            continue

        # For simulation, we keep entry at the limit level to match the intent of the setup.
        entry = float(entry_limit)

        # Confluence tags (computed without lookahead where possible)
        engulfing = _is_engulfing(df_m5, entry_time, direction=direction)
        structure = _structure_asof(df_h1, window=pivot_window, asof_i=breakout_i)
        has_fvg, fvg = _fvg_confluence_h1(df_h1, level=level, breakout_i=breakout_i, direction=direction)

        if direction == 'BUY':
            sl = entry - sl_atr_mult * a
            tp = entry + tp_atr_mult * a
        else:
            sl = entry + sl_atr_mult * a
            tp = entry - tp_atr_mult * a

        exit_info = _simulate_exit(df_m5, entry_time, direction, entry, sl, tp)
        if not exit_info:
            continue

        outcome = exit_info['outcome']
        exit_time = exit_info['exit_time']
        exit_price = float(exit_info['exit_price'])

        risk = abs(entry - sl)
        r_mult = (exit_price - entry) / risk if direction == 'BUY' else (entry - exit_price) / risk

        pips = (exit_price - entry) / pip_size(pair) if direction == 'BUY' else (entry - exit_price) / pip_size(pair)

        trades.append({
            'pair': pair,
            'direction': direction,
            'level': level,
            'pivot_i': pv_i,
            'pivot_time': pv_time,
            'breakout_i': breakout_i,
            'breakout_time': breakout_time,
            'entry_time': entry_time,
            'entry': entry,
            'entry_limit': entry_limit,
            'sl': sl,
            'tp': tp,
            'retest_tol': tol,
            'exact_level': bool(exact_level),
            'retest_open': ret_o,
            'retest_high': ret_h,
            'retest_low': ret_l,
            'retest_close': ret_c,
            'limit_fillable': limit_fillable,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'outcome': outcome,
            'R': float(r_mult),
            'pips': float(pips),
            'bias_note': bias_note,
            'confluence': {
                'retest': True,
                'engulfing': bool(engulfing),
                'h1_structure': structure,
                'h1_fvg_intersects_level': bool(has_fvg),
                'h1_fvg': fvg,
            },
        })

        if not allow_overlap:
            last_trade_exit_time = exit_time

    return {'pair': pair, 'trades': trades, 'note': None}


def summarize(results: list[dict]):
    all_trades = [t for r in results for t in r['trades']]
    if not all_trades:
        return {
            'trades': 0,
            'win_rate': 0.0,
            'avg_R': 0.0,
            'total_R': 0.0,
            'avg_pips': 0.0,
        }

    wins = [t for t in all_trades if t['outcome'] == 'TP']
    completed = [t for t in all_trades if t['outcome'] in {'TP', 'SL'}]

    win_rate = (len(wins) / len(completed) * 100.0) if completed else 0.0
    avg_R = float(np.mean([t['R'] for t in all_trades]))
    total_R = float(np.sum([t['R'] for t in all_trades]))
    avg_pips = float(np.mean([t['pips'] for t in all_trades]))

    return {
        'trades': len(all_trades),
        'completed': len(completed),
        'win_rate': float(win_rate),
        'avg_R': avg_R,
        'total_R': total_R,
        'avg_pips': avg_pips,
    }


def main():
    parser = argparse.ArgumentParser(description='Backtest and/or list weekly trades for the H1 breakout + M5 retest strategy.')
    parser.add_argument('--days', default='7d', help='History window to test/list (default: 7d).')
    parser.add_argument('--pairs', default=None, help='Comma-separated instruments to run (e.g., EURUSD,GBPUSD,BTCUSD). Defaults to config pairs.')
    parser.add_argument('--list', action='store_true', help='Print a detailed per-trade list for manual chart checking.')
    parser.add_argument('--suite', choices=['pattern', 'filtered', 'both'], default='filtered', help='Which suite to list when --list is set (default: filtered).')
    parser.add_argument('--summary-suite', choices=['pattern', 'filtered', 'both'], default='both', help='Which suite(s) to run in summary mode (default: both).')
    parser.add_argument('--csv', default='weekly_trades.csv', help='CSV output path when --list is set (default: weekly_trades.csv).')
    parser.add_argument('--rr', type=float, default=3.0, help='Reward:risk ratio used for TP display (default: 3.0).')
    parser.add_argument('--lot', type=float, default=0.01, help='Lot size for informational risk/profit display (default: 0.01).')
    parser.add_argument('--include-outcome', action='store_true', help='Include exit/outcome lines (default: off for manual backtest).')
    parser.add_argument('--exact-level', action='store_true', default=True, help='Require exact touch of the level on retest (default: on).')
    parser.add_argument('--no-exact-level', action='store_false', dest='exact_level', help='Allow tolerance band around the level (less strict).')
    parser.add_argument('--tolerance-atr-mult', type=float, default=0.15, help='Retest tolerance as ATR multiple when --no-exact-level is used (default: 0.15).')
    parser.add_argument('--pivot-window', type=int, default=2, help='Pivot window for H1 swing detection (default: 2). Smaller => more pivots/trades.')
    parser.add_argument('--buffer-atr-mult', type=float, default=0.10, help='Breakout buffer as ATR multiple (default: 0.10). Smaller => more breakouts.')
    parser.add_argument('--allow-overlap', action='store_true', help='Allow overlapping trades (do not block new entries until prior exit).')
    parser.add_argument('--no-require-candle-color', action='store_false', dest='require_candle_color', help='Do not require bullish/bearish retest candle color (more trades, less strict).')
    parser.set_defaults(require_candle_color=True)
    args = parser.parse_args()

    with open('app/config.yaml') as f:
        config = yaml.safe_load(f)

    # Apply optional yfinance symbol overrides (e.g., XAUUSD -> GC=F)
    try:
        overrides = config.get('yfinance_symbols', {}) or {}
        if isinstance(overrides, dict):
            for k, v in overrides.items():
                if k and v:
                    YF_SYMBOL_OVERRIDES[str(k).strip().upper()] = str(v).strip()
    except Exception:
        pass

    pairs = config.get('pairs', [])
    if args.pairs:
        try:
            requested = [p.strip().upper() for p in str(args.pairs).split(',') if p.strip()]
            if requested:
                pairs = requested
        except Exception:
            pass
    weights = config.get('weights', {'M15': 0.7, 'M5': 0.3})
    gap_threshold = float(config.get('gap_threshold', 2.0))
    account_balance_usd = config.get('account_balance_usd', None)
    local_utc_offset_hours = config.get('local_utc_offset_hours', -5)
    try:
        local_utc_offset_hours = float(local_utc_offset_hours)
    except Exception:
        local_utc_offset_hours = -5.0

    local_tz = timezone(timedelta(hours=local_utc_offset_hours))
    local_tz_label = (
        f"UTC{int(local_utc_offset_hours):+d}"
        if float(local_utc_offset_hours).is_integer()
        else f"UTC{local_utc_offset_hours:+g}"
    )
    # if config sets rr, prefer it unless user explicitly passed a different --rr
    try:
        cfg_rr = float(config.get('rr', 3.0))
        if cfg_rr > 0 and float(args.rr) == 3.0:
            args.rr = cfg_rr
    except Exception:
        pass
    lookback_cfg = config.get('strength_lookback_bars', {})
    # defaults ~1 day
    lookback_m15 = int(lookback_cfg.get('M15', 96))
    lookback_m5 = int(lookback_cfg.get('M5', 288))

    print(f"Backtest: H1 breakout + M5 retest (last {args.days})")
    print('Now includes: optional historical strength-gap bias filter (M15+M5) at breakout time.')
    print('')

    # Pre-fetch strength data once (shared across pairs)
    df_m15_by_pair = {p: fetch_fx(p, '15m', args.days) for p in pairs}
    df_m5_by_pair = {p: fetch_fx(p, '5m', args.days) for p in pairs}

    def run_suite(apply_bias_filter: bool):
        tag = 'WITH strength-gap bias filter' if apply_bias_filter else 'PATTERN ONLY'
        print(f"--- {tag} ---")
        results = []
        for p in pairs:
            r = backtest_pair(
                p,
                days=args.days,
                pivot_window=int(args.pivot_window),
                buffer_atr_mult=float(args.buffer_atr_mult),
                sl_atr_mult=1.0,
                tp_atr_mult=float(args.rr),
                retest_atr_mult=float(args.tolerance_atr_mult),
                exact_level=bool(args.exact_level),
                allow_overlap=bool(args.allow_overlap),
                require_candle_color=bool(args.require_candle_color),
                apply_bias_filter=apply_bias_filter,
                require_extremes=bool(config.get('extremes_filter', False)),
                extremes_top_n=int(config.get('extremes_top_n', 1)),
                gap_threshold=gap_threshold,
                pairs_for_strength=pairs,
                df_m15_by_pair=df_m15_by_pair,
                df_m5_by_pair=df_m5_by_pair,
                lookback_m15=lookback_m15,
                lookback_m5=lookback_m5,
                weights=weights,
                exclude_from_strength=(config.get('exclude_from_strength', []) or []),
            )
            results.append(r)
            trades = r['trades']
            if r['note']:
                print(f"{p}: {r['note']}")
                continue
            if not trades:
                print(f"{p}: 0 trades")
                continue
            completed = [t for t in trades if t['outcome'] in {'TP', 'SL'}]
            wins = [t for t in completed if t['outcome'] == 'TP']
            wr = (len(wins) / len(completed) * 100.0) if completed else 0.0
            total_R = sum(t['R'] for t in trades)
            print(f"{p}: trades={len(trades)} completed={len(completed)} win%={wr:.1f} totalR={total_R:.2f}")

        s = summarize(results)
        print('')
        print('=== Summary ===')
        print(f"Trades: {s['trades']} (completed {s['completed']})")
        print(f"Win rate (completed only): {s['win_rate']:.1f}%")
        print(f"Avg R / trade: {s['avg_R']:.3f}")
        print(f"Total R: {s['total_R']:.2f}")
        print(f"Avg pips / trade: {s['avg_pips']:.1f}")
        print('')
        return results

    def _flatten_trades(results: list[dict], suite_name: str) -> list[dict]:
        out = []
        for r in results:
            for t in (r.get('trades') or []):
                tt = dict(t)
                tt['_suite'] = suite_name
                out.append(tt)
        out.sort(key=lambda x: x.get('entry_time') or x.get('breakout_time') or pd.Timestamp.min)
        return out

    def _fmt_ts(ts) -> str:
        try:
            if ts is None:
                return 'N/A'
            return pd.Timestamp(ts).strftime('%Y-%m-%d %H:%M') + ' UTC'
        except Exception:
            return str(ts)

    def _fmt_ts_local(ts) -> str:
        """Format timestamp in a fixed UTC offset timezone (e.g., UTC-5)."""
        try:
            if ts is None:
                return 'N/A'
            t = pd.Timestamp(ts)
            if t.tzinfo is None:
                t = t.tz_localize('UTC')
            t2 = t.tz_convert(local_tz)
            return t2.strftime('%Y-%m-%d %H:%M') + f" {local_tz_label}"
        except Exception:
            return 'N/A'

    def _fmt_date(ts) -> str:
        try:
            if ts is None:
                return 'N/A'
            return pd.Timestamp(ts).strftime('%Y-%m-%d')
        except Exception:
            return 'N/A'

    def _confluence_tags(tr: dict) -> list[str]:
        c = tr.get('confluence') or {}
        tags = ['retest']
        if c.get('engulfing'):
            tags.append('engulfing')
        if c.get('h1_fvg_intersects_level'):
            tags.append('FVG@level')
        st = (c.get('h1_structure') or {}).get('trend')
        if st:
            tags.append(f"H1-structure:{st}")
        if tr.get('bias_note'):
            tags.append('strength-gap')
        return tags

    def _build_entry_row(tr: dict, rr: float, lot: float) -> dict:
        """Build a flat, CSV-friendly row and a concise entry_line for manual backtesting."""
        pair = str(tr.get('pair'))
        direction = str(tr.get('direction') or '').upper()
        entry_time = tr.get('entry_time')
        breakout_time = tr.get('breakout_time')
        entry = float(tr.get('entry'))
        entry_limit = tr.get('entry_limit', entry)
        try:
            entry_limit = float(entry_limit)
        except Exception:
            entry_limit = float(entry)

        sl = float(tr.get('sl'))
        risk = abs(entry - sl)
        if direction == 'BUY':
            tp = entry + rr * risk
        else:
            tp = entry - rr * risk

        psize = _pip_size(pair)
        risk_pips = (risk / psize) if psize else None
        reward_pips = (abs(tp - entry) / psize) if psize else None

        pip_usd = _pip_value_usd_est(pair, entry, lot)
        risk_usd = (risk_pips * pip_usd) if (pip_usd is not None and risk_pips is not None) else None
        reward_usd = (reward_pips * pip_usd) if (pip_usd is not None and reward_pips is not None) else None

        bal = None
        try:
            bal = float(account_balance_usd) if account_balance_usd is not None else None
        except Exception:
            bal = None
        risk_pct = (risk_usd / bal * 100.0) if (bal and risk_usd is not None) else None
        reward_pct = (reward_usd / bal * 100.0) if (bal and reward_usd is not None) else None

        tags_list = _confluence_tags(tr)
        tags = ", ".join(tags_list)

        c = tr.get('confluence') or {}
        st = (c.get('h1_structure') or {}).get('trend')
        fvg = c.get('h1_fvg') or {}
        fvg_type = fvg.get('type') if isinstance(fvg, dict) else None
        fvg_lower = None
        fvg_upper = None
        fvg_time = None
        try:
            if isinstance(fvg, dict) and fvg.get('lower') is not None and fvg.get('upper') is not None:
                fvg_lower = float(fvg.get('lower'))
                fvg_upper = float(fvg.get('upper'))
                fvg_time = fvg.get('time')
        except Exception:
            pass

        # One-line copy/paste string
        entry_line = (
            f"{_fmt_date(entry_time)} {pair} {direction}: { _fmt_price(pair, entry) }, "
            f"TP { _fmt_price(pair, tp) }, SL { _fmt_price(pair, sl) } "
            f"(RR 1:{rr:g}, lot {lot:g})"
        )

        return {
            'date': _fmt_date(entry_time),
            'pair': pair,
            'direction': direction,
            'entry_time_utc': _fmt_ts(entry_time),
            'breakout_time_utc': _fmt_ts(breakout_time),
            'entry_time_local': _fmt_ts_local(entry_time),
            'breakout_time_local': _fmt_ts_local(breakout_time),
            'entry': entry,
            'entry_limit': entry_limit,
            'sl': sl,
            'tp': float(tp),
            'rr': float(rr),
            'lot': float(lot),
            'risk_pips': float(risk_pips) if risk_pips is not None else None,
            'reward_pips': float(reward_pips) if reward_pips is not None else None,
            'risk_usd_est': float(risk_usd) if risk_usd is not None else None,
            'reward_usd_est': float(reward_usd) if reward_usd is not None else None,
            'risk_pct_balance_est': float(risk_pct) if risk_pct is not None else None,
            'reward_pct_balance_est': float(reward_pct) if reward_pct is not None else None,
            'retest_tol': tr.get('retest_tol'),
            'exact_level': tr.get('exact_level'),
            'retest_open': tr.get('retest_open'),
            'retest_high': tr.get('retest_high'),
            'retest_low': tr.get('retest_low'),
            'retest_close': tr.get('retest_close'),
            'limit_fillable': tr.get('limit_fillable'),
            'tags': tags,
            'h1_structure_trend': st,
            'h1_fvg_type': fvg_type,
            'h1_fvg_lower': fvg_lower,
            'h1_fvg_upper': fvg_upper,
            'h1_fvg_time_utc': _fmt_ts(fvg_time) if fvg_time is not None else None,
            'h1_fvg_time_local': _fmt_ts_local(fvg_time) if fvg_time is not None else None,
            'strength_bias_note': tr.get('bias_note'),
            'entry_line': "".join(entry_line),
        }

    def _pip_size(pair: str) -> float:
        pair = str(pair)
        if pair == 'BTCUSD':
            return 1.0
        # Gold spot is typically quoted to 2 decimals; treat 0.10 as "1 pip" (10 cents)
        # so the pip counts are not enormous. This is a convention for reporting only.
        if pair == 'XAUUSD':
            return 0.1
        return 0.01 if pair.endswith('JPY') else 0.0001

    def _units_from_lot(lot: float) -> float:
        # FX convention: 1.0 lot = 100,000 units
        # XAUUSD common convention: 1.0 lot = 100 troy oz
        try:
            if float(lot) <= 0:
                return 0.0
            # pair is not available here; used for FX only.
            return float(lot) * 100_000.0
        except Exception:
            return 0.0

    def _pip_value_usd_est(pair: str, entry_price: float, lot: float) -> float | None:
        """Estimate pip value in USD for 0.01 lot.

        Works when USD is quote (XXXUSD) or USD is base (USDXXX). Otherwise returns None.
        """
        try:
            pair = str(pair)
            base, quote = pair[:3], pair[3:]
            psize = _pip_size(pair)
            # Estimate contract sizing
            if pair == 'XAUUSD':
                # 1.0 lot ~= 100 oz
                units = float(lot) * 100.0
            elif pair == 'BTCUSD':
                # Treat "lot" as BTC quantity for crypto: 1.0 lot = 1 BTC
                units = float(lot)
            else:
                units = _units_from_lot(lot)
            pip_value_in_quote = units * psize
            if quote == 'USD':
                return float(pip_value_in_quote)
            if base == 'USD' and entry_price and entry_price > 0:
                # quote currency per 1 USD = entry_price, so 1 quote = 1/entry USD
                return float(pip_value_in_quote / float(entry_price))
        except Exception:
            return None
        return None

    def _format_money(x: float | None) -> str:
        if x is None:
            return 'N/A'
        try:
            return f"${float(x):.2f}"
        except Exception:
            return 'N/A'

    def _fmt_price(pair: str, px: float) -> str:
        # keep JPY pairs to 3 decimals, XAUUSD to 2, others 5 for readability
        try:
            pair = str(pair)
            if pair == 'BTCUSD':
                return f"{float(px):.2f}"
            if pair == 'XAUUSD':
                return f"{float(px):.2f}"
            if pair.endswith('JPY'):
                return f"{float(px):.3f}"
            return f"{float(px):.5f}"
        except Exception:
            return str(px)

    def list_trades(results: list[dict], suite_label: str) -> list[dict]:
        trades = _flatten_trades(results, suite_label)
        if not trades:
            print(f"No trades to list for suite: {suite_label}")
            print('')
            return []
        print(f"\nENTRIES LIST ({suite_label})")
        if account_balance_usd is not None:
            try:
                print(f"Budget: ${float(account_balance_usd):.2f} USD")
            except Exception:
                print(f"Budget: {account_balance_usd} USD")
        rr = float(args.rr) if args.rr and args.rr > 0 else 2.0
        lot = float(args.lot) if args.lot and args.lot > 0 else 0.01

        last_day = None
        for i, tr in enumerate(trades, start=1):
            row = _build_entry_row(tr, rr=rr, lot=lot)

            # Date header like your example, but also put date on each line for clarity
            try:
                et = tr.get('entry_time')
                day_label = pd.Timestamp(et).strftime('%b %d, %Y') if et is not None else 'N/A'
            except Exception:
                day_label = 'N/A'
            if day_label != last_day:
                print(day_label)
                last_day = day_label

            # Shorter, less-wrappy line with date included
            date_inline = row.get('date')
            pair = row.get('pair')
            direction = row.get('direction')
            entry = row.get('entry')
            tp = row.get('tp')
            sl = row.get('sl')
            entry_limit = row.get('entry_limit')
            risk_pips = row.get('risk_pips')
            reward_pips = row.get('reward_pips')
            risk_usd = row.get('risk_usd_est')
            reward_usd = row.get('reward_usd_est')
            risk_pct = row.get('risk_pct_balance_est')
            reward_pct = row.get('reward_pct_balance_est')

            tags = row.get('tags') or ''
            extras = []
            if row.get('h1_fvg_type'):
                extras.append(f"H1FVG:{row.get('h1_fvg_type')}")
            if row.get('strength_bias_note'):
                extras.append(f"strength:{row.get('strength_bias_note')}")
            extra_s = (" | " + "; ".join(extras)) if extras else ""

            print(
                f"{i}. {date_inline} {pair} {direction}: {_fmt_price(pair, entry)}, TP {_fmt_price(pair, tp)}, SL {_fmt_price(pair, sl)}"
                f" (RR 1:{rr:g}, lot {lot:g})"
            )
            if entry_limit is not None and float(entry_limit) != float(entry):
                print(f"   entry_limit(level): {_fmt_price(pair, entry_limit)}")
            print(
                f"   risk {risk_pips:.1f}p/{_format_money(risk_usd)}"
                f"{f' ({risk_pct:.1f}% bal)' if risk_pct is not None else ''}"
                f" | reward {reward_pips:.1f}p/{_format_money(reward_usd)}"
                f"{f' ({reward_pct:.1f}% bal)' if reward_pct is not None else ''}"
            )
            # Show the actual retest candle OHLC so manual chart checks match the data source.
            try:
                rh = row.get('retest_high'); rl = row.get('retest_low'); rc = row.get('retest_close')
                tol = row.get('retest_tol')
                lf = row.get('limit_fillable')
                if rh is not None or rl is not None:
                    touch_line = f"retest OHLC (H/L/C): { _fmt_price(pair, rh) if rh is not None else 'N/A' }/{ _fmt_price(pair, rl) if rl is not None else 'N/A' }/{ _fmt_price(pair, rc) if rc is not None else 'N/A' }"
                    if tol is not None:
                        touch_line += f" | tol={float(tol):.5f}"
                    if lf is not None:
                        touch_line += f" | limit@level filled? {bool(lf)}"
                    print(f"   {touch_line}")
            except Exception:
                pass
            print(f"   signals: {tags}{extra_s}")
            print(
                f"   breakout: {row.get('breakout_time_utc')} ({row.get('breakout_time_local')})"
                f" | entry: {row.get('entry_time_utc')} ({row.get('entry_time_local')})"
            )

            if args.include_outcome:
                outcome = tr.get('outcome')
                rmult = tr.get('R')
                pips = tr.get('pips')
                print(f"   outcome: {outcome} | R {float(rmult):+.2f} | pips {float(pips):+.1f} | exit: {_fmt_ts(tr.get('exit_time'))}")

        print('')
        return trades

    if args.list:
        # In list mode, run the chosen suite(s) and print every trade with confluences.
        all_rows = []
        if args.suite in {'pattern', 'both'}:
            pat = run_suite(False)
            rows = list_trades(pat, 'PATTERN ONLY')
            all_rows.extend(rows)
        if args.suite in {'filtered', 'both'}:
            filt = run_suite(True)
            rows = list_trades(filt, 'FILTERED (strength-gap + config)')
            all_rows.extend(rows)

        # Write CSV for easier manual backtesting.
        if all_rows:
            rr = float(args.rr) if args.rr and args.rr > 0 else 3.0
            lot = float(args.lot) if args.lot and args.lot > 0 else 0.01
            rows = []
            for r in all_rows:
                # all_rows are trade dicts from backtest_pair
                try:
                    rows.append(_build_entry_row(r, rr=rr, lot=lot))
                except Exception:
                    # fallback
                    rows.append({'pair': r.get('pair'), 'direction': r.get('direction'), 'entry_time_utc': _fmt_ts(r.get('entry_time'))})

            # Put the most important columns first
            preferred = [
                'date', 'pair', 'direction', 'entry_time_utc', 'breakout_time_utc',
                'entry_time_local', 'breakout_time_local',
                'entry', 'entry_limit', 'sl', 'tp', 'rr', 'lot',
                'risk_pips', 'reward_pips', 'risk_usd_est', 'reward_usd_est', 'risk_pct_balance_est', 'reward_pct_balance_est',
                'retest_tol', 'retest_open', 'retest_high', 'retest_low', 'retest_close', 'limit_fillable',
                'tags', 'h1_structure_trend', 'h1_fvg_type', 'h1_fvg_lower', 'h1_fvg_upper', 'h1_fvg_time_utc', 'h1_fvg_time_local',
                'strength_bias_note', 'entry_line',
            ]
            df_out = pd.DataFrame(rows)
            cols = [c for c in preferred if c in df_out.columns] + [c for c in df_out.columns if c not in preferred]
            df_out = df_out[cols]
            try:
                df_out.to_csv(args.csv, index=False)
                print(f"Wrote CSV: {args.csv}")
            except Exception as e:
                print(f"CSV write failed ({args.csv}): {e}")
        return

    # Default mode: keep the existing summary output
    if args.summary_suite in {'pattern', 'both'}:
        run_suite(False)
    if args.summary_suite in {'filtered', 'both'}:
        run_suite(True)


if __name__ == '__main__':
    main()
