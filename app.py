import os
import pandas as pd
import yaml
from dotenv import load_dotenv
import yfinance as yf
from app import scoring, s_r, telegram_bot

load_dotenv()

TG_TOKEN = os.getenv("TG_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
DRY_RUN = os.getenv("DRY_RUN", "0").strip().lower() in {"1", "true", "yes", "y"}

with open('app/config.yaml') as f:
    config = yaml.safe_load(f)

pairs = config['pairs']
intervals = config['intervals']
weights = config['weights']
ema_short = config['ema_period']['short']
ema_long = config['ema_period']['long']
s_r_config = config['s_r']


def _interval_minutes(interval: str) -> int | None:
    # yfinance intervals like "5m", "15m", "1h" (we only use minutes here)
    if not interval or not isinstance(interval, str):
        return None
    interval = interval.strip().lower()
    if interval.endswith('m'):
        try:
            return int(interval[:-1])
        except Exception:
            return None
    return None


def _default_lookback_bars(interval: str) -> int:
    """Default to ~1 trading day worth of bars for intraday intervals."""
    mins = _interval_minutes(interval)
    if not mins:
        return 96
    return max(2, int((24 * 60) / mins))

def fetch_alpha(pair, interval_str, period="5d"):
    p = str(pair).strip().upper()
    symbol_overrides = config.get('yfinance_symbols', {}) or {}
    symbol = symbol_overrides.get(p) or f"{p[:3]}{p[3:]}=X"
    interval = interval_str  # "5m" or "15m"

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    if df.empty or len(df) < 50:
        raise RuntimeError(f"No data for {pair} {interval}")
    return df[['Open','High','Low','Close','Volume']].dropna()

def run():
    pair_scores_all = {}
    s_r_info = {}
    strength_per_tf = {}
    strength_norm_per_tf = {}

    # 1️⃣ Compute MarketMilk-style performance (pair % change) -> currency strength
    lookback_cfg = config.get('strength_lookback_bars', {})
    exclude_from_strength = set(str(x).strip().upper() for x in (config.get('exclude_from_strength', []) or []) if x)
    for tf_key, interval in intervals.items():
        pair_scores = {}
        lookback = int(lookback_cfg.get(tf_key, _default_lookback_bars(interval)))
        for pair in pairs:
            if str(pair).strip().upper() in exclude_from_strength:
                continue
            df = fetch_alpha(pair, interval, period="5d")
            # score is pair performance over lookback window (percent)
            score = scoring.pair_performance_score(df, lookback_bars=lookback)
            pair_scores[pair] = score
        # store raw pair scores for trading decision
        pair_scores_all[tf_key] = pair_scores
        # compute currency strength per timeframe for messaging
        strength = scoring.build_currency_strength(pair_scores)
        normalized = scoring.normalize_strength(strength)
        strength_per_tf[tf_key] = normalized
        strength_norm_per_tf[tf_key] = normalized

    # 1.5️⃣ Weighted currency strength across configured timeframes (for extremes filter + recommendation)
    currencies = sorted({c for tf in strength_per_tf.values() for c in tf.keys()})
    weighted_strength = {c: 0.0 for c in currencies}
    for tf_key, w in weights.items():
        tf_strength = strength_norm_per_tf.get(tf_key, {})
        for c in currencies:
            weighted_strength[c] += float(tf_strength.get(c, 0)) * float(w)

    # Currency ranks: 0 = strongest
    ranked = sorted(weighted_strength.items(), key=lambda kv: kv[1], reverse=True)
    rank_of = {cur: idx for idx, (cur, _) in enumerate(ranked)}
    extremes_filter = bool(config.get('extremes_filter', False))
    extremes_top_n = int(config.get('extremes_top_n', 1))
    if extremes_top_n < 1:
        extremes_top_n = 1
    ncur = max(1, len(ranked))

    # 2️⃣ Bias from *currency strength gap* (strong base - weak quote)
    gap_threshold = float(config.get('gap_threshold', 2.0))
    pair_biases = {}
    pair_gap_scores = {}
    for pair in pairs:
        base, quote = pair[:3], pair[3:]
        total_gap = 0.0
        for tf_key, w in weights.items():
            tf_strength = strength_norm_per_tf.get(tf_key, {})
            total_gap += (tf_strength.get(base, 0) - tf_strength.get(quote, 0)) * float(w)
        pair_gap_scores[pair] = float(total_gap)

        # initial direction based on gap
        if total_gap > gap_threshold:
            bias = 'BUY'
        elif total_gap < -gap_threshold:
            bias = 'SELL'
        else:
            bias = 'NEUTRAL'

        # optional extremes filter: only strongest-vs-weakest baskets get a directional bias
        if extremes_filter and bias in {'BUY', 'SELL'}:
            rb = rank_of.get(base, None)
            rq = rank_of.get(quote, None)
            if rb is None or rq is None:
                bias = 'NEUTRAL'
            else:
                top_ok = rb <= (extremes_top_n - 1)
                bot_ok = rq >= (ncur - extremes_top_n)
                if bias == 'BUY' and not (top_ok and bot_ok):
                    bias = 'NEUTRAL'
                if bias == 'SELL':
                    # for sells: base should be weak, quote strong
                    top_ok_sell = rq <= (extremes_top_n - 1)
                    bot_ok_sell = rb >= (ncur - extremes_top_n)
                    if not (top_ok_sell and bot_ok_sell):
                        bias = 'NEUTRAL'

        pair_biases[pair] = bias

    # 3️⃣ Higher-probability setup on H1, entry trigger on M5
    rr = float(config.get('rr', 3.0))
    if rr <= 0:
        rr = 3.0
    for pair in pairs:
        bias = pair_biases.get(pair, 'NEUTRAL')

        df_h1 = fetch_alpha(pair, "60m", period="30d")
        close_h1 = df_h1['Close']

        h1_last_time = None
        try:
            h1_last_time = df_h1.index[-1]
        except Exception:
            h1_last_time = None

        highs, lows = s_r.find_swing_points(close_h1, s_r_config['swing_window'])
        res, sup = s_r.pick_zones(highs, lows)

        atr_h1 = (df_h1['High'] - df_h1['Low']).rolling(14).mean().iloc[-1]
        price_h1_close = float(close_h1.iloc[-1])

        pivots_h1 = s_r.find_pivots(close_h1, window=s_r_config['swing_window'])
        structure = s_r.infer_structure(pivots_h1)

        # Entry timeframe data
        df_m5 = fetch_alpha(pair, intervals['M5'], period="5d")
        price_m5_close = None
        try:
            price_m5_close = float(df_m5['Close'].iloc[-1])
        except Exception:
            price_m5_close = None

        # Use the most recent M5 close as "current" for display (still subject to yfinance delay).
        price = price_m5_close if price_m5_close is not None else price_h1_close

        m5_last_time = None
        try:
            m5_last_time = df_m5.index[-1]
        except Exception:
            m5_last_time = None

        # Default status
        setup_note = None
        entry_ideas = []

        if bias in {'BUY', 'SELL'} and atr_h1 is not None and not pd.isna(atr_h1):
            br = s_r.detect_breakout(df_h1, pivots_h1, float(atr_h1), direction=bias)
            if not br:
                setup_note = "Waiting for H1 breakout close beyond last swing."
            else:
                level = float(br['level'])
                breakout_time = br.get('breakout_time')
                ret = s_r.confirm_retest(
                    df=df_m5,
                    level=level,
                    atr=float(atr_h1),
                    direction=bias,
                    start_time=breakout_time,
                )
                if not ret:
                    setup_note = f"H1 breakout detected at {breakout_time}; waiting for M5 retest/confirmation near {level:.5f}."
                else:
                    entry = float(ret['entry'])
                    a = float(atr_h1)
                    retest_time = ret.get('retest_time')
                    if bias == 'BUY':
                        sl = entry - 1.0 * a
                        tp = float(res[0]) if res and float(res[0]) > entry else (entry + rr * a)
                    else:
                        sl = entry + 1.0 * a
                        tp = float(sup[0]) if sup and float(sup[0]) < entry else (entry - rr * a)

                    why = f"H1 breakout @ {level:.5f}, then {ret.get('note','M5 retest')} (time {retest_time})."
                    entry_ideas = [{
                        'name': 'H1 breakout + M5 retest',
                        'entry': entry,
                        'sl': sl,
                        'tp': tp,
                        'why': why,
                        'breakout_time': breakout_time,
                        'retest_time': retest_time,
                    }]
        elif bias in {'BUY', 'SELL'}:
            setup_note = "No ATR/structure yet; waiting."

        # Tight TP/SL
        s_r_info[pair] = {
            'current_price': price,
            'current_price_h1_close': price_h1_close,
            'current_price_m5_close': price_m5_close,
            'atr': atr_h1,
            'supports': sup,
            'resistances': res,
            'structure': structure,
            'setup_tf': 'H1',
            'entry_tf': 'M5',
            'setup_note': setup_note,
            'h1_last_time': h1_last_time,
            'm5_last_time': m5_last_time,
            'entry_ideas': entry_ideas,
            'sl_buy': price - float(atr_h1) * 1.0 if atr_h1 is not None and not pd.isna(atr_h1) else None,
            'tp_buy': price + float(atr_h1) * 1.5 if atr_h1 is not None and not pd.isna(atr_h1) else None,
            'sl_sell': price + float(atr_h1) * 1.0 if atr_h1 is not None and not pd.isna(atr_h1) else None,
            'tp_sell': price - float(atr_h1) * 1.5 if atr_h1 is not None and not pd.isna(atr_h1) else None
        }

    # compute aggregated weighted gap score per pair for display + decision
    pair_total_scores = dict(pair_gap_scores)

    # sort pairs by absolute gap (largest first) so "huge movement" candidates appear on top
    pairs_sorted = sorted(pairs, key=lambda p: abs(pair_total_scores.get(p, 0.0)), reverse=True)
    pair_biases = {p: pair_biases[p] for p in pairs_sorted}
    pair_total_scores = {p: pair_total_scores[p] for p in pairs_sorted}

    # 4️⃣ Determine strongest vs weakest currencies (weighted across timeframes)
    strongest, weakest = scoring.find_strongest_weakest(weighted_strength)
    rec_pair, rec_bias = scoring.pick_pair_from_extremes(strongest, weakest, pairs)
    if rec_pair is None:
        # fallback: use the best available pair by gap score
        rec_pair = pairs_sorted[0] if pairs_sorted else None
        rec_bias = pair_biases.get(rec_pair) if rec_pair else None

    top_pairs = []
    for p in pairs_sorted[:3]:
        top_pairs.append({'pair': p, 'bias': pair_biases.get(p), 'gap_score': float(pair_total_scores.get(p, 0.0))})

    recommendation = {
        'strongest': strongest,
        'weakest': weakest,
        'pair': rec_pair,
        'bias': rec_bias,
        'gap_score': float(pair_total_scores.get(rec_pair, 0.0)) if rec_pair else 0.0,
        'top_pairs': top_pairs,
        'gap_threshold': gap_threshold,
    }

    lot_size = config.get('lot_size', 0.01)
    account_balance_usd = config.get('account_balance_usd', None)

    poster = telegram_bot.TelegramPoster(TG_TOKEN, CHAT_ID)
    msg = poster.post_scorecard(
        strength_per_tf,
        pair_biases,
        s_r_info,
        pair_scores=pair_total_scores,
        lot_size=lot_size,
        recommendation=recommendation,
        account_balance_usd=account_balance_usd,
        dry_run=DRY_RUN,
        local_utc_offset_hours=config.get('local_utc_offset_hours', None),
        show_currency_strength=bool(config.get('show_currency_strength', False)),
        show_neutral_pairs=bool(config.get('show_neutral_pairs', False)),
    )
    if DRY_RUN:
        print(msg)
    else:
        print("Sent scorecard to Telegram.")
    return msg

if __name__ == "__main__":
    run()
