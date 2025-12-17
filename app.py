import os
import pandas as pd
import yaml
from dotenv import load_dotenv
import yfinance as yf
from app import scoring, s_r, telegram_bot

load_dotenv()

TG_TOKEN = os.getenv("TG_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

with open('app/config.yaml') as f:
    config = yaml.safe_load(f)

pairs = config['pairs']
intervals = config['intervals']
weights = config['weights']
ema_short = config['ema_period']['short']
ema_long = config['ema_period']['long']
s_r_config = config['s_r']

def fetch_alpha(pair, interval_str, period="5d"):
    symbol = f"{pair[:3]}{pair[3:]}=X"
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

    # 1️⃣ Compute intraday momentum
    for tf_key, interval in intervals.items():
        pair_scores = {}
        for pair in pairs:
            df = fetch_alpha(pair, interval, period="5d")
            score = scoring.pair_momentum_score(df, ema_short, ema_long)
            pair_scores[pair] = score
        # store raw pair scores for trading decision
        pair_scores_all[tf_key] = pair_scores
        # compute currency strength per timeframe for messaging
        strength = scoring.build_currency_strength(pair_scores)
        normalized = scoring.normalize_strength(strength)
        strength_per_tf[tf_key] = normalized

    # 2️⃣ Intraday bias (no D1/H4)
    pair_biases = {}
    for pair in pairs:
        total = 0
        for tf_key, w in weights.items():
            total += pair_scores_all[tf_key][pair] * w

        if total > 0.3:
            pair_biases[pair] = 'BUY'
        elif total < -0.3:
            pair_biases[pair] = 'SELL'
        else:
            pair_biases[pair] = 'NEUTRAL'

    # 3️⃣ Intraday S/R + ATR (M15 only)
    for pair in pairs:
        df = fetch_alpha(pair, intervals['M15'], period="2d")
        close = df['Close']

        highs, lows = s_r.find_swing_points(close, s_r_config['swing_window'])
        res, sup = s_r.pick_zones(highs, lows)

        atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        price = close.iloc[-1]

        # Tight TP/SL
        s_r_info[pair] = {
            'current_price': price,
            'atr': atr,
            'supports': sup,
            'resistances': res,
            'sl_buy': price - atr * 1.0,
            'tp_buy': price + atr * 1.5,
            'sl_sell': price + atr * 1.0,
            'tp_sell': price - atr * 1.5
        }

    # compute aggregated weighted score per pair for display + decision
    pair_total_scores = {}
    for pair in pairs:
        total = 0
        for tf_key, w in weights.items():
            total += pair_scores_all[tf_key][pair] * w
        pair_total_scores[pair] = float(total)

    lot_size = config.get('lot_size', 0.01)

    poster = telegram_bot.TelegramPoster(TG_TOKEN, CHAT_ID)
    poster.post_scorecard(strength_per_tf, pair_biases, s_r_info, pair_scores=pair_total_scores, lot_size=lot_size)

if __name__ == "__main__":
    run()
