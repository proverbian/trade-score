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

def fetch_alpha(pair, interval_str, period="60d"):
    symbol = f"{pair[:3]}{pair[3:]}=X"
    if interval_str == "60m":
        interval = "60m"
    elif interval_str == "240m":
        interval = "1h"  # approximation for H4
    elif interval_str == "1d":
        interval = "1d"
    else:
        raise ValueError(f"Unsupported interval: {interval_str}")

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty or len(df) < 50:  # Ensure enough data
            raise RuntimeError(f"Insufficient data for {pair} {interval_str}")
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to fetch data for {pair}: {str(e)}")

def run():
    strength_per_tf = {}
    pair_scores_all = {}
    s_r_info = {}

    for tf_key, interval in intervals.items():
        pair_scores = {}
        for pair in pairs:
            df = fetch_alpha(pair, interval)
            score = scoring.pair_momentum_score(df, ema_short, ema_long)
            pair_scores[pair] = score
        strength = scoring.build_currency_strength(pair_scores)
        normalized = scoring.normalize_strength(strength)
        strength_per_tf[tf_key] = normalized
        pair_scores_all[tf_key] = pair_scores

    # Aggregate pair biases with weights
    pair_total_scores = {}
    for pair in pairs:
        total = 0
        for tf_key, w in weights.items():
            total += pair_scores_all[tf_key][pair] * w
        pair_total_scores[pair] = total

    pair_biases = {pair: 'BUY' if score > 0 else 'SELL' if score < 0 else 'NEUTRAL' for pair, score in pair_total_scores.items()}

    # For S/R, use H1 data and calculate ATR for SL/TP
    h1_interval = intervals['H1']
    for pair in pairs:
        df = fetch_alpha(pair, h1_interval)
        close = df['Close']
        highs, lows = s_r.find_swing_points(close, s_r_config['swing_window'])
        res, sup = s_r.pick_zones(highs, lows, top_n=3)
        # Calculate ATR for volatility-adjusted SL/TP
        atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1] if len(df) > 14 else 0.001
        current_price = close.iloc[-1]
        s_r_info[pair] = {'resistances': res, 'supports': sup, 'atr': atr, 'current_price': current_price}

    poster = telegram_bot.TelegramPoster(TG_TOKEN, CHAT_ID)
    poster.post_scorecard(strength_per_tf, pair_biases, s_r_info)

if __name__ == "__main__":
    run()
