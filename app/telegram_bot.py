# telegram_bot.py
import requests
from datetime import datetime

class TelegramPoster:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id

    def post_scorecard(self, strength_per_tf: dict, pair_biases: dict, s_r_info: dict = None):
        """
        strength_per_tf: { 'D1': {cur:score,...}, 'H4': {...}, 'H1': {...} }
        pair_biases: { 'EURUSD': 'SELL', 'GBPUSD': 'BUY', ... }
        s_r_info: optional dict { pair: {'resistances': [...], 'supports': [...] } }
        """
        utc_now = datetime.utcnow()
        hour = utc_now.hour
        if 0 <= hour < 9:
            status = "Tokyo session (00:00-09:00 UTC) - Lower liquidity for EUR/USD/GBP pairs"
        elif 8 <= hour < 17:
            status = "London session (08:00-17:00 UTC) - High liquidity for EUR/GBP pairs"
        elif 13 <= hour < 22:
            status = "New York session (13:30-22:00 UTC) - High liquidity for USD pairs"
        else:
            status = "Overnight/Thin liquidity (22:00-00:00 UTC) - Avoid major moves"

        lines = [f"==WAGFOREX SCORECARD==\nMarket Status: {status} (UTC {utc_now.strftime('%H:%M')})"]
        # print per currency per timeframe like in screenshot
        currencies = sorted({c for tf in strength_per_tf.values() for c in tf.keys()})
        for cur in currencies:
            row = f"{cur}: "
            parts = []
            for tf in ['D1','H4','H1']:
                val = strength_per_tf.get(tf, {}).get(cur, 0)
                parts.append(f"{tf}: {val:+d}")
            row += " | ".join(parts)
            lines.append(row)

        lines.append("\nPAIRS:")
        for p, b in pair_biases.items():
            lines.append(f"{p}: {b}")
            if s_r_info and p in s_r_info:
                res = s_r_info[p].get('resistances', [])
                sup = s_r_info[p].get('supports', [])
                atr = s_r_info[p].get('atr', 0.001)
                current_price = s_r_info[p].get('current_price', 0)
                lines.append(f"Current Price: {current_price:.4f}")
                if b == 'BUY' and sup:
                    order_at = sup[0]
                    tp = res[0] if res else None
                    sl = order_at - atr * 1.5  # Wider stop based on volatility
                elif b == 'SELL' and res:
                    order_at = res[0]
                    tp = sup[0] if sup else None
                    sl = order_at + atr * 1.5
                else:
                    order_at = tp = sl = None
                lines.append(f"ORDER AT: {order_at:.4f}" if order_at else "ORDER AT: N/A")
                lines.append(f"TP: {tp:.4f}" if tp else "TP: N/A")
                lines.append(f"SL: {sl:.4f}" if sl else "SL: N/A")
                lines.append("SUPPORT/RESIST:")
                lines.append(f"  R: {', '.join(f'{r:.4f}' for r in res)}")
                lines.append(f"  S: {', '.join(f'{s:.4f}' for s in sup)}")
            lines.append("")  # blank line between pairs

        lines.append("\nDisclaimer: This is for educational purposes only. Trading involves risk. Not financial advice.")

        # Remove the old S/R section

        msg = "\n".join(lines)
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        requests.post(url, data={"chat_id": self.chat_id, "text": msg})
