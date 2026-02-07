# telegram_bot.py
import requests
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

class TelegramPoster:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id

    def post_scorecard(self, strength_per_tf: dict, pair_biases: dict, s_r_info: dict = None, pair_scores: dict = None, lot_size: float = 0.01, recommendation: dict | None = None, dry_run: bool = False, account_balance_usd: float | None = None) -> str:
        """
        strength_per_tf: { 'D1': {cur:score,...}, 'H4': {...}, 'H1': {...} }
        pair_biases: { 'EURUSD': 'SELL', 'GBPUSD': 'BUY', ... }
        s_r_info: optional dict { pair: {'resistances': [...], 'supports': [...] } }
        """
        utc_now = datetime.now(timezone.utc)
        ny_tz = ZoneInfo("America/New_York")
        ny_now = utc_now.astimezone(ny_tz)

        def _fmt_hm(dt: datetime) -> str:
            return dt.strftime('%H:%M')

        def _fmt_countdown(td: timedelta) -> str:
            secs = int(td.total_seconds())
            if secs < 0:
                secs = 0
            h = secs // 3600
            m = (secs % 3600) // 60
            if h and m:
                return f"{h}h {m}m"
            if h:
                return f"{h}h"
            return f"{m}m"

        def _session_window(now_local: datetime, start_h: int, start_m: int, end_h: int, end_m: int):
            """Return (active, start_dt, end_dt) for a daily session in local time.

            Handles wrap-around sessions that cross midnight.
            """
            start_t = now_local.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
            end_t = now_local.replace(hour=end_h, minute=end_m, second=0, microsecond=0)

            wraps = (start_h, start_m) > (end_h, end_m)
            if not wraps:
                active_ = start_t <= now_local < end_t
                return active_, start_t, end_t

            # wraps midnight: active if now >= start OR now < end
            if now_local >= start_t:
                # session started today, ends tomorrow
                return True, start_t, end_t + timedelta(days=1)
            # now is after midnight but before end
            return True, start_t - timedelta(days=1), end_t

        # Forex session hours in New York time (commonly used ranges).
        # Note: NY time will be EST/EDT automatically via timezone.
        # These are approximations; broker hours and DST effects vary.
        session_defs = {
            'Sydney': (17, 0, 2, 0),    # 5:00 PM – 2:00 AM
            'Tokyo': (19, 0, 4, 0),     # 7:00 PM – 4:00 AM
            'London': (3, 0, 12, 0),    # 3:00 AM – 12:00 PM
            'New York': (8, 0, 16, 0),  # 8:00 AM – 4:00 PM
        }

        active_sessions = []
        for name, (sh, sm, eh, em) in session_defs.items():
            is_active, s_dt, e_dt = _session_window(ny_now, sh, sm, eh, em)
            if is_active:
                closes_in = e_dt - ny_now
                active_sessions.append((name, closes_in))

        if not active_sessions:
            session_line = f"Sessions: None (UTC {utc_now.strftime('%H:%M')}, NY {_fmt_hm(ny_now)})"
            status = "Thin liquidity / rollover period"
        else:
            # Example desired format: Sessions: Asia (UTC 00:04, closing in: 4h)
            parts = []
            for name, closes_in in active_sessions:
                parts.append(f"{name} (closing in: {_fmt_countdown(closes_in)})")
            session_line = f"Sessions: {', '.join(parts)} (UTC {utc_now.strftime('%H:%M')}, NY {_fmt_hm(ny_now)})"
            if len(active_sessions) == 1:
                status = f"{active_sessions[0][0]} session"
            else:
                status = "Overlap: " + " + ".join([s[0] for s in active_sessions])

        lines = [
            "==FOREX SCORECARD==",
            f"Generated: {utc_now.strftime('%Y-%m-%d %H:%M:%S')} UTC",
            session_line,
            f"Market Status: {status}",
        ]

        if account_balance_usd is not None:
            try:
                lines.append(f"Account Balance: ${float(account_balance_usd):.2f} USD")
            except Exception:
                lines.append(f"Account Balance: {account_balance_usd} USD")

        # Reference session hours list (as requested)
        lines.append("\nSESSION HOURS (New York time):")
        lines.append("Sydney: 5:00 PM - 2:00 AM (NY)")
        lines.append("Tokyo: 7:00 PM - 4:00 AM (NY)")
        lines.append("London: 3:00 AM - 12:00 PM (NY)")
        lines.append("New York: 8:00 AM - 4:00 PM (NY)")
        # print per currency per timeframe using the provided timeframes
        tf_keys = list(strength_per_tf.keys()) if strength_per_tf else []

        def pip_size(pair: str) -> float:
            # JPY pairs typically have 2 decimal places, others 4
            if pair[3:] == 'JPY' or pair.endswith('JPY'):
                return 0.01
            return 0.0001
        currencies = sorted({c for tf in strength_per_tf.values() for c in tf.keys()})
        for cur in currencies:
            row = f"{cur}: "
            parts = []
            for tf in tf_keys:
                val = strength_per_tf.get(tf, {}).get(cur, 0)
                parts.append(f"{tf}: {val:+d}")
            row += " | ".join(parts)
            lines.append(row)

        if recommendation:
            strongest = recommendation.get('strongest')
            weakest = recommendation.get('weakest')
            rec_pair = recommendation.get('pair')
            rec_bias = recommendation.get('bias')
            rec_gap = recommendation.get('gap_score', 0.0)
            gap_thr = recommendation.get('gap_threshold', None)
            lines.append("\nTOP GAP (MarketMilk-style strength):")
            if strongest and weakest:
                lines.append(f"Strongest: {strongest} | Weakest: {weakest}")
            if rec_pair and rec_bias:
                thr_str = f" (threshold {gap_thr:.2f})" if isinstance(gap_thr, (int, float)) else ""
                lines.append(f"Suggested: {rec_pair} {rec_bias} | Gap score: {rec_gap:+.2f}{thr_str}")
            top_pairs = recommendation.get('top_pairs') or []
            if top_pairs:
                lines.append("Top pairs by gap:")
                for i, item in enumerate(top_pairs[:3], start=1):
                    p = item.get('pair')
                    b = item.get('bias')
                    g = item.get('gap_score', 0.0)
                    if p and b:
                        lines.append(f"  {i}. {p}: {b} (gap {float(g):+.2f})")

        lines.append("\nPAIRS:")
        for p, b in pair_biases.items():
            score = None
            if pair_scores and p in pair_scores:
                try:
                    score = float(pair_scores[p])
                except Exception:
                    score = None
            size_str = f"Size: {lot_size}" if lot_size is not None else "Size: N/A"
            score_str = f"Score: {score:+.2f}" if score is not None else "Score: N/A"
            lines.append(f"{p}: {b}  {size_str}  {score_str}")
            if s_r_info and p in s_r_info:
                res = s_r_info[p].get('resistances', [])
                sup = s_r_info[p].get('supports', [])
                atr = s_r_info[p].get('atr', 0.001)
                current_price = s_r_info[p].get('current_price', 0)
                structure = s_r_info[p].get('structure', {}) or {}
                entry_ideas = s_r_info[p].get('entry_ideas', []) or []
                setup_tf = s_r_info[p].get('setup_tf')
                entry_tf = s_r_info[p].get('entry_tf')
                setup_note = s_r_info[p].get('setup_note')
                h1_last_time = s_r_info[p].get('h1_last_time')
                m5_last_time = s_r_info[p].get('m5_last_time')
                lines.append(f"Current Price: {current_price:.4f}")
                if h1_last_time or m5_last_time:
                    h1_s = str(h1_last_time) if h1_last_time is not None else "N/A"
                    m5_s = str(m5_last_time) if m5_last_time is not None else "N/A"
                    lines.append(f"Last candles: H1 {h1_s} | M5 {m5_s}")

                # Market structure summary
                trend = structure.get('trend')
                labels = structure.get('labels', {}) or {}
                hi_lbl = labels.get('high')
                lo_lbl = labels.get('low')
                if trend:
                    parts = [f"Structure: {trend}"]
                    if hi_lbl or lo_lbl:
                        parts.append(f"({hi_lbl or '--'}/{lo_lbl or '--'})")
                    lines.append(" ".join(parts))
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
                # show order/TP/SL with pip distances relative to current price
                psize = pip_size(p)
                if order_at:
                    try:
                        order_pips = abs(order_at - current_price) / psize if current_price else None
                    except Exception:
                        order_pips = None
                    lines.append(f"ORDER AT: {order_at:.4f} ({order_pips:.1f} pips)" if order_pips is not None else f"ORDER AT: {order_at:.4f}")
                else:
                    lines.append("ORDER AT: N/A")

                if tp:
                    try:
                        tp_pips = abs(tp - current_price) / psize if current_price else None
                    except Exception:
                        tp_pips = None
                    lines.append(f"TP: {tp:.4f} ({tp_pips:.1f} pips)" if tp_pips is not None else f"TP: {tp:.4f}")
                else:
                    lines.append("TP: N/A")

                if sl:
                    try:
                        sl_pips = abs(sl - current_price) / psize if current_price else None
                    except Exception:
                        sl_pips = None
                    lines.append(f"SL: {sl:.4f} ({sl_pips:.1f} pips)" if sl_pips is not None else f"SL: {sl:.4f}")
                else:
                    lines.append("SL: N/A")
                lines.append("SUPPORT/RESIST:")
                lines.append(f"  R: {', '.join(f'{r:.4f}' for r in res)}")
                lines.append(f"  S: {', '.join(f'{s:.4f}' for s in sup)}")

                if entry_ideas:
                    if setup_tf and entry_tf:
                        lines.append(f"SETUP ({setup_tf} breakout + {entry_tf} retest, educational):")
                    else:
                        lines.append("SETUP (breakout + retest, educational):")
                    for idea in entry_ideas[:1]:
                        name = idea.get('name', 'Idea')
                        entry = idea.get('entry')
                        sl = idea.get('sl')
                        tp = idea.get('tp')
                        why = idea.get('why', '')
                        try:
                            entry_s = f"{float(entry):.4f}" if entry is not None else "N/A"
                            sl_s = f"{float(sl):.4f}" if sl is not None else "N/A"
                            tp_s = f"{float(tp):.4f}" if tp is not None else "N/A"
                        except Exception:
                            entry_s, sl_s, tp_s = "N/A", "N/A", "N/A"
                        lines.append(f"  - {name}: Entry {entry_s} | SL {sl_s} | TP {tp_s}")
                        if why:
                            lines.append(f"    Why: {why}")
                else:
                    if b in {'BUY', 'SELL'}:
                        if setup_note:
                            lines.append(f"SETUP: {setup_note}")
                        else:
                            lines.append("SETUP: Waiting for breakout + retest confirmation.")
            lines.append("")  # blank line between pairs

        lines.append("\nDisclaimer: This is for educational purposes only. Trading involves risk. Not financial advice.")

        # Remove the old S/R section

        msg = "\n".join(lines)
        if dry_run:
            return msg

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        requests.post(url, data={"chat_id": self.chat_id, "text": msg})
        return msg
