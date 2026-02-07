# telegram_bot.py
import requests
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

class TelegramPoster:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id

    def post_scorecard(
        self,
        strength_per_tf: dict,
        pair_biases: dict,
        s_r_info: dict = None,
        pair_scores: dict = None,
        lot_size: float = 0.01,
        recommendation: dict | None = None,
        dry_run: bool = False,
        account_balance_usd: float | None = None,
        local_utc_offset_hours: float | None = None,
        local_timezone: str | None = None,
        show_currency_strength: bool = True,
        show_neutral_pairs: bool = True,
    ) -> str:
        """
        strength_per_tf: { 'D1': {cur:score,...}, 'H4': {...}, 'H1': {...} }
        pair_biases: { 'EURUSD': 'SELL', 'GBPUSD': 'BUY', ... }
        s_r_info: optional dict { pair: {'resistances': [...], 'supports': [...] } }
        """
        utc_now = datetime.now(timezone.utc)
        ny_tz = ZoneInfo("America/New_York")
        ny_now = utc_now.astimezone(ny_tz)

        # Optional user local time (prefer TZ database name for DST correctness).
        local_tz = None
        local_tz_label = None
        if local_timezone:
            try:
                local_tz = ZoneInfo(str(local_timezone).strip())
                local_tz_label = str(local_timezone).strip()
            except Exception:
                local_tz = None
                local_tz_label = None
        if local_utc_offset_hours is not None:
            try:
                off = float(local_utc_offset_hours)
                # Only use fixed offset if no named timezone was provided.
                if local_tz is None:
                    local_tz = timezone(timedelta(hours=off))
                    local_tz_label = f"UTC{int(off):+d}" if float(off).is_integer() else f"UTC{off:+g}"
            except Exception:
                local_tz = None
                local_tz_label = None

        def _to_dt(x):
            if x is None:
                return None
            try:
                if hasattr(x, 'to_pydatetime'):
                    x = x.to_pydatetime()
            except Exception:
                pass
            if isinstance(x, datetime):
                if x.tzinfo is None:
                    return x.replace(tzinfo=timezone.utc)
                return x
            return None

        def _fmt_dt_utc_and_local(x) -> str:
            d = _to_dt(x)
            if d is None:
                return "N/A"
            d_utc = d.astimezone(timezone.utc)
            base = d_utc.strftime('%Y-%m-%d %H:%M') + ' UTC'
            if local_tz is None:
                return base
            try:
                d_loc = d_utc.astimezone(local_tz)
                return base + f" ({d_loc.strftime('%Y-%m-%d %H:%M')} {local_tz_label})"
            except Exception:
                return base

        def _is_fx_pair(sym: str) -> bool:
            sym = str(sym or '').strip().upper()
            # Crypto trades 24/7; do not apply FX weekend closure messaging.
            if sym in {'BTCUSD'}:
                return False
            # crude but effective for this project: 6-letter FX pairs
            if len(sym) == 6 and sym.isalpha():
                return True
            # treat XAUUSD as FX-like session behavior
            if sym == 'XAUUSD':
                return True
            return False

        def _is_crypto(sym: str) -> bool:
            sym = str(sym or '').strip().upper()
            return sym in {'BTCUSD'}

        def _fx_market_open(now_ny: datetime) -> bool:
            """Approx FX hours in NY time: open Sun 5pm, close Fri 5pm."""
            wd = now_ny.weekday()  # Mon=0 ... Sun=6
            # Saturday closed
            if wd == 5:
                return False
            # Sunday before 5pm closed
            if wd == 6 and now_ny.hour < 17:
                return False
            # Friday after 5pm closed
            if wd == 4 and now_ny.hour >= 17:
                return False
            return True

        def _next_fx_open(now_ny: datetime) -> datetime:
            """Return the next FX open time in NY timezone."""
            wd = now_ny.weekday()  # Mon=0 ... Sun=6
            # If Sunday before 5pm, next open is today 5pm.
            if wd == 6 and now_ny.hour < 17:
                return now_ny.replace(hour=17, minute=0, second=0, microsecond=0)
            # Otherwise, next open is Sunday 5pm of the upcoming week.
            # Compute days until Sunday.
            days_until_sun = (6 - wd) % 7
            target = (now_ny + timedelta(days=days_until_sun)).replace(hour=17, minute=0, second=0, microsecond=0)
            # If we're already past this Sunday's 5pm (e.g., Sunday after open), move to next week.
            if target <= now_ny:
                target = target + timedelta(days=7)
            return target

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

        fx_open = _fx_market_open(ny_now)
        local_hm = None
        try:
            if local_tz is not None:
                local_hm = utc_now.astimezone(local_tz).strftime('%H:%M')
        except Exception:
            local_hm = None

        if not fx_open:
            # Override sessions when the FX market is closed (weekend close).
            bits = [f"UTC {utc_now.strftime('%H:%M')}", f"NY {_fmt_hm(ny_now)}"]
            if local_hm and local_tz_label:
                bits.append(f"Local {local_hm} {local_tz_label}")
            session_line = "CLOSED MARKET (FX weekend close) - " + " | ".join(bits)
            status = "CLOSED MARKET"
            try:
                nxt = _next_fx_open(ny_now)
                lines_next = f"Next open: {nxt.strftime('%a %H:%M')} NY (in {_fmt_countdown(nxt - ny_now)})"
            except Exception:
                lines_next = None
        else:
            active_sessions = []
            for name, (sh, sm, eh, em) in session_defs.items():
                is_active, s_dt, e_dt = _session_window(ny_now, sh, sm, eh, em)
                if is_active:
                    closes_in = e_dt - ny_now
                    active_sessions.append((name, closes_in))

            if not active_sessions:
                bits = [f"UTC {utc_now.strftime('%H:%M')}", f"NY {_fmt_hm(ny_now)}"]
                if local_hm and local_tz_label:
                    bits.append(f"Local {local_hm} {local_tz_label}")
                session_line = "Sessions: None (" + ", ".join(bits) + ")"
                status = "Thin liquidity / rollover period"
            else:
                parts = []
                for name, closes_in in active_sessions:
                    parts.append(f"{name} (closing in: {_fmt_countdown(closes_in)})")
                bits = [f"UTC {utc_now.strftime('%H:%M')}", f"NY {_fmt_hm(ny_now)}"]
                if local_hm and local_tz_label:
                    bits.append(f"Local {local_hm} {local_tz_label}")
                session_line = f"Sessions: {', '.join(parts)} (" + ", ".join(bits) + ")"
                if len(active_sessions) == 1:
                    status = f"{active_sessions[0][0]} session"
                else:
                    status = "Overlap: " + " + ".join([s[0] for s in active_sessions])
            lines_next = None

        lines = [
            "==FOREX SCORECARD==",
            "",
            f"Generated: {utc_now.strftime('%Y-%m-%d %H:%M:%S')} UTC",
            session_line,
            f"Market Status: {status}",
        ]

        if (not fx_open) and 'lines_next' in locals() and lines_next:
            lines.append(lines_next)

        if local_tz is not None and local_tz_label:
            try:
                lines.append(f"Local time: {utc_now.astimezone(local_tz).strftime('%Y-%m-%d %H:%M:%S')} {local_tz_label}")
            except Exception:
                pass

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
        # print per currency per timeframe using the provided timeframes (optional)
        tf_keys = list(strength_per_tf.keys()) if strength_per_tf else []

        def pip_size(pair: str) -> float:
            pair = str(pair)
            # BTCUSD: use $1 moves as the reporting unit
            if pair == 'BTCUSD':
                return 1.0
            # XAUUSD (gold) is typically quoted to 2 decimals.
            # For reporting, treat 0.10 as "1 pip" (10 cents) to keep pip counts readable.
            if pair == 'XAUUSD':
                return 0.1
            # JPY pairs typically have 2 decimal places, others 4
            if pair[3:] == 'JPY' or pair.endswith('JPY'):
                return 0.01
            return 0.0001

        def pip_label(pair: str) -> str:
            pair = str(pair)
            if pair == 'XAUUSD':
                return 'points'
            if pair == 'BTCUSD':
                return 'USD'
            return 'pips'

        def fmt_price(pair: str, px: float | None) -> str:
            if px is None:
                return 'N/A'
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
        if show_currency_strength and strength_per_tf:
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
                    if p and b and (show_neutral_pairs or str(b).upper() != 'NEUTRAL'):
                        lines.append(f"  {i}. {p}: {b} (gap {float(g):+.2f})")

        lines.append("\nPAIRS:")
        for p, b in pair_biases.items():
            if (not show_neutral_pairs) and str(b).upper() == 'NEUTRAL':
                continue
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
                price_h1 = s_r_info[p].get('current_price_h1_close', None)
                price_m5 = s_r_info[p].get('current_price_m5_close', None)
                structure = s_r_info[p].get('structure', {}) or {}
                entry_ideas = s_r_info[p].get('entry_ideas', []) or []
                setup_tf = s_r_info[p].get('setup_tf')
                entry_tf = s_r_info[p].get('entry_tf')
                setup_note = s_r_info[p].get('setup_note')
                h1_last_time = s_r_info[p].get('h1_last_time')
                m5_last_time = s_r_info[p].get('m5_last_time')
                lines.append(f"Current Price (latest close): {fmt_price(p, current_price)}")
                if price_m5 is not None or price_h1 is not None:
                    parts = []
                    if price_m5 is not None:
                        parts.append(f"M5 close {fmt_price(p, price_m5)}")
                    if price_h1 is not None:
                        parts.append(f"H1 close {fmt_price(p, price_h1)}")
                    if parts:
                        lines.append("Price refs: " + " | ".join(parts))

                # Explain stuck prices: show age of last M5 candle and whether FX market is closed.
                try:
                    if m5_last_time is not None:
                        m5_dt = _to_dt(m5_last_time)
                        if m5_dt is not None:
                            age_min = (utc_now - m5_dt.astimezone(timezone.utc)).total_seconds() / 60.0
                            if age_min >= 0:
                                lines.append(f"Data age: {age_min:.0f} min since last M5 candle")
                                if _is_fx_pair(p) and (not fx_open):
                                    lines.append("Market note: FX market is CLOSED (weekend/after Fri close). Price will not update until Sunday open.")
                                elif _is_crypto(p) and age_min > 30:
                                    lines.append("Market note: Crypto trades 24/7; if this looks stuck, it's likely a delayed/stale yfinance feed.")
                                elif age_min > 30:
                                    lines.append("Market note: Data looks delayed/stale (yfinance can lag on intraday).")
                except Exception:
                    pass
                if h1_last_time or m5_last_time:
                    h1_s = _fmt_dt_utc_and_local(h1_last_time) if h1_last_time is not None else "N/A"
                    m5_s = _fmt_dt_utc_and_local(m5_last_time) if m5_last_time is not None else "N/A"
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
                plabel = pip_label(p)
                if order_at:
                    try:
                        order_pips = abs(order_at - current_price) / psize if current_price else None
                    except Exception:
                        order_pips = None
                    lines.append(
                        f"ORDER AT: {fmt_price(p, order_at)} ({order_pips:.1f} {plabel})"
                        if order_pips is not None
                        else f"ORDER AT: {fmt_price(p, order_at)}"
                    )
                else:
                    lines.append("ORDER AT: N/A")

                if tp:
                    try:
                        base_px = order_at if order_at else current_price
                        tp_pips = abs(tp - base_px) / psize if base_px else None
                    except Exception:
                        tp_pips = None
                    lines.append(
                        f"TP: {fmt_price(p, tp)} ({tp_pips:.1f} {plabel})"
                        if tp_pips is not None
                        else f"TP: {fmt_price(p, tp)}"
                    )
                else:
                    lines.append("TP: N/A")

                if sl:
                    try:
                        base_px = order_at if order_at else current_price
                        sl_pips = abs(sl - base_px) / psize if base_px else None
                    except Exception:
                        sl_pips = None
                    lines.append(
                        f"SL: {fmt_price(p, sl)} ({sl_pips:.1f} {plabel})"
                        if sl_pips is not None
                        else f"SL: {fmt_price(p, sl)}"
                    )
                else:
                    lines.append("SL: N/A")

                # RR summary when we have an entry + TP + SL
                if order_at and tp and sl:
                    try:
                        risk = abs(order_at - sl)
                        reward = abs(tp - order_at)
                        if risk and risk > 0 and reward is not None:
                            rr = reward / risk
                            lines.append(f"RR (from entry): 1:{rr:.2f}")
                    except Exception:
                        pass
                lines.append("SUPPORT/RESIST:")
                lines.append(f"  R: {', '.join(fmt_price(p, r) for r in res)}")
                lines.append(f"  S: {', '.join(fmt_price(p, s) for s in sup)}")

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

        def _post_text(text: str):
            resp = requests.post(url, data={"chat_id": self.chat_id, "text": text}, timeout=30)
            if not resp.ok:
                snippet = (resp.text or "").strip().replace("\n", " ")[:300]
                raise RuntimeError(f"Telegram send failed: HTTP {resp.status_code} | {snippet}")

        # Telegram hard limit is 4096 chars; keep a safety margin.
        max_len = 3800
        if len(msg) <= max_len:
            _post_text(msg)
            return msg

        # Split by lines to preserve readability.
        lines_with_nl = msg.splitlines(True)
        chunks: list[str] = []
        cur = ""
        for ln in lines_with_nl:
            if len(cur) + len(ln) > max_len:
                if cur:
                    chunks.append(cur)
                    cur = ""
                if len(ln) > max_len:
                    for i in range(0, len(ln), max_len):
                        chunks.append(ln[i:i + max_len])
                else:
                    cur = ln
            else:
                cur += ln
        if cur:
            chunks.append(cur)

        total = len(chunks)
        for i, ch in enumerate(chunks, start=1):
            prefix = f"(part {i}/{total})\n" if total > 1 else ""
            _post_text(prefix + ch)
        return msg
