import os
import hashlib
from datetime import datetime, timezone

from zoneinfo import ZoneInfo

import importlib.util
import pathlib

import yaml
from dotenv import load_dotenv

from app import telegram_bot


def _is_fx_pair(sym: str) -> bool:
    sym = (sym or "").strip().upper()
    if sym in {"BTCUSD"}:
        return False
    if sym == "XAUUSD":
        return True
    return len(sym) == 6 and sym.isalpha()


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


def _fx_market_open(now_ny: datetime) -> bool:
    """Approx FX hours in NY time: open Sun 5pm, close Fri 5pm."""
    wd = now_ny.weekday()  # Mon=0 ... Sun=6
    if wd == 5:  # Saturday
        return False
    if wd == 6 and now_ny.hour < 17:  # Sunday before 5pm
        return False
    if wd == 4 and now_ny.hour >= 17:  # Friday after 5pm
        return False
    return True


def _fmt_dt_local(utc_dt: datetime, tz_name: str | None) -> str:
    if not tz_name:
        return utc_dt.strftime("%Y-%m-%d %H:%M:%S") + " UTC"
    try:
        loc = utc_dt.astimezone(ZoneInfo(tz_name))
        return loc.strftime("%Y-%m-%d %H:%M:%S") + f" {tz_name}"
    except Exception:
        return utc_dt.strftime("%Y-%m-%d %H:%M:%S") + " UTC"


def _build_signal_message(data: dict, cfg: dict) -> tuple[str, str]:
    utc_now = datetime.now(timezone.utc)
    tz_name = cfg.get("local_timezone")

    ny_now = utc_now.astimezone(ZoneInfo("America/New_York"))
    fx_open = _fx_market_open(ny_now)

    pair_biases: dict = data.get("pair_biases") or {}
    s_r_info: dict = data.get("s_r_info") or {}

    # Entries-only: include any pair with a confirmed entry idea.
    # We also build a stable fingerprint so we don't re-send the same entry if only SL/TP moves.
    entry_fps: list[str] = []
    entry_lines: list[str] = []
    for p in (cfg.get("pairs") or []):
        p = str(p or "").strip().upper()
        info = s_r_info.get(p) or {}

        # Do not alert FX/XAU when the FX market is closed (weekend close).
        if _is_fx_pair(p) and not fx_open:
            continue

        # Skip stale feeds: use last M5 candle age.
        # FX/XAU should be fairly fresh; BTC is 24/7 but yfinance can still lag.
        m5_last = _to_dt(info.get("m5_last_time"))
        if m5_last is not None:
            age_min = (utc_now - m5_last.astimezone(timezone.utc)).total_seconds() / 60.0
            if _is_fx_pair(p) and age_min > 30:
                continue
            if p == "BTCUSD" and age_min > 180:
                continue

        ideas = info.get("entry_ideas") or []
        if not ideas:
            continue
        idea = ideas[0] or {}
        direction = str(idea.get("direction") or "").strip().upper()
        entry = idea.get("entry")
        sl = idea.get("sl")
        tp = idea.get("tp")
        why = idea.get("why")
        retest_time = idea.get("retest_time")
        b = str(pair_biases.get(p, "")).upper() or "N/A"
        side = direction if direction in {"BUY", "SELL"} else b

        # formatting per instrument
        dec = 2 if p in {"XAUUSD", "BTCUSD"} else 5
        try:
            entry_s = f"{float(entry):.{dec}f}" if entry is not None else "N/A"
        except Exception:
            entry_s = "N/A"
        try:
            sl_s = f"{float(sl):.{dec}f}" if sl is not None else "N/A"
        except Exception:
            sl_s = "N/A"
        try:
            tp_s = f"{float(tp):.{dec}f}" if tp is not None else "N/A"
        except Exception:
            tp_s = "N/A"

        entry_lines.append(f"- {p}: {side} | ENTRY {entry_s} | SL {sl_s} | TP {tp_s}")
        if why:
            entry_lines.append(f"  Why: {why}")
        entry_lines.append("")

        # Fingerprint: pair + side + rounded entry + retest time (to minute)
        try:
            ent_fp = float(entry) if entry is not None else None
        except Exception:
            ent_fp = None
        ent_fp_s = entry_s
        rt = _to_dt(retest_time)
        rt_s = rt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ") if rt is not None else "NA"
        entry_fps.append(f"{p}|{side}|{ent_fp_s}|{rt_s}")

    if not entry_lines:
        return "", ""

    lines: list[str] = []
    lines.append("==KOHIX SIGNALS==")
    lines.append(f"Time: {_fmt_dt_local(utc_now, tz_name)}")

    lines.append("")
    lines.append("ENTRIES FOUND (breakout + retest):")
    # drop trailing blank line
    while entry_lines and not entry_lines[-1].strip():
        entry_lines.pop()
    lines.extend(entry_lines)

    # Fingerprint used for dedup.
    # We do NOT include SL/TP in the fingerprint on purpose.
    fp_payload = "\n".join(sorted(set(entry_fps))).strip()
    fp = hashlib.sha256(fp_payload.encode("utf-8")).hexdigest() if fp_payload else ""
    return "\n".join(lines).strip(), fp


def _should_send(message: str, cache_path: str, fingerprint: str | None = None) -> bool:
    """Avoid spamming: don't send if message is empty or unchanged since last send."""
    if not message.strip():
        return False

    h = (fingerprint or "").strip() or hashlib.sha256(message.encode("utf-8")).hexdigest()
    try:
        if os.path.exists(cache_path):
            prev = (open(cache_path, "r", encoding="utf-8").read() or "").strip()
            if prev == h:
                return False
    except Exception:
        pass

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(h)
    except Exception:
        # If we can't write cache, still send (better than missing signals)
        pass

    return True


def main() -> int:
    load_dotenv()

    token = os.getenv("TG_TOKEN")
    chat_id = os.getenv("CHAT_ID")
    dry_run = os.getenv("DRY_RUN", "0").strip().lower() in {"1", "true", "yes", "y"}
    always_send = os.getenv("ALWAYS_SEND", "0").strip().lower() in {"1", "true", "yes", "y"}

    if not token or not chat_id:
        print("Missing TG_TOKEN or CHAT_ID in environment/.env")
        return 2

    with open("app/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # NOTE: There's a folder package named `app/` and a script named `app.py`.
    # Importing `import app` would grab the package, so we load app.py by file path.
    app_py = pathlib.Path(__file__).with_name("app.py")
    spec = importlib.util.spec_from_file_location("wag_app_main", str(app_py))
    if spec is None or spec.loader is None:
        print("Failed to load app.py module")
        return 2
    wag_app_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wag_app_main)

    if not hasattr(wag_app_main, "compute_scorecard_data"):
        print("app.py does not expose compute_scorecard_data (unexpected).")
        return 2

    data = wag_app_main.compute_scorecard_data(cfg)
    msg, fp = _build_signal_message(data, cfg)

    cache_path = os.path.join(os.path.dirname(__file__), ".last_signal_hash.txt")
    if (not always_send) and (not _should_send(msg, cache_path=cache_path, fingerprint=fp)):
        # Silent success for cron jobs.
        return 0

    if always_send and not msg.strip():
        return 0

    poster = telegram_bot.TelegramPoster(token, chat_id)
    poster.send_text(msg, dry_run=dry_run)

    if dry_run:
        print(msg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
