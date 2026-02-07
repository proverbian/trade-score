import os
import re
import time
import subprocess
from typing import Optional

import requests
from dotenv import load_dotenv


TELEGRAM_API_BASE = "https://api.telegram.org"


def _chunk_text(text: str, max_len: int = 3800) -> list[str]:
    if text is None:
        return [""]
    if len(text) <= max_len:
        return [text]

    # Prefer splitting on line boundaries.
    lines = text.splitlines(True)
    chunks: list[str] = []
    cur = ""
    for ln in lines:
        if len(cur) + len(ln) > max_len:
            if cur:
                chunks.append(cur)
                cur = ""
            if len(ln) > max_len:
                for i in range(0, len(ln), max_len):
                    chunks.append(ln[i : i + max_len])
            else:
                cur = ln
        else:
            cur += ln
    if cur:
        chunks.append(cur)
    return chunks


def tg_send(token: str, chat_id: str | int, text: str) -> None:
    url = f"{TELEGRAM_API_BASE}/bot{token}/sendMessage"
    chunks = _chunk_text(text)
    total = len(chunks)
    for i, ch in enumerate(chunks, start=1):
        prefix = f"(part {i}/{total})\n" if total > 1 else ""
        resp = requests.post(url, data={"chat_id": chat_id, "text": prefix + ch}, timeout=30)
        if not resp.ok:
            snippet = (resp.text or "").strip().replace("\n", " ")[:300]
            raise RuntimeError(f"Telegram send failed: HTTP {resp.status_code} | {snippet}")


def tg_get_updates(token: str, offset: Optional[int], timeout_s: int = 30) -> dict:
    url = f"{TELEGRAM_API_BASE}/bot{token}/getUpdates"
    params = {"timeout": timeout_s}
    if offset is not None:
        params["offset"] = int(offset)
    resp = requests.get(url, params=params, timeout=timeout_s + 10)
    resp.raise_for_status()
    return resp.json()


def _run_app_scorecard(*, show_neutral_pairs: bool | None = None) -> str:
    """Run app.py in DRY_RUN mode and capture the printed scorecard."""
    env = dict(os.environ)
    env["DRY_RUN"] = "1"
    if show_neutral_pairs is True:
        env["SHOW_NEUTRAL_PAIRS"] = "1"
    elif show_neutral_pairs is False:
        env["SHOW_NEUTRAL_PAIRS"] = "0"

    proc = subprocess.run(
        [sys_executable(), os.path.join(os.path.dirname(__file__), "app.py")],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        msg = "Failed to generate scorecard."
        if err:
            msg += "\n" + err[:1200]
        return msg

    # In DRY_RUN mode, app.py prints the whole message.
    if not out:
        return "Generated scorecard was empty (unexpected)."
    return out


def _normalize_pair_token(tok: str) -> str:
    tok = (tok or "").strip()
    if not tok:
        return ""
    # drop separators like '-', '/', etc.
    tok = re.sub(r"[^A-Za-z0-9]", "", tok).upper()
    # common shortcuts
    if tok in {"BTC", "XBT"}:
        return "BTCUSD"
    if tok in {"GOLD", "XAU"}:
        return "XAUUSD"
    return tok


def _filter_scorecard_pairs(scorecard_text: str, allowed_pairs: set[str]) -> str:
    """Filter the app.py scorecard text to show only selected pair blocks.

    This is a display-only filter: calculations still run for all pairs.
    """
    if not scorecard_text:
        return scorecard_text
    allowed_pairs = {str(p or "").strip().upper() for p in (allowed_pairs or set()) if p}
    if not allowed_pairs:
        return scorecard_text

    lines = scorecard_text.splitlines()
    out: list[str] = []
    i = 0

    # Keep header up to PAIRS:, but hide suggestions/top-pair list entries
    # that don't match the filtered view.
    while i < len(lines):
        ln = lines[i]
        if ln.strip() == "PAIRS:":
            out.append(ln)
            i += 1
            break

        m_sug = re.search(r"\bSuggested:\s*([A-Z0-9]{6,12})\b", ln)
        if m_sug and m_sug.group(1).upper() not in allowed_pairs:
            i += 1
            continue

        m_top = re.match(r"\s*\d+\.\s*([A-Z0-9]{6,12}):", ln)
        if m_top and m_top.group(1).upper() not in allowed_pairs:
            i += 1
            continue

        out.append(ln)
        i += 1

    # Pair blocks: start on a line like "EURUSD: BUY ..."
    kept_any = False
    keep_block = False
    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("Disclaimer:"):
            out.append(ln)
            out.extend(lines[i + 1 :])
            break

        m_pair = re.match(r"^\s*([A-Z0-9]{6,12}):\s", ln)
        if m_pair:
            pair = m_pair.group(1).upper()
            keep_block = pair in allowed_pairs
            if keep_block:
                kept_any = True
                out.append(ln)
        else:
            if keep_block:
                out.append(ln)

        i += 1

    if not kept_any:
        return (
            f"No matching pairs found in scorecard for filter: {', '.join(sorted(allowed_pairs))}.\n"
            + scorecard_text
        )

    prefix = f"[Filtered view: {', '.join(sorted(allowed_pairs))}]"
    return (prefix + "\n" + "\n".join(out)).strip()


def _run_backtest_summary(pair: str, days: str) -> str:
    """Run a concise backtest summary as a subprocess and return the text."""
    pair = pair.strip().upper()
    days = days.strip()
    proc = subprocess.run(
        [
            sys_executable(),
            os.path.join(os.path.dirname(__file__), "backtest.py"),
            "--pairs",
            pair,
            "--days",
            days,
            "--summary-suite",
            "pattern",
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        msg = f"Backtest failed for {pair} {days}."
        if err:
            msg += "\n" + err[:1200]
        return msg

    # Return just the BTC line + summary block if we can.
    lines = out.splitlines()
    keep: list[str] = []
    for ln in lines:
        if ln.strip().startswith(pair + ":"):
            keep.append(ln)
        if ln.strip() == "=== Summary ===":
            keep.append(ln)
            # include next 4 lines
            idx = lines.index(ln)
            keep.extend(lines[idx + 1 : idx + 6])
            break
    return "\n".join(keep) if keep else out


def sys_executable() -> str:
    # Prefer the same interpreter running this script.
    return os.environ.get("PYTHON", None) or os.sys.executable


HELP_TEXT = (
    "Send one of:\n"
    "  /scorecard  - get the latest live scorecard\n"
    "  /scorecard btc  - show only BTCUSD\n"
    "  /scorecard BTCUSD  - show only BTCUSD\n"
    "  /backtest BTCUSD 14d  - pattern-only backtest summary\n"
)


def main() -> int:
    load_dotenv()

    token = os.getenv("TG_TOKEN")
    if not token:
        print("Missing TG_TOKEN in environment/.env")
        return 2

    # Default allowlist: CHAT_ID if provided.
    allow_ids = set()
    chat_id_env = os.getenv("CHAT_ID")
    if chat_id_env:
        allow_ids.add(str(chat_id_env).strip())

    extra = os.getenv("ALLOWED_CHAT_IDS")
    if extra:
        for x in extra.split(","):
            if x.strip():
                allow_ids.add(x.strip())

    offset = None
    print("Telegram responder is running. Send /scorecard or /backtest <PAIR> <DAYS>.")

    while True:
        try:
            payload = tg_get_updates(token, offset=offset, timeout_s=30)
        except Exception as e:
            print(f"getUpdates failed: {e}")
            time.sleep(3)
            continue

        if not payload.get("ok"):
            print(f"Telegram API returned ok=false: {payload}")
            time.sleep(3)
            continue

        updates = payload.get("result") or []
        for upd in updates:
            try:
                offset = int(upd.get("update_id", 0)) + 1
            except Exception:
                pass

            msg = upd.get("message") or upd.get("edited_message")
            if not msg:
                continue

            chat = msg.get("chat") or {}
            chat_id = chat.get("id")
            if chat_id is None:
                continue

            # Optional allowlist
            if allow_ids and str(chat_id) not in allow_ids:
                continue

            text = (msg.get("text") or "").strip()
            if not text:
                continue

            try:
                if text.startswith("/help"):
                    tg_send(token, chat_id, HELP_TEXT)
                elif text.startswith("/scorecard"):
                    # /scorecard [btc|BTCUSD|XAUUSD|...]
                    parts = re.split(r"\s+", text)
                    allowed: set[str] = set()
                    if len(parts) >= 2:
                        for tok in parts[1:]:
                            norm = _normalize_pair_token(tok)
                            if norm:
                                allowed.add(norm)
                    sc = _run_app_scorecard(show_neutral_pairs=True if allowed else None)
                    if allowed:
                        sc = _filter_scorecard_pairs(sc, allowed_pairs=allowed)
                    tg_send(token, chat_id, sc)
                elif text.startswith("/backtest"):
                    # /backtest BTCUSD 14d
                    m = re.split(r"\s+", text)
                    if len(m) >= 3:
                        pair = m[1]
                        days = m[2]
                        bt = _run_backtest_summary(pair=pair, days=days)
                        tg_send(token, chat_id, bt)
                    else:
                        tg_send(token, chat_id, "Usage: /backtest BTCUSD 14d")
                else:
                    # Default: scorecard
                    sc = _run_app_scorecard()
                    tg_send(token, chat_id, sc)
            except Exception as e:
                try:
                    tg_send(token, chat_id, f"Error: {e}")
                except Exception:
                    pass

        # avoid tight loop when no updates
        if not updates:
            time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
