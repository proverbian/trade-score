import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv


def main() -> int:
    load_dotenv()

    token = os.getenv("TG_TOKEN")
    chat_id = os.getenv("CHAT_ID")

    if not token or not chat_id:
        print("Missing TG_TOKEN and/or CHAT_ID. Set them in your environment or .env.")
        return 2

    utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    text = f"wag telegram test | {utc_now} UTC"

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=20)
    except Exception as e:
        print(f"Telegram request failed: {e}")
        return 3

    if resp.ok:
        print("Telegram test message sent successfully.")
        return 0

    # Do NOT print token; response might include debug info but safe to show a snippet.
    snippet = (resp.text or "").strip().replace("\n", " ")[:300]
    print(f"Telegram send failed: HTTP {resp.status_code} | {snippet}")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
