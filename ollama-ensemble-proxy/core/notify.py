"""Email + webhook notifications.

Per-user prefs stored on the user record:
  - email: str
  - notification_mode: "off" | "immediate" | "digest"
  - webhook_url: str (optional)
  - webhook_secret: str (auto-generated; user can regenerate)

Daily digest queue lives in `data/notifications/digest_<user_id>.jsonl`.
A background asyncio task (digest_loop) drains and sends digests at 09:00
local time every day.

Webhook signing: header `X-AIDocGen-Signature: sha256=<hex>` over the raw body,
HMAC-SHA256 with the user's webhook_secret.
"""
import asyncio
import hashlib
import hmac
import json
import os
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Callable

import httpx

from . import audit
from .logging_config import get_logger

log = get_logger("aidocgen.notify")

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USERNAME)
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "AIDocGen")


def smtp_configured() -> bool:
    return bool(SMTP_HOST and SMTP_USERNAME and SMTP_PASSWORD and SMTP_FROM)


def _send_email_sync(to: str, subject: str, text: str, html: str | None = None) -> None:
    msg = MIMEMultipart("alternative")
    msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM}>"
    msg["To"] = to
    msg["Subject"] = subject
    msg.attach(MIMEText(text, "plain", "utf-8"))
    if html:
        msg.attach(MIMEText(html, "html", "utf-8"))
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
        s.ehlo()
        s.starttls()
        s.ehlo()
        s.login(SMTP_USERNAME, SMTP_PASSWORD)
        s.sendmail(SMTP_FROM, [to], msg.as_string())


async def send_email(to: str, subject: str, text: str, html: str | None = None) -> tuple[bool, str]:
    """Returns (ok, error_message). Never raises."""
    if not smtp_configured():
        return False, "smtp not configured"
    if not to:
        return False, "no recipient"
    try:
        await asyncio.to_thread(_send_email_sync, to, subject, text, html)
        return True, ""
    except Exception as e:
        log.warning("send_email failed", extra={"error": str(e), "to": to})
        return False, str(e)[:200]


def _sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


async def send_webhook(url: str, secret: str, payload: dict[str, Any]) -> tuple[bool, str]:
    """Returns (ok, error_message). Never raises."""
    if not url:
        return False, "no url"
    body = json.dumps(payload, ensure_ascii=False).encode()
    headers = {"Content-Type": "application/json", "User-Agent": "AIDocGen-Webhook/1.0"}
    if secret:
        headers["X-AIDocGen-Signature"] = _sign(secret, body)
    try:
        async with httpx.AsyncClient(timeout=15) as cli:
            r = await cli.post(url, content=body, headers=headers)
        if 200 <= r.status_code < 300:
            return True, ""
        return False, f"http {r.status_code}"
    except Exception as e:
        log.warning("send_webhook failed", extra={"error": str(e), "url": url})
        return False, str(e)[:200]


# --- digest queue ---
def _digest_dir(data_dir: Path) -> Path:
    p = Path(data_dir) / "notifications"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _digest_path(data_dir: Path, user_id: str) -> Path:
    return _digest_dir(data_dir) / f"digest_{user_id}.jsonl"


def queue_digest(data_dir: Path, user_id: str, entry: dict) -> None:
    p = _digest_path(Path(data_dir), user_id)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def drain_digest(data_dir: Path, user_id: str) -> list[dict]:
    p = _digest_path(Path(data_dir), user_id)
    if not p.exists():
        return []
    entries: list[dict] = []
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except Exception:
                    continue
        p.unlink()
    except Exception as e:
        log.warning("drain_digest error", extra={"error": str(e), "user_id": user_id})
    return entries


def _format_event_line(e: dict) -> str:
    kind = e.get("kind", "?")
    status = e.get("status", "?")
    job_id = e.get("job_id", "?")
    title = (e.get("title") or "").strip()
    err = (e.get("error") or "").strip()
    line = f"[{kind}] {status} — {job_id}"
    if title:
        line += f" — {title}"
    if status == "failed" and err:
        line += f"\n   ↳ {err[:200]}"
    return line


async def notify_event(data_dir: Path, user: dict, *, event: str, kind: str, status: str,
                       job_id: str, title: str = "", error: str = "",
                       details: dict | None = None) -> None:
    """Dispatch a job-completion event for a single user.

    `event` is the human label (eg. "dossier.done", "audiobook.failed").
    `kind` is the short kind ("dossier", "audiobook", "pdf", "video", "audio", "web").
    `status` is "done" or "failed".

    Behavior:
      mode=off       → nothing
      mode=immediate → email + webhook (per event)
      mode=digest    → email queued for daily 09:00 send + webhook (per event)
    """
    if not user:
        return
    mode = (user.get("notification_mode") or "off").strip()
    if mode == "off":
        return
    email = (user.get("email") or "").strip()
    webhook_url = (user.get("webhook_url") or "").strip()
    webhook_secret = user.get("webhook_secret") or ""
    user_id = user.get("id") or ""
    user_name = user.get("username") or ""

    payload = {
        "event": event,
        "kind": kind,
        "status": status,
        "job_id": job_id,
        "title": title,
        "error": error or "",
        "user_id": user_id,
        "user_name": user_name,
        "ts": int(time.time()),
    }
    if details:
        payload["details"] = details

    if webhook_url:
        ok, err = await send_webhook(webhook_url, webhook_secret, payload)
        audit.record(data_dir,
                     action="notify.webhook_sent" if ok else "notify.webhook_fail",
                     actor_id=user_id, actor_name=user_name,
                     target=job_id, kind=kind, status=status, error=err if not ok else "")

    if mode == "immediate" and email:
        subject = f"[AIDocGen] {kind} {status} — {(title or job_id)[:80]}"
        text = _format_event_line(payload) + "\n\n— AIDocGen"
        ok, err = await send_email(email, subject, text)
        audit.record(data_dir,
                     action="notify.email_sent" if ok else "notify.email_fail",
                     actor_id=user_id, actor_name=user_name,
                     target=job_id, kind=kind, status=status, error=err if not ok else "")
    elif mode == "digest" and email:
        try:
            queue_digest(data_dir, user_id, payload)
        except Exception as e:
            log.warning("queue_digest failed", extra={"error": str(e), "user_id": user_id})


async def send_user_digest(data_dir: Path, user: dict) -> bool:
    email = (user.get("email") or "").strip()
    user_id = user.get("id") or ""
    user_name = user.get("username") or ""
    if not email or not user_id:
        return False
    entries = drain_digest(data_dir, user_id)
    if not entries:
        return False
    lines = [_format_event_line(e) for e in entries]
    n = len(entries)
    subject = f"[AIDocGen] Récap journalier — {n} événement{'s' if n > 1 else ''}"
    text = "\n".join(lines) + "\n\n— AIDocGen"
    ok, err = await send_email(email, subject, text)
    audit.record(data_dir,
                 action="notify.digest_sent" if ok else "notify.digest_fail",
                 actor_id=user_id, actor_name=user_name,
                 count=n, error=err if not ok else "")
    return ok


async def digest_loop(data_dir: Path, get_users_fn: Callable[[], list[dict]]) -> None:
    """Daily 09:00 (local time) digest dispatcher. Runs forever."""
    log.info("digest loop started")
    while True:
        try:
            now = time.localtime()
            target = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, 9, 0, 0, 0, 0, -1))
            nowsec = time.time()
            if target <= nowsec + 30:
                target += 86400
            sleep_s = max(60, target - nowsec)
            await asyncio.sleep(sleep_s)
            users = get_users_fn() or []
            for u in users:
                if (u.get("notification_mode") or "off") == "digest":
                    try:
                        await send_user_digest(Path(data_dir), u)
                    except Exception as e:
                        log.warning("digest send error",
                                    extra={"error": str(e), "user_id": u.get("id")})
        except asyncio.CancelledError:
            log.info("digest loop cancelled")
            raise
        except Exception as e:
            log.warning("digest loop error", extra={"error": str(e)})
            await asyncio.sleep(300)
