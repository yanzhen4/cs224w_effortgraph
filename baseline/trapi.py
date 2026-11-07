"""TRAPI Responses client with the safeguards this machine's runs need.

Entra ID bearer token, one stable X-TRAPI-Workload-ID per process and model. The endpoint and token scope come from
TRAPI_ENDPOINT and TRAPI_SCOPE, set in the environment or in the untracked .env at the repo root (see .env.example).
TRAPI blocks a workload (403 retry-flood) after a handful of throttled retries, so:
  - a per-model token bucket paces requests by what they reserve (prompt estimate + max output);
  - each response's x-ratelimit-remaining/reset-tokens headers make later requests wait for the reset;
  - a 429 is retried at most RATE_LIMIT_RETRIES times, honouring Retry-After;
  - RAW_429_LIMIT 429s within RAW_429_WINDOW_S, or any 403 block, stop every request in the process and write a
    hold file that keeps new runs from starting for HOLD_HOURS.
"""
import json
import os
import random
import re
import threading
import time
from pathlib import Path

RATE_LIMIT_RETRIES = 2
OTHER_RETRIES = 3
RAW_429_LIMIT, RAW_429_WINDOW_S = 3, 900
HOLD_HOURS = float(os.environ.get("EG_HOLD_HOURS", "2.0"))
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
HOLD = RUNS / ".trapi_hold"
STOP = threading.Event()
_RETRY_AFTER = re.compile(r"(?:try again in|retry after)\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds?|s|seconds?)?", re.I)


def setting(name):
    """A TRAPI setting from the environment, else from KEY=VALUE lines in the repo's .env."""
    if os.environ.get(name):
        return os.environ[name]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == name and value.strip():
                return value.strip()
    raise SystemExit(f"{name} is not set (environment or {env}); see .env.example")


class Stopped(RuntimeError):
    """Requests were stopped in this process (TRAPI block or the 429 breaker)."""


class CallFailed(RuntimeError):
    pass


def hold(reason):
    STOP.set()
    RUNS.mkdir(parents=True, exist_ok=True)
    HOLD.write_text(json.dumps({"ts": time.time(), "reason": str(reason)[:400]}) + "\n")


def check_hold():
    if not HOLD.exists():
        return
    h = json.loads(HOLD.read_text())
    left = HOLD_HOURS * 3600 - (time.time() - h["ts"])
    if left > 0:
        raise SystemExit(f"TRAPI hold ({h['reason']}); no run may start for another {left / 60:.0f} min "
                         f"(delete {HOLD} to override)")


class Lock:
    """One run per model deployment at a time: runs/.trapi.<deployment>.lock."""

    def __init__(self, deployment, name):
        self.path, self.name = RUNS / f".trapi.{deployment}.lock", name

    def __enter__(self):
        import fcntl
        RUNS.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a+")
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.f.seek(0)
            raise SystemExit(f"another run holds {self.path} ({self.f.read().strip()}); not starting {self.name}")
        self.f.seek(0)
        self.f.truncate()
        self.f.write(f"{self.name} pid {os.getpid()}\n")
        self.f.flush()
        return self

    def __exit__(self, *a):
        self.f.close()


class Bucket:
    """Tokens per minute this process may reserve on one deployment, plus the deployment's own reported quota."""

    def __init__(self, per_min):
        self.cap, self.tokens, self.t = per_min, per_min / 4, time.monotonic()
        self.remaining, self.reset_at, self.seen = None, 0.0, 0.0
        self.lock = threading.Lock()

    def note(self, headers):
        try:
            remaining = int(headers.get("x-ratelimit-remaining-tokens"))
            reset = float(headers.get("x-ratelimit-reset-tokens") or 0)
        except (TypeError, ValueError):
            return
        with self.lock:
            self.remaining, self.reset_at, self.seen = remaining, time.time() + reset, time.time()

    def take(self, n):
        n = min(n, self.cap)
        while True:
            if STOP.is_set():
                raise Stopped("requests stopped in this process; none sent")
            with self.lock:
                now = time.time()
                if self.remaining is not None and now - self.seen < 60 and now < self.reset_at and self.remaining < 2 * n:
                    wait = self.reset_at - now + random.uniform(0.5, 3.0)
                else:
                    mono = time.monotonic()
                    self.tokens = min(self.cap, self.tokens + (mono - self.t) * self.cap / 60)
                    self.t = mono
                    if self.tokens >= n:
                        self.tokens -= n
                        return
                    wait = (n - self.tokens) * 60 / self.cap
            time.sleep(min(wait, 5))


_429s, _429_lock = [], threading.Lock()


def note_429(tag):
    now = time.time()
    with _429_lock:
        _429s[:] = [t for t in _429s if now - t < RAW_429_WINDOW_S] + [now]
        n = len(_429s)
    if n >= RAW_429_LIMIT:
        hold(f"429 breaker: {n} rate-limit responses within {RAW_429_WINDOW_S // 60} min (last: {tag})")


def retry_after(exc, attempt):
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    for key in ("retry-after-ms", "retry-after"):
        if headers.get(key):
            try:
                return float(headers[key]) / (1000 if key.endswith("ms") else 1)
            except ValueError:
                pass
    m = _RETRY_AFTER.search(str(exc))
    if m:
        value = float(m.group(1))
        return value / 1000 if (m.group(2) or "s").lower().startswith("m") else value
    return min(120, 15 * 2 ** attempt) + random.uniform(0, 5)


def make_client(workload):
    from azure.identity import AzureCliCredential, ChainedTokenCredential, ManagedIdentityCredential, get_bearer_token_provider
    from openai import OpenAI
    token = get_bearer_token_provider(ChainedTokenCredential(AzureCliCredential(), ManagedIdentityCredential()), setting("TRAPI_SCOPE"))
    return OpenAI(base_url=setting("TRAPI_ENDPOINT"), api_key=token, max_retries=0, timeout=1800,
                  default_headers={"X-TRAPI-Workload-ID": workload})


def wait_background(client, r, poll_s=15, max_errors=8):
    """Poll a background response (GET /responses/{id}) until it is no longer queued or in progress."""
    errors = 0
    while r.status in ("queued", "in_progress"):
        time.sleep(poll_s)
        try:
            r = client.responses.retrieve(r.id)
            errors = 0
        except Exception as exc:  # noqa: BLE001 - transient poll errors are retried
            if getattr(exc, "status_code", None) not in (None, 500, 502, 503, 504) or errors >= max_errors:
                raise
            errors += 1
    if r.status in ("failed", "cancelled"):
        raise RuntimeError(f"background response {r.status}: {r.error}")
    return r


def respond(client, bucket, deployment, prompt, effort, max_output_tokens, tag="", background=False):
    """One /v1/responses call. Returns a dict with the text, usage, status and attempt log.

    The gateway answers a call with 504 once it has run ~240 s, streamed or not; background=True submits the request
    and polls for its result, so no single HTTP request runs that long.
    """
    reserve = len(prompt) // 3 + max_output_tokens
    attempts, rate_limits, errors = [], 0, 0
    while True:
        bucket.take(reserve)
        started = time.time()
        try:
            raw = client.responses.with_raw_response.create(
                model=deployment, input=[{"role": "user", "content": prompt}],
                reasoning={"effort": effort}, max_output_tokens=max_output_tokens, **({"background": True} if background else {}))
            bucket.note(raw.headers)
            r = wait_background(client, raw.parse()) if background else raw.parse()
            usage = r.usage.model_dump() if r.usage else {}
            return {"text": r.output_text, "status": r.status,
                    "incomplete_reason": getattr(r.incomplete_details, "reason", None) if r.incomplete_details else None,
                    "usage": usage, "response_id": r.id, "latency_s": round(time.time() - started, 2),
                    "attempts": attempts + [{"ok": True, "ts": started}]}
        except Exception as exc:  # noqa: BLE001 - classified below
            status = getattr(exc, "status_code", None)
            msg = f"{status} {str(exc)[:300]}"
            if status == 429:  # TRAPI's 429 body is empty; the headers say which limit was hit
                headers = getattr(getattr(exc, "response", None), "headers", None) or {}
                msg += f" {dict((k, v) for k, v in headers.items() if 'ratelimit' in k or 'retry' in k or k.startswith('x-ms'))}"
            attempts.append({"ok": False, "ts": started, "error": msg})
            if status == 403 or "PrincipalBlocked" in str(exc) or "retry-flood" in str(exc):
                hold(f"403 block on {deployment}: {msg}")
                raise Stopped(msg) from exc
            if status == 429:
                note_429(tag)
                rate_limits += 1
                if STOP.is_set():
                    raise Stopped(msg) from exc
                if rate_limits > RATE_LIMIT_RETRIES:
                    raise CallFailed(f"rate-limited {rate_limits} times: {msg}") from exc
                time.sleep(retry_after(exc, rate_limits))
                continue
            if status is not None and status not in (408, 409, 500, 502, 503, 504):
                raise CallFailed(msg) from exc
            errors += 1
            if errors > OTHER_RETRIES:
                raise CallFailed(f"{errors} transport errors: {msg}") from exc
            time.sleep(min(120, 10 * 2 ** errors) + random.uniform(0, 5))
