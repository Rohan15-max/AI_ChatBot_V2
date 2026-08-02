"""
JWT secret rotation utility.

UPGRADE NOTES (this revision fixes 4 issues found in review):

1. settings.jwt_secret is a pydantic SecretStr (see config.py) — passing
   it directly to redis.setex() would store the SecretStr object's repr,
   not the actual secret string. Now uses .get_secret_value().

2. HONEST LIMITATION, now stated up front rather than implied: this script
   writes the new secret to .env, but the running Flask process holds its
   own in-memory copy via config.py's @lru_cache'd get_settings(). Editing
   .env does NOT change what an already-running process uses for signing
   or verifying tokens — the app must be restarted (or, if you're running
   in Docker, the container recreated) for the new secret to actually take
   effect. Until that restart happens, JWTs are still being signed AND
   verified with the OLD secret, so the previous version's docstring claim
   ("Old secret valid for 1 hour") was misleading: the old secret doesn't
   become invalid on any timer at all — it stays in effect indefinitely
   until you restart, at which point it stops being accepted immediately,
   not gradually over an hour.

   Storing the old secret in Redis under "jwt_old_secret" was also
   functionally inert: nothing in auth_middleware.py (or anywhere else
   in this codebase) reads that key. For Redis-backed dual-secret
   validation during a rotation window to actually work, decode_access_token()
   in auth_middleware.py would need to try the new secret first and fall
   back to checking this Redis key's value on failure — that doesn't exist
   today. This script no longer pretends otherwise; it still records the
   old secret in Redis (useful as an audit trail / manual recovery aid),
   but says plainly that this alone does not achieve zero-downtime
   rotation.

3. .env is now backed up to .env.bak-<timestamp> before being overwritten,
   so a bad rotation can be undone by hand.

4. If no line starting with "JWT_SECRET=" exists in .env at all (e.g. a
   fresh environment that's never set one), the previous version silently
   did nothing — looked successful, accomplished nothing. This version now
   appends the line if it's missing, and reports clearly which case
   occurred.

Usage:
    python rotate_secrets.py
"""

import os
import shutil
import secrets
import sys
from datetime import datetime, timezone

import redis

from config import get_settings

ENV_FILE = ".env"


def rotate_jwt_secret() -> None:
    settings = get_settings()
    r = redis.from_url(settings.redis_url)

    # FIX: .get_secret_value() — settings.jwt_secret is a SecretStr;
    # passing it directly to redis would store the wrapper object's repr
    # (something like "SecretStr('**********')"), not the actual secret.
    old_secret_value = settings.jwt_secret.get_secret_value()
    new_secret_value = secrets.token_urlsafe(32)

    # Recorded for manual audit/recovery only — see module docstring.
    # This does NOT make the old secret automatically expire on a timer;
    # nothing currently reads this key back during token validation.
    try:
        r.setex("jwt_old_secret", 3600, old_secret_value)
    except Exception as exc:
        print(f"WARNING: could not record old secret in Redis (continuing anyway): {exc}", file=sys.stderr)

    if not os.path.exists(ENV_FILE):
        print(f"ERROR: {ENV_FILE} not found in the current directory. Aborting — nothing was changed.", file=sys.stderr)
        sys.exit(1)

    # FIX: backup before overwriting, so a bad rotation can be undone.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = f"{ENV_FILE}.bak-{timestamp}"
    shutil.copy2(ENV_FILE, backup_path)

    with open(ENV_FILE, "r") as f:
        lines = f.readlines()

    found_line = False
    new_lines = []
    for line in lines:
        if line.startswith("JWT_SECRET="):
            new_lines.append(f"JWT_SECRET={new_secret_value}\n")
            found_line = True
        else:
            new_lines.append(line)

    # FIX: if JWT_SECRET= never existed in .env, append it instead of
    # silently producing a file identical to the input.
    if not found_line:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines.append("\n")
        new_lines.append(f"JWT_SECRET={new_secret_value}\n")

    with open(ENV_FILE, "w") as f:
        f.writelines(new_lines)

    print(f"Backup of previous .env saved to: {backup_path}")
    print(f"JWT_SECRET {'updated' if found_line else 'added (no existing line found)'} in {ENV_FILE}.")
    print(
        "\nIMPORTANT: the running application process(es) still hold the OLD "
        "secret in memory (config.py caches Settings via @lru_cache) and will "
        "keep signing and verifying tokens with it until restarted. This script "
        "only edits the file on disk — it does not hot-reload the app's config. "
        "Restart (or recreate, if running in Docker) every process that uses "
        "JWT_SECRET now: the Flask app, and any Celery workers if they ever "
        "decode tokens. Until you do, both old and new secrets exist but only "
        "the old one is actually in effect."
    )


if __name__ == "__main__":
    rotate_jwt_secret()