"""
Development/test data seeder.

REWRITE NOTE: the original version targeted the dead parallel app —
`from database import SessionLocal` (instantiated directly, not via the
context manager) and `from models import User, Conversation, Message`.
The real database.py has no `models` module to import from (the models
live directly in database.py), there is no `Conversation` model (the
real equivalent is `Thread`), and the correct session pattern is the
`get_db()` context manager, not a bare `SessionLocal()` you have to
remember to commit/close yourself.

Field names are also corrected to match the real models: User has
`username`/`password_hash` (not `email`-as-identity/`hashed_password`),
Thread has `user_id`/`title`/`mode` (not Conversation's `user_id`/`title`
without mode), and Message has `thread_id`/`role`/`content`/`model_used`
(not `conversation_id`).

Passwords are now hashed with auth_middleware.py's real hash_password()
instead of a literal "fake_hash" string — seeded users couldn't actually
log in through the real /api/v1/auth/login route otherwise, since
verify_password() does a real bcrypt check against whatever's stored.

This script is synchronous throughout — the original wrapped everything
in `asyncio.run()` for no reason; nothing in database.py's session API or
auth_middleware.py's hashing is async, so there was no actual concurrency
being achieved, just unnecessary event-loop overhead.
"""

import random
import sys

from database import get_db, User, Thread, Message, utc_now
from auth_middleware import hash_password
from security.pii_redactor import redact_pii

SEEDED_USERNAME_PREFIX = "seed_test_user_"


def seed_users(n: int = 10) -> list:
    """
    Create n test users with real (bcrypt-hashed) passwords so they can
    actually authenticate through /api/v1/auth/login afterward.

    Returns the list of created user IDs.
    """
    created_ids = []
    with get_db() as db:
        for i in range(n):
            username = f"{SEEDED_USERNAME_PREFIX}{i}"
            existing = db.query(User).filter(User.username == username).first()
            if existing:
                print(f"Skipping {username} — already exists.")
                created_ids.append(existing.id)
                continue

            user = User(
                username=username,
                email=f"{username}@example.test",
                password_hash=hash_password("SeedTestPassword123!"),
                display_name=username,
                tier="free",
                is_active=True,
            )
            db.add(user)
            db.flush()  # populate user.id before the session closes
            created_ids.append(user.id)

    print(f"Seeded {len(created_ids)} users (prefix: {SEEDED_USERNAME_PREFIX}).")
    return created_ids


def seed_threads_and_messages(user_ids: list) -> None:
    """
    For each given user, create 1-5 threads, each with 2-10 messages.
    Message content is passed through redact_pii() — not because seeded
    test content is expected to contain real PII, but to exercise the same
    code path real conversation storage uses, catching any signature
    drift in redact_pii() itself as a side effect of running this script.
    """
    thread_count = 0
    message_count = 0

    with get_db() as db:
        for user_id in user_ids:
            for t in range(random.randint(1, 5)):
                mode = random.choice(["chat", "rag"])
                thread = Thread(
                    user_id=user_id,
                    title=f"Seeded test thread {t}",
                    mode=mode,
                )
                db.add(thread)
                db.flush()  # populate thread.id
                thread_count += 1

                n_messages = random.randint(2, 10)
                for m in range(n_messages):
                    role = "user" if m % 2 == 0 else "assistant"
                    content = redact_pii(f"This is seeded test message #{m} in thread {t}.")
                    db.add(Message(
                        thread_id=thread.id,
                        role=role,
                        content=content,
                        model_used="seed-script" if role == "assistant" else None,
                    ))
                    message_count += 1

                thread.message_count = n_messages
                thread.updated_at = utc_now()

    print(f"Seeded {thread_count} threads and {message_count} messages.")


def main():
    if "--confirm" not in sys.argv:
        print(
            "This will insert test data into whatever database your .env currently "
            "points at. If that's a production database, this is almost certainly "
            "not what you want.\n"
            "Re-run with --confirm to proceed: python seed_test_data.py --confirm"
        )
        sys.exit(1)

    user_ids = seed_users(n=10)
    seed_threads_and_messages(user_ids)


if __name__ == "__main__":
    main()