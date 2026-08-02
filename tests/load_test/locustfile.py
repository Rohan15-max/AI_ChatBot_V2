"""
Load test for /api/v1/chat and /api/v1/health.

REWRITE NOTE: the original file posted directly to `/chat` with a raw
`user_id` field and no authentication at all. The real route is
`/api/v1/chat`, requires a JWT via @require_auth, and derives the user
entirely from the token — there's no user_id field in the request body.

Each simulated Locust user now registers a real account once `on_start`
(matching how a real client would behave: log in once, then make many
authenticated requests), and reuses that token for every subsequent chat
request in the run.
"""

import uuid
from locust import HttpUser, task, between


class ChatUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        """
        Register a fresh test account once per simulated user at the start
        of the run, and store the access token for reuse. Real load
        against /api/v1/chat is meaningless without auth, since the real
        route 401s immediately otherwise — that would load-test the auth
        rejection path, not the chat pipeline this file is meant to stress.
        """
        username = f"loadtest_{uuid.uuid4().hex[:12]}"
        resp = self.client.post(
            "/api/v1/auth/register",
            json={"username": username, "password": "LoadTestPassword123!"},
        )
        if resp.status_code == 201:
            self.token = resp.json()["data"]["access_token"]
        else:
            self.token = None

    @property
    def auth_headers(self):
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    @task
    def chat(self):
        if not self.token:
            return
        self.client.post(
            "/api/v1/chat",
            json={"message": "What is the weather like for outdoor activities?", "use_grounding": False},
            headers=self.auth_headers,
        )

    @task(3)
    def health(self):
        # Unauthenticated by design — matches a real orchestrator's
        # liveness probe traffic pattern, which is why it's weighted 3x
        # more frequent than the chat task above.
        self.client.get("/api/v1/health")