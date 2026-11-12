"""Access control for state-changing endpoints (approve, feedback, policy reset, replay start).

These actions change the ledger and the learned policy, and feedback accepts the "true
label" from the caller, so they must not be reachable by anyone who can see the port.

* Default: allowed only from the local machine (the dashboard runs on localhost).
* If ZEROSHIELD_ADMIN_TOKEN is set, a matching ``X-Admin-Token`` header is required instead,
  which allows controlled remote use by API clients.
"""
import hmac
import os

from fastapi import HTTPException, Request

LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


def require_operator(request: Request) -> None:
    token = os.getenv("ZEROSHIELD_ADMIN_TOKEN")
    if token:
        supplied = request.headers.get("x-admin-token", "")
        if not hmac.compare_digest(supplied.encode(), token.encode()):
            raise HTTPException(401, "A valid X-Admin-Token header is required.")
        return
    host = request.client.host if request.client else ""
    if host not in LOCAL_HOSTS:
        raise HTTPException(403, "Operator actions are limited to localhost. "
                                 "Set ZEROSHIELD_ADMIN_TOKEN to allow token-based remote use.")
