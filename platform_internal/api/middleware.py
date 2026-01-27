from __future__ import annotations

import logging
import time
import uuid
from typing import Callable

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from fastapi import HTTPException

log = logging.getLogger(__name__)


class RequestIDMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid.uuid4().hex
        scope.setdefault("state", {})
        scope["state"]["request_id"] = request_id

        start = time.perf_counter()

        async def send_wrapper(message):
            if message.get("type") == "http.response.start":
                headers = message.setdefault("headers", [])
                # Add x-request-id header
                headers.append((b"x-request-id", request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            dur_ms = int((time.perf_counter() - start) * 1000)
            path = scope.get("path", "")
            method = scope.get("method", "")
            log.info("request_completed", extra={"method": method, "path": path, "duration_ms": dur_ms, "request_id": request_id})


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    req_id = getattr(getattr(request, "state", None), "request_id", None) or ""
    code = "service_unavailable" if exc.status_code == 503 else "http_error"
    body = {"error": {"code": code, "message": str(exc.detail), "request_id": req_id}}
    # Header added by RequestIDMiddleware; avoid duplicates here
    return JSONResponse(status_code=exc.status_code, content=body)


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    req_id = getattr(getattr(request, "state", None), "request_id", None) or ""
    body = {"error": {"code": "internal_error", "message": str(exc), "request_id": req_id}}
    # Header added by RequestIDMiddleware; avoid duplicates here
    return JSONResponse(status_code=500, content=body)
