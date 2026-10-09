"""Reject oversized request bodies before they are buffered or parsed."""
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings
from app.core.exceptions import PayloadTooLarge

# Multipart framing overhead allowed on top of the file size.
_MULTIPART_SLACK = 64 * 1024


def _is_upload(scope) -> bool:
    return scope["method"] == "POST" and scope["path"] == f"{settings.API_V1_STR}/documents"


def _limit_for(scope) -> int:
    return settings.MAX_UPLOAD_BYTES + _MULTIPART_SLACK if _is_upload(scope) else settings.MAX_JSON_BODY_BYTES


def _error_for(scope) -> PayloadTooLarge:
    if _is_upload(scope):
        # Same message and code as the per-file check in document_service, so the user is told
        # what to do whichever check catches the file first.
        return PayloadTooLarge(f"Files must be at most {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.", code="FILE_TOO_LARGE")
    return PayloadTooLarge()


class BodySizeLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        limit = _limit_for(scope)
        content_length = dict(scope["headers"]).get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > limit:
                    return await _reject(scope, receive, send)
            except ValueError:
                return await _reject(scope, receive, send)

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    # FastAPI re-raises HTTPException from body parsing (anything else becomes a 400);
                    # the HTTP exception handler formats it with code PAYLOAD_TOO_LARGE.
                    raise StarletteHTTPException(413, _error_for(scope).message)
            return message

        await self.app(scope, limited_receive, send)


async def _reject(scope, receive, send):
    from app.middleware.error_handler import error_response

    response = error_response(_error_for(scope), request_id=scope.get("state", {}).get("request_id"))
    await response(scope, receive, send)
