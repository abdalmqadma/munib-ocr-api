import asyncio
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from firebase_admin import auth, credentials, get_app, initialize_app

import fast_ocr

logger = logging.getLogger("munib_ocr.security")

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(4 * 1024 * 1024)))
OCR_TIMEOUT_SECONDS = float(os.getenv("OCR_TIMEOUT_SECONDS", "20"))
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "6"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "application/octet-stream"}
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

_rate_lock = threading.Lock()
_rate_buckets = defaultdict(deque)


def _init_firebase():
    try:
        get_app()
        return
    except ValueError:
        pass

    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if service_account_json:
        try:
            initialize_app(credentials.Certificate(json.loads(service_account_json)))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Invalid Firebase service account configuration") from exc
    else:
        initialize_app()


_init_firebase()
app = FastAPI(title="Munib Imsakia API")


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "munib-imsakia",
        "ocr_backend": "tesseract-grid-viterbi",
    }


def _bearer_token(authorization):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authentication required")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    return token.strip()


async def verify_firebase_user(authorization: str | None = Header(default=None)):
    token = _bearer_token(authorization)
    try:
        decoded = await asyncio.to_thread(
            lambda: auth.verify_id_token(token, check_revoked=True)
        )
    except Exception:
        logger.info("Firebase token verification failed")
        raise HTTPException(status_code=401, detail="Invalid or expired authentication token")
    if not decoded.get("uid"):
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return decoded


def _enforce_rate_limit(uid, request):
    now = time.monotonic()
    client_ip = request.client.host if request.client else "unknown"
    key = f"{uid}:{client_ip}"
    with _rate_lock:
        bucket = _rate_buckets[key]
        cutoff = now - RATE_LIMIT_WINDOW_SECONDS
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_REQUESTS:
            retry_after = max(1, int(RATE_LIMIT_WINDOW_SECONDS - (now - bucket[0])))
            raise HTTPException(
                status_code=429,
                detail="Too many extraction requests",
                headers={"Retry-After": str(retry_after)},
            )
        bucket.append(now)


def _valid_image(contents, content_type):
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        return False
    return contents.startswith(JPEG_MAGIC) or contents.startswith(PNG_MAGIC)


def _run_fast_extract(file):
    return asyncio.run(fast_ocr.extract_imsakia(file))


@app.post("/extract")
async def secure_extract(
    request: Request,
    file: UploadFile = File(...),
    firebase_user=Depends(verify_firebase_user),
):
    _enforce_rate_limit(firebase_user["uid"], request)

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_UPLOAD_BYTES + 256 * 1024:
                raise HTTPException(status_code=413, detail="Image is too large")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header")

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large")
    if not _valid_image(contents, file.content_type):
        raise HTTPException(status_code=415, detail="Only JPEG and PNG images are supported")

    await file.seek(0)
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_run_fast_extract, file),
            timeout=OCR_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("OCR timeout uid=%s", firebase_user["uid"])
        raise HTTPException(status_code=504, detail="Image processing timed out")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled OCR extraction error")
        raise HTTPException(status_code=500, detail="Image processing failed")
