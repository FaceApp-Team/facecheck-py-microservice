import logging

from fastapi import APIRouter, HTTPException

from .face.embedding import detect_and_embed
from .qdrant.client import save_embedding, search_embedding

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/enroll")
async def enroll(user_id: str, image_url: str):
    try:
        data = await detect_and_embed(image_url)
    except Exception as exc:  # surface network/decoding/model errors
        logger.exception("Image processing failed")
        raise HTTPException(status_code=400, detail=f"Image processing failed: {exc}")

    if not data:
        raise HTTPException(status_code=404, detail="No face detected")

    try:
        save_embedding(user_id, data["embedding"])
    except Exception as exc:  # e.g. Qdrant connectivity/credentials
        logger.exception("Failed to save embedding")
        raise HTTPException(status_code=502, detail=f"Failed to save embedding: {exc}")

    return {"status": "enrolled"}


@router.post("/recognize")
async def recognize(image_url: str):
    try:
        data = await detect_and_embed(image_url)
    except Exception as exc:
        logger.exception("Image processing failed")
        raise HTTPException(status_code=400, detail=f"Image processing failed: {exc}")

    if not data:
        return {"match": False}

    try:
        result = search_embedding(data["embedding"])
    except Exception as exc:
        logger.exception("Failed to search embedding")
        raise HTTPException(
            status_code=502, detail=f"Failed to search embedding: {exc}"
        )

    return result
