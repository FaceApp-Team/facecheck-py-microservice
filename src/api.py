import logging

from fastapi import APIRouter, HTTPException

from .face.embedding import detect_and_embed
from .qdrant.client import save_embedding, search_embedding

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/enroll")
async def enroll(user_id: str, image_url: str):

    faces = await detect_and_embed(image_url)

    if not faces:
        raise HTTPException(404, "No valid face detected")

    for face in faces:
        save_embedding(user_id, face["embedding"])

    return {
        "status": "enrolled",
        "embeddings_saved": len(faces),
    }


@router.post("/recognize")
async def recognize(image_url: str):
    faces = await detect_and_embed(image_url)

    if not faces:
        return {"match": False}

    for face in faces:
        result = search_embedding(face["embedding"])
        if result.points:
            best = result.points[0]
            return {
                "match": True,
                "user_id": best.payload["user_id"],
                "score": best.score,
            }

    return {"match": False}
