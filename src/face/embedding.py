from .models import face_app
from ..utils.image_loader import load_image_from_url


async def detect_and_embed(image_url: str):
    img = await load_image_from_url(image_url)
    faces = face_app.get(img)

    if not faces:
        return None

    face = max(faces, key=lambda f: f.bbox[2] - f.bbox[0])

    return {"embedding": face.embedding.tolist(), "bbox": face.bbox.tolist()}
