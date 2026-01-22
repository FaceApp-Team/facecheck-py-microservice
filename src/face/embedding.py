from .models import face_app
from ..utils.image_loader import load_image_from_url
import cv2
import numpy as np


async def detect_and_embed(image_url: str):
    img = await load_image_from_url(image_url)

    # normalize color space
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = face_app.get(img)

    if not faces:
        return None

    results = []

    for face in faces:
        w = face.bbox[2] - face.bbox[0]
        h = face.bbox[3] - face.bbox[1]

        # quality gates
        if face.det_score < 0.6:
            continue
        if min(w, h) < 80:
            continue

        emb = face.embedding
        emb = emb / np.linalg.norm(emb)  # enforce normalization

        results.append(
            {
                "embedding": emb.tolist(),
                "bbox": face.bbox.tolist(),
                "det_score": float(face.det_score),
            }
        )

    return results or None
