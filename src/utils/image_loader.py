import httpx
import cv2
import numpy as np


async def load_image_from_url(url: str) -> np.ndarray:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        image_data = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image from the provided URL.")
        return image
