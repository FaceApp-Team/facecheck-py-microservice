from qdrant_client import QdrantClient, models
from ..config import settings
from qdrant_client.http.models import Distance, VectorParams
import uuid

qdrant = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


def save_embedding(user_id: str, embedding: list):
    data = qdrant.upsert(
        collection_name="face_embeddings",
        points=[
            {
                "id": str(uuid.uuid4()),  # or use Prisma user ID
                "vector": {"face": embedding},  # this is the key Qdrant expects
                "payload": {"user_id": user_id},
            }
        ],
    )


def create_collection_if_not_exists():
    collection_name = "face_embeddings"

    try:
        existing = qdrant.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists.")
        return
    except Exception:
        # collection does not exist
        pass

    qdrant.recreate_collection(
        collection_name="face_embeddings",
        vectors_config={
            "face": models.VectorParams(size=512, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={"sparse-vector": models.SparseVectorParams()},
    )


def search_embedding(embedding: list):
    result = qdrant.query_points(
        collection_name="face_embeddings",
        query=embedding,  # vector to compare
        limit=1,
        using="face",  # must match collection config
        score_threshold=0.75,
    )
    return result
