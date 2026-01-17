from fastapi import FastAPI
from .api import router
from .qdrant.client import create_collection_if_not_exists

app = FastAPI(title="Face Recognition Service")

# create collection on startup
@app.on_event("startup")
async def startup_event():
    create_collection_if_not_exists()


app.include_router(router)
