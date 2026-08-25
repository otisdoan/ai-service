from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.routers import chat, recommendations, sync

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="ASRP DineX AI Microservice for RAG, Conversational Ordering, and Hybrid Recommendations."
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(chat.router)
app.include_router(chat.router, prefix="/api/ai")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(recommendations.router)
app.include_router(recommendations.router, prefix="/api/ai")
app.include_router(sync.router)
app.include_router(sync.router, prefix="/api/ai")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
