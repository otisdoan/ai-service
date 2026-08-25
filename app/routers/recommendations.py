from fastapi import APIRouter
from app.models.schemas import UserTasteProfileInput, RecommendationResponse
from app.services.recommendation_service import recommendation_service

router = APIRouter(prefix="/api/v1/recommendations", tags=["Recommendations"])

@router.post("/personalized", response_model=RecommendationResponse)
async def get_personalized_recommendations(profile: UserTasteProfileInput, limit: int = 10):
    """
    Generates personalized recommendations based on search history, orders, cart items and context.
    """
    return await recommendation_service.get_personalized_recommendations(profile, limit=limit)
