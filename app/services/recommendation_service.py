import datetime
from typing import List, Dict, Any, Optional
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store
from app.models.schemas import UserTasteProfileInput, RecommendationResponse, RecommendationItem

class RecommendationService:
    def __init__(self):
        pass

    def _get_time_of_day_keywords(self) -> List[str]:
        hour = datetime.datetime.now().hour
        if 5 <= hour < 11:
            return ["phở", "bún bò", "bánh mì", "cà phê", "điểm tâm", "sáng"]
        elif 11 <= hour < 14:
            return ["cơm tấm", "cơm trưa", "bún thịt nướng", "mì ý", "trà đá"]
        elif 14 <= hour < 18:
            return ["trà sữa", "nước ép", "sinh tố", "ăn vặt", "bánh tráng"]
        else:
            return ["lẩu", "đồ nướng", "cơm gia đình", "mì xào", "bún chả"]

    async def get_personalized_recommendations(
        self,
        profile: UserTasteProfileInput,
        limit: int = 10
    ) -> RecommendationResponse:
        """
        Generates personalized recommendations based on search history, orders, cart and time context.
        """
        # 1. Synthesize user taste profile text
        taste_keywords = []
        if profile.searchHistory:
            taste_keywords.extend(profile.searchHistory[:5])
        
        if profile.recentOrders:
            for ord_item in profile.recentOrders[:5]:
                name = ord_item.get("productName") or ord_item.get("name")
                if name:
                    taste_keywords.append(name)

        if profile.cartItems:
            for c_item in profile.cartItems[:3]:
                name = c_item.get("name")
                if name:
                    taste_keywords.append(name)

        # Contextual time keywords
        time_keywords = self._get_time_of_day_keywords()
        combined_text = " ".join(taste_keywords + time_keywords)
        if not combined_text.strip():
            combined_text = "món ngon bán chạy đặc sản cơm phở bún trà sữa"

        # 2. Get user taste profile vector
        user_vector = await embedding_service.get_embedding(combined_text)

        # 3. Retrieve candidates
        candidates = vector_store.search_hybrid(
            query_vector=user_vector,
            doc_type="menu_item",
            top_k=limit * 3,
            filters={"is_available": True, "is_sold_out": False}
        )

        # 4. Multi-factor Scoring and Re-ranking
        ranked_items: List[RecommendationItem] = []
        for cand in candidates:
            meta = cand["metadata"]
            sim_score = cand["score"]
            rating = meta.get("rating", 5.0)
            sold_count = meta.get("sold_count", 0)
            price = meta.get("price", 0.0)
            name = meta.get("name", "")

            # Re-rank formula:
            # 60% semantic similarity + 20% popularity + 10% rating bonus + 10% time bonus
            time_bonus = 0.1 if any(tk in name.lower() for tk in time_keywords) else 0.0
            pop_bonus = min(0.2, (sold_count / 100.0) * 0.2)
            rating_bonus = (rating / 5.0) * 0.1
            
            final_score = (sim_score * 0.6) + pop_bonus + rating_bonus + time_bonus

            # Determine human-friendly reason
            if any(sh.lower() in name.lower() for sh in (profile.searchHistory or [])):
                reason = "Dựa trên lịch sử tìm kiếm gần đây của bạn"
                tag = "Dành cho bạn"
            elif any(tk in name.lower() for tk in time_keywords):
                reason = "Gợi ý phù hợp cho bữa ăn thời điểm này"
                tag = "Món ngon đúng giờ"
            elif sold_count > 50:
                reason = f"Đã bán {sold_count}+ phần, rất được ưa chuộng"
                tag = "Bán chạy nhất"
            else:
                reason = f"Đánh giá cao {rating}⭐ từ khách hàng"
                tag = "Được yêu thích"

            price_text = f"{int(price):,}đ".replace(",", ".")

            ranked_items.append(RecommendationItem(
                menuItemId=str(meta.get("menu_item_id", "")),
                branchId=str(meta.get("branch_id", "")),
                dishName=name,
                storeName=meta.get("branch_name", "DineX Branch"),
                priceAmount=price,
                priceText=price_text,
                imageUrl=meta.get("image_url", ""),
                rating=rating,
                reviews=meta.get("review_count", 0),
                score=round(final_score, 3),
                reason=reason,
                tag=tag,
                distance="1.0 km"
            ))

        ranked_items.sort(key=lambda x: x.score, reverse=True)
        return RecommendationResponse(
            title="Được đề xuất cho bạn",
            items=ranked_items[:limit]
        )

recommendation_service = RecommendationService()
