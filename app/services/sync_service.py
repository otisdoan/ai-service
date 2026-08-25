from typing import List
from app.models.schemas import SyncMenuItem, SyncBranch, SyncBatchRequest, SyncResponse
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store

class SyncService:
    def __init__(self):
        pass

    async def sync_single_menu_item(self, item: SyncMenuItem) -> bool:
        """
        Ingests or updates a single menu item in the vector store with rich metadata.
        """
        price = item.customPrice if item.customPrice is not None else item.basePrice
        toppings_str = ", ".join(item.toppings) if item.toppings else "Không có"
        
        # Structured chunk for embedding
        text_chunk = (
            f"[MÓN ĂN]: {item.name}\n"
            f"[GIÁ BÁN]: {int(price):,} VNĐ\n"
            f"[DANH MỤC]: {item.categoryName}\n"
            f"[CHI NHÁNH]: {item.branchName} - Khu vực: {item.district}, {item.address}\n"
            f"[MÔ TẢ]: {item.description or 'Món ăn đặc sản thơm ngon'}\n"
            f"[TOPPING & TÙY CHỌN]: {toppings_str}"
        )

        vector = await embedding_service.get_embedding(text_chunk)
        
        metadata = {
            "menu_item_id": item.id,
            "name": item.name,
            "price": float(price),
            "base_price": float(item.basePrice),
            "custom_price": float(item.customPrice) if item.customPrice is not None else None,
            "branch_id": item.branchId,
            "branch_name": item.branchName,
            "brand_id": item.brandId,
            "category_name": item.categoryName,
            "image_url": item.imageUrl,
            "rating": item.rating,
            "review_count": item.reviewCount,
            "sold_count": item.soldCount,
            "is_available": item.isAvailable,
            "is_sold_out": item.isSoldOut,
            "district": item.district or "",
            "address": item.address or "",
            "latitude": item.latitude,
            "longitude": item.longitude,
            "toppings": item.toppings or []
        }

        doc_id = f"item_{item.branchId}_{item.id}"
        vector_store.add_or_update(
            doc_id=doc_id,
            doc_type="menu_item",
            text=text_chunk,
            vector=vector,
            metadata=metadata
        )
        return True

    async def sync_batch(self, request: SyncBatchRequest) -> SyncResponse:
        count = 0
        if request.menuItems:
            for it in request.menuItems:
                await self.sync_single_menu_item(it)
                count += 1
        
        return SyncResponse(
            success=True,
            syncedCount=count,
            message=f"Đã đồng bộ thành công {count} bản ghi vào Vector Store."
        )

    def remove_menu_item(self, branch_id: str, item_id: str):
        doc_id = f"item_{branch_id}_{item_id}"
        vector_store.remove(doc_id)

sync_service = SyncService()
