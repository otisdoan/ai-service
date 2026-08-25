from fastapi import APIRouter
from app.models.schemas import SyncMenuItem, SyncBatchRequest, SyncResponse
from app.services.sync_service import sync_service

router = APIRouter(prefix="/api/v1/sync", tags=["Data Sync"])

@router.post("/menu-item", response_model=SyncResponse)
async def sync_menu_item(item: SyncMenuItem):
    """
    Syncs a single menu item to vector database in real-time.
    """
    await sync_service.sync_single_menu_item(item)
    return SyncResponse(
        success=True,
        syncedCount=1,
        message=f"Đã cập nhật vector thành công cho món '{item.name}'."
    )

@router.post("/batch", response_model=SyncResponse)
async def sync_batch(request: SyncBatchRequest):
    """
    Batch synchronizes menu items and branches into vector database.
    """
    return await sync_service.sync_batch(request)

@router.delete("/menu-item/{branch_id}/{item_id}")
async def delete_menu_item(branch_id: str, item_id: str):
    """
    Removes a menu item from the vector database.
    """
    sync_service.remove_menu_item(branch_id, item_id)
    return {"success": True, "message": "Đã xóa món khỏi Vector Store."}
