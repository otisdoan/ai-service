import json
import asyncio
from app.models.schemas import SyncMenuItem, SyncBatchRequest
from app.services.sync_service import sync_service
from app.services.vector_store import vector_store

async def main():
    with open("data/pg_dump.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Clear old stale vector store
    vector_store.documents = {}

    sync_items = []
    for item in data:
        sync_items.append(SyncMenuItem(
            id=str(item["id"]),
            name=item["name"],
            description=item.get("description") or "",
            basePrice=float(item["basePrice"]),
            customPrice=float(item["customPrice"]) if item.get("customPrice") is not None else None,
            branchId=str(item["branchId"]),
            branchName=item["branchName"],
            brandId=str(item.get("brandId") or ""),
            categoryName=item.get("categoryName") or "Món ngon",
            imageUrl=item.get("imageUrl") or "",
            rating=float(item.get("rating") or 5.0),
            reviewCount=int(item.get("reviewCount") or 10),
            soldCount=int(item.get("soldCount") or 20),
            isAvailable=True,
            isSoldOut=False,
            district=item.get("district") or "",
            address=item.get("address") or "",
            latitude=float(item["latitude"]) if item.get("latitude") is not None else None,
            longitude=float(item["longitude"]) if item.get("longitude") is not None else None
        ))

    print(f"Syncing {len(sync_items)} items from PostgreSQL into vector_store.json...")
    res = await sync_service.sync_batch(SyncBatchRequest(menuItems=sync_items))
    print("Done:", res.message)

if __name__ == "__main__":
    asyncio.run(main())
