import asyncio
import httpx
from app.models.schemas import SyncMenuItem, SyncBatchRequest
from app.services.sync_service import sync_service

BACKEND_URL = "http://localhost:5100"

async def sync_database_to_vectors():
    print("[Sync] Fetching all branches and menu items from .NET backend...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Fetch branches
        resp = await client.get(f"{BACKEND_URL}/api/branches?pageSize=100")
        if resp.status_code != 200:
            print(f"[Sync] Failed to fetch branches: {resp.status_code}")
            return

        data = resp.json()
        branches = data.get("items", [])
        print(f"[Sync] Found {len(branches)} branches.")

        sync_items = []
        for b in branches:
            branch_id = b["id"]
            branch_name = b["name"]
            category_name = b.get("category") or "Món ngon"
            district = b.get("address") or ""
            lat = b.get("latitude")
            lng = b.get("longitude")

            # Fetch branch menu items
            menu_resp = await client.get(f"{BACKEND_URL}/api/branches/{branch_id}/menu-items")
            if menu_resp.status_code == 200:
                menu_data = menu_resp.json()
                items = menu_data if isinstance(menu_data, list) else menu_data.get("items", [])
                for mi in items:
                    item_id = mi.get("menuItemId") or mi.get("id")
                    name = mi.get("name") or mi.get("menuItemName") or "Món ăn"
                    base_price = float(mi.get("basePrice") or mi.get("price") or 35000.0)
                    custom_price = float(mi.get("customPrice")) if mi.get("customPrice") else None

                    sync_items.append(SyncMenuItem(
                        id=str(item_id),
                        name=name,
                        description=mi.get("description") or "",
                        basePrice=base_price,
                        customPrice=custom_price,
                        branchId=str(branch_id),
                        branchName=branch_name,
                        brandId=b.get("brandId"),
                        categoryName=category_name,
                        imageUrl=mi.get("imageUrl") or b.get("imageUrl") or "",
                        rating=float(b.get("averageRating") or 4.8),
                        reviewCount=int(b.get("reviewCount") or 10),
                        soldCount=int(mi.get("soldCount") or 25),
                        isAvailable=True,
                        isSoldOut=False,
                        district=district,
                        address=district,
                        latitude=lat,
                        longitude=lng
                    ))

        if sync_items:
            print(f"[Sync] Ingesting {len(sync_items)} menu items into Vector Database...")
            res = await sync_service.sync_batch(SyncBatchRequest(menuItems=sync_items))
            print(f"[Sync] {res.message}")
        else:
            print("[Sync] No items found across branches.")

if __name__ == "__main__":
    asyncio.run(sync_database_to_vectors())
