import json
import asyncio
import subprocess
from app.models.schemas import SyncMenuItem, SyncBatchRequest
from app.services.sync_service import sync_service
from app.services.vector_store import vector_store

async def run_sync():
    print("🔄 [1/3] Đang xuất dữ liệu mới nhất từ PostgreSQL RDS...")
    cmd = """PGPASSWORD='Asrp_admin^123' psql -h restaurant-db.czkw6qs4c8kh.ap-southeast-1.rds.amazonaws.com -U asrp -d restaurant-db -t -A -c "
SELECT json_agg(t) FROM (
  SELECT 
    mi.\\"Id\\" as id,
    mi.\\"Name\\" as name,
    COALESCE(mi.\\"Description\\", '') as description,
    mi.\\"BasePrice\\" as \\"basePrice\\",
    bmi.\\"CustomPrice\\" as \\"customPrice\\",
    b.\\"Id\\" as \\"branchId\\",
    b.\\"Name\\" as \\"branchName\\",
    b.\\"BrandId\\" as \\"brandId\\",
    COALESCE(c.\\"Name\\", 'Món ngon') as \\"categoryName\\",
    COALESCE(mi.\\"ImageUrl\\", b.\\"ImageUrl\\", '') as \\"imageUrl\\",
    COALESCE(mi.\\"Rating\\", 5.0) as rating,
    10 as \\"reviewCount\\",
    COALESCE(mi.\\"SoldCount\\", 20) as \\"soldCount\\",
    mi.\\"IsAvailable\\" as \\"isAvailable\\",
    bmi.\\"IsSoldOut\\" as \\"isSoldOut\\",
    COALESCE(b.\\"Address\\", '') as district,
    COALESCE(b.\\"Address\\", '') as address,
    b.\\"Latitude\\" as latitude,
    b.\\"Longitude\\" as longitude
  FROM \\"BranchMenuItems\\" bmi
  JOIN \\"Branches\\" b ON bmi.\\"BranchId\\" = b.\\"Id\\"
  JOIN \\"MenuItems\\" mi ON bmi.\\"MenuItemId\\" = mi.\\"Id\\"
  LEFT JOIN \\"Categories\\" c ON mi.\\"CategoryId\\" = c.\\"Id\\"
  WHERE b.\\"IsActive\\" = true AND bmi.\\"IsActive\\" = true AND mi.\\"IsAvailable\\" = true AND bmi.\\"IsSoldOut\\" = false
) t;
" """
    raw_json = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
    if not raw_json or raw_json == "null":
        print("❌ Không tìm thấy dữ liệu món ăn đang bán.")
        return

    data = json.loads(raw_json)
    print(f"✅ Đã tải {len(data)} món ăn từ CSDL.")

    print("🔄 [2/3] Đang tính toán Vector Embeddings và cập nhật vector_store.json...")
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

    res = await sync_service.sync_batch(SyncBatchRequest(menuItems=sync_items))
    print(f"✅ [3/3] Hoàn tất: {res.message}")
    print("🎉 Dữ liệu vector đã được cập nhật thành công vào data/vectors/vector_store.json!")

if __name__ == "__main__":
    asyncio.run(run_sync())
