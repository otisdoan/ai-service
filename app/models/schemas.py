from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# ================= CHAT & RAG MODELS =================

class ChatMessageItem(BaseModel):
    role: str  # "user" | "model" | "assistant"
    content: str

class MenuContextItem(BaseModel):
    name: str
    price: float
    description: Optional[str] = ""
    toppings: Optional[List[str]] = []

class PythonChatRequest(BaseModel):
    message: str
    branchId: Optional[str] = "00000000-0000-0000-0000-000000000000"
    sessionId: Optional[str] = "default_session"
    chatHistory: Optional[List[ChatMessageItem]] = []
    menuContext: Optional[List[MenuContextItem]] = []
    userLocation: Optional[Dict[str, float]] = None  # {"lat": 10.7769, "lng": 106.7009}
    userId: Optional[str] = None

class BranchRecommendationCard(BaseModel):
    branchId: str
    menuItemId: Optional[str] = None
    storeName: str
    dishName: str
    priceAmount: float
    priceText: str
    rating: float = 4.8
    reviews: int = 120
    deliveryTime: str = "20-30 phút"
    distance: str = "1.5 km"
    imageUrl: Optional[str] = ""
    tag: Optional[str] = "Gợi ý hàng đầu"

class OrderDraftItem(BaseModel):
    menuItemId: Optional[str] = None
    name: str
    quantity: int = 1
    price: float = 0.0
    notes: Optional[str] = None
    toppings: Optional[List[str]] = []

class OrderDraft(BaseModel):
    branchId: Optional[str] = None
    items: List[OrderDraftItem] = []
    subtotal: float = 0.0
    discountAmount: float = 0.0
    finalAmount: float = 0.0
    pickupTime: Optional[str] = None
    notes: Optional[str] = None

class PythonChatResponse(BaseModel):
    reply: str
    orderDraft: Optional[OrderDraft] = None
    recommendations: Optional[List[str]] = []
    branchRecommendations: Optional[List[BranchRecommendationCard]] = []

# ================= RECOMMENDATION MODELS =================

class UserTasteProfileInput(BaseModel):
    userId: Optional[str] = None
    searchHistory: Optional[List[str]] = []
    recentOrders: Optional[List[Dict[str, Any]]] = []
    cartItems: Optional[List[Dict[str, Any]]] = []
    favoriteBranchIds: Optional[List[str]] = []
    userLocation: Optional[Dict[str, float]] = None

class RecommendationItem(BaseModel):
    menuItemId: str
    branchId: str
    dishName: str
    storeName: str
    priceAmount: float
    priceText: str
    imageUrl: Optional[str] = ""
    rating: float = 5.0
    reviews: int = 0
    score: float = 0.0
    reason: str = "Gợi ý phù hợp khẩu vị của bạn"
    tag: str = "Được đề xuất"
    distance: Optional[str] = "1.0 km"

class RecommendationResponse(BaseModel):
    title: str = "Được đề xuất cho bạn"
    items: List[RecommendationItem] = []

# ================= DATA SYNC MODELS =================

class SyncMenuItem(BaseModel):
    id: str
    name: str
    description: Optional[str] = ""
    basePrice: float
    customPrice: Optional[float] = None
    branchId: str
    branchName: str
    brandId: Optional[str] = None
    categoryName: Optional[str] = ""
    imageUrl: Optional[str] = ""
    rating: float = 5.0
    reviewCount: int = 0
    soldCount: int = 0
    isAvailable: bool = True
    isSoldOut: bool = False
    toppings: Optional[List[str]] = []
    district: Optional[str] = ""
    address: Optional[str] = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class SyncBranch(BaseModel):
    id: str
    brandId: str
    name: str
    address: str
    district: Optional[str] = ""
    city: Optional[str] = "TP. Hồ Chí Minh"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    isActive: bool = True
    imageUrl: Optional[str] = ""

class SyncBatchRequest(BaseModel):
    menuItems: Optional[List[SyncMenuItem]] = []
    branches: Optional[List[SyncBranch]] = []

class SyncResponse(BaseModel):
    success: bool
    syncedCount: int
    message: str
