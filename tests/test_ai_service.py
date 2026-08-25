import asyncio
import pytest
from app.models.schemas import (
    SyncMenuItem,
    SyncBatchRequest,
    UserTasteProfileInput,
    PythonChatRequest
)
from app.services.sync_service import sync_service
from app.services.rag_search_service import rag_search_service
from app.services.order_agent_service import order_agent_service
from app.services.recommendation_service import recommendation_service

@pytest.mark.asyncio
async def test_multi_criteria_rag_search_under_50k():
    # 1. Seed test menu items
    branch1_id = "11111111-1111-1111-1111-111111111111"
    branch2_id = "22222222-2222-2222-2222-222222222222"

    items = [
        SyncMenuItem(
            id="item-pho-45k",
            name="Phở bò tái nạm",
            basePrice=45000.0,
            branchId=branch1_id,
            branchName="DineX Phở Quận 1",
            categoryName="Món nước",
            district="Quận 1",
            rating=4.9,
            soldCount=150,
            isAvailable=True,
            isSoldOut=False
        ),
        SyncMenuItem(
            id="item-pho-75k",
            name="Phở bò thố đá đặc biệt",
            basePrice=75000.0,
            branchId=branch2_id,
            branchName="DineX Phở Premium",
            categoryName="Món nước",
            district="Bình Thạnh",
            rating=4.8,
            soldCount=80,
            isAvailable=True,
            isSoldOut=False
        ),
        SyncMenuItem(
            id="item-com-40k",
            name="Cơm tấm sườn nướng",
            basePrice=40000.0,
            branchId=branch1_id,
            branchName="DineX Cơm Tấm",
            categoryName="Cơm",
            district="Quận 1",
            rating=4.7,
            soldCount=90,
            isAvailable=True,
            isSoldOut=False
        ),
    ]

    await sync_service.sync_batch(SyncBatchRequest(menuItems=items))

    # 2. Query: "Có chi nhánh nào bán phở giá dưới 50k không?"
    query = "Có chi nhánh nào bán phở giá dưới 50k không?"
    reply, cards, chips = await rag_search_service.search_branches_and_dishes(query)

    # 3. Assertions
    assert len(cards) > 0
    for card in cards:
        assert "phở" in card.dishName.lower()
        assert card.priceAmount <= 50000.0
    
    assert any(c.dishName == "Phở bò tái nạm" for c in cards)
    assert any(c.priceAmount == 45000.0 for c in cards)

@pytest.mark.asyncio
async def test_order_agent_slot_filling():
    msg = "Cho mình 2 tô Phở bò tái nạm và 1 ly trà đá ít đường"
    is_order = order_agent_service.is_order_intent(msg)
    assert is_order is True

    reply, draft, rec_chips, branch_cards = await order_agent_service.parse_and_build_order_draft(
        message=msg,
        branch_id="11111111-1111-1111-1111-111111111111"
    )

    assert draft is not None
    assert len(draft.items) >= 1
    assert draft.subtotal > 0

@pytest.mark.asyncio
async def test_order_agent_conversational_disambiguation():
    msg = "Cho tôi 2 tô Bún, 1 ly nước"
    is_order = order_agent_service.is_order_intent(msg)
    assert is_order is True

    reply, draft, rec_chips, branch_cards = await order_agent_service.parse_and_build_order_draft(
        message=msg
    )

    assert draft is None
    assert len(branch_cards) > 0
    assert len(rec_chips) > 0

@pytest.mark.asyncio
async def test_personalized_recommendations():
    profile = UserTasteProfileInput(
        searchHistory=["phở", "bún bò"],
        recentOrders=[{"productName": "Phở bò tái nạm"}],
        cartItems=[]
    )

    response = await recommendation_service.get_personalized_recommendations(profile, limit=5)
    assert len(response.items) > 0
    assert response.title == "Được đề xuất cho bạn"

@pytest.mark.asyncio
async def test_off_topic_query_rejection():
    # Queries that have nothing to do with food or DineX
    off_topic_queries = [
        "Thời tiết hôm nay thế nào?",
        "asdfghjkl123",
        "viết code python giúp tôi",
        "kể một câu chuyện cười đi",
        "1 + 1 bằng mấy"
    ]
    for query in off_topic_queries:
        is_order = order_agent_service.is_order_intent(query)
        assert is_order is False, f"Query '{query}' should not be recognized as order intent"

        reply, cards, chips = await rag_search_service.search_branches_and_dishes(query)
        assert len(cards) == 0, f"Query '{query}' should not return food cards"
        assert "Trợ lý Ẩm thực DineX" in reply or "không hỗ trợ" in reply or "chỉ có thể hỗ trợ" in reply

@pytest.mark.asyncio
async def test_greeting_and_identity_queries():
    greetings = ["Xin chào", "Hello bot", "Bạn là ai?", "Cảm ơn bạn"]
    for query in greetings:
        is_order = order_agent_service.is_order_intent(query)
        assert is_order is False

        reply, cards, chips = await rag_search_service.search_branches_and_dishes(query)
        assert len(cards) == 0, f"Greeting '{query}' should not return food cards"
        assert len(reply) > 0
        assert len(chips) > 0

@pytest.mark.asyncio
async def test_branch_missing_dish_rag():
    # Branch 2 only has "Phở bò thố đá đặc biệt" (no cơm tấm)
    branch2_id = "22222222-2222-2222-2222-222222222222"
    query = "Có cơm tấm sườn nướng không?"
    reply, cards, chips = await rag_search_service.search_branches_and_dishes(
        query=query,
        active_branch_id=branch2_id
    )
    # Must NOT show Pho as a substitute for Com tam!
    assert len(cards) == 0
    assert "hiện không có" in reply
    assert "DineX Phở Premium" in reply

@pytest.mark.asyncio
async def test_order_with_referential_branch_filler():
    # User message: "cho 1 phần Phở ngon có trong chi nhánh đó"
    # With chat_history containing the previous exploration of "DineX Phở Quận 1"
    chat_history = [
        {"role": "user", "content": "Chi nhánh DineX Phở Quận 1 bán những món nào?"},
        {"role": "assistant", "content": "Dạ, chi nhánh DineX Phở Quận 1 hiện đang phục vụ 2 món..."}
    ]
    msg = "cho 1 phần Phở ngon có trong chi nhánh đó"
    is_order = order_agent_service.is_order_intent(msg)
    assert is_order is True

    reply, draft, chips, cards = await order_agent_service.parse_and_build_order_draft(
        message=msg,
        branch_id=None,
        chat_history=chat_history
    )

    assert draft is not None
    assert len(draft.items) == 1
    assert "Phở bò tái nạm" in draft.items[0].name
    assert draft.items[0].quantity == 1
    assert draft.subtotal == 45000.0
    assert "DineX Phở Quận 1" in reply

@pytest.mark.asyncio
async def test_search_branch_all_dishes():
    query = "Chi nhánh DineX Phở Quận 1 bán những món nào?"
    reply, cards, chips = await rag_search_service.search_branches_and_dishes(query)
    
    assert len(cards) == 2
    assert "DineX Phở Quận 1" in reply
    dish_names = [c.dishName for c in cards]
    assert "Phở bò tái nạm" in dish_names
    assert "Cơm tấm sườn nướng" in dish_names
    assert any("Đặt món tại DineX Phở Quận 1" in chip for chip in chips)

@pytest.mark.asyncio
async def test_exact_dish_name_priority_pho_ngon():
    branch_id = "33333333-3333-3333-3333-333333333333"
    items = [
        SyncMenuItem(
            id="item-pho-xao",
            name="Phở Xào Bò Mềm",
            basePrice=70000.0,
            branchId=branch_id,
            branchName="DineX Cơm Gà Bảy Món",
            categoryName="Món Xào",
            district="Quận 1",
            rating=4.8,
            soldCount=50,
            isAvailable=True,
            isSoldOut=False
        ),
        SyncMenuItem(
            id="item-pho-ngon",
            name="Phở ngon",
            basePrice=65000.0,
            branchId=branch_id,
            branchName="DineX Cơm Gà Bảy Món",
            categoryName="Món Nước",
            district="Quận 1",
            rating=4.9,
            soldCount=120,
            isAvailable=True,
            isSoldOut=False
        )
    ]
    await sync_service.sync_batch(SyncBatchRequest(menuItems=items))

    msg = "Cho 2 phần Phở ngon"
    reply, draft, chips, cards = await order_agent_service.parse_and_build_order_draft(
        message=msg,
        branch_id=branch_id
    )

    assert draft is not None
    assert len(draft.items) == 1
    assert draft.items[0].name == "Phở ngon"
    assert draft.items[0].price == 65000.0
    assert draft.items[0].quantity == 2
    assert draft.subtotal == 130000.0

@pytest.mark.asyncio
async def test_search_branch_all_dishes_ban_cai_gi():
    query = "Quán Cơm gà bảy món bán cái gì ?"
    reply, cards, chips = await rag_search_service.search_branches_and_dishes(query)
    
    assert len(cards) >= 2
    assert "Cơm Gà Bảy Món" in reply
    dish_names_lower = [c.dishName.lower() for c in cards]
    assert "phở xào bò mềm" in dish_names_lower
    assert "phở ngon" in dish_names_lower







