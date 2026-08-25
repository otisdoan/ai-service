from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import PythonChatRequest, PythonChatResponse
from app.services.order_agent_service import order_agent_service
from app.services.rag_search_service import rag_search_service

router = APIRouter(prefix="", tags=["AI Chat & RAG"])

@router.post("/chat", response_model=PythonChatResponse)
@router.post("/api/v1/chat", response_model=PythonChatResponse)
async def handle_chat(request: PythonChatRequest):
    """
    Main Chatbox Endpoint handling both:
    1. Multi-criteria semantic search RAG (e.g. "Có chi nhánh nào bán phở giá dưới 50k không?")
    2. Conversational ordering & draft preparation (e.g. "Cho mình 1 tô phở bò và 1 ly trà đá")
    """
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Tin nhắn không được để trống.")

    # 1. Check if user is placing an order
    if order_agent_service.is_order_intent(message):
        reply, draft, rec_chips, branch_cards = await order_agent_service.parse_and_build_order_draft(
            message=message,
            branch_id=request.branchId if request.branchId and request.branchId != "00000000-0000-0000-0000-000000000000" else None,
            menu_context=request.menuContext,
            user_location=request.userLocation,
            chat_history=request.chatHistory
        )
        return PythonChatResponse(
            reply=reply,
            orderDraft=draft,
            recommendations=rec_chips,
            branchRecommendations=branch_cards
        )

    # 2. Multi-criteria Search & RAG
    reply, cards, suggested_chips = await rag_search_service.search_branches_and_dishes(
        query=message,
        active_branch_id=request.branchId,
        user_location=request.userLocation,
        top_k=5
    )

    return PythonChatResponse(
        reply=reply,
        orderDraft=None,
        recommendations=suggested_chips,
        branchRecommendations=cards
    )

