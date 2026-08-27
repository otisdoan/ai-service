import re
from typing import Dict, Any, List, Optional, Tuple
from app.services.vector_store import vector_store
from app.services.rag_search_service import rag_search_service
from app.models.schemas import OrderDraft, OrderDraftItem, MenuContextItem, BranchRecommendationCard

class OrderAgentService:
    def __init__(self):
        pass

    def is_order_intent(self, message: str) -> bool:
        msg = message.lower().strip()
        
        # Check if message contains food / drink keywords or quantity units
        has_food_or_unit = bool(
            re.search(r'(?:tô|phần|ly|suất|chén|dĩa|cái|chai|lon|hộp|gói|combo|set|size)', msg) or
            re.search(r'(?:bún|phở|cơm|mì|hủ tiếu|bánh|trà|cà phê|cafe|cf|nước|gà|bò|heo|thịt|cá|tôm|hải sản|chè|kem|lẩu|nướng|sinh tố|cháo|xôi)', msg)
        )

        # 1. Direct explicit ordering phrases
        explicit_order_phrases = [
            "đặt món", "lên đơn", "order món", "gọi món", "đặt hàng",
            "cho mình đặt", "cho tôi đặt", "cho em đặt", "cho anh đặt"
        ]
        if any(p in msg for p in explicit_order_phrases):
            return True

        # 2. Common colloquial order patterns: "Cho mình 2 tô phở", "Đặt 1 bún bò"
        order_prefixes = [
            "cho tôi", "cho to", "cho minh", "cho mình", "cho em", "cho anh", "cho chị",
            "lấy cho", "lay cho", "đặt", "dat", "order", "mua giúp", "mua hộ", "lên đơn",
            "cho 1", "cho 2", "cho 3", "cho một", "cho hai", "cho ba"
        ]
        if any(msg.startswith(prefix) for prefix in order_prefixes) and has_food_or_unit:
            return True

        # 3. Explicit quantity + dish pattern: e.g. "2 tô bún bò tái", "1 ly trà đào"
        if re.search(r'\d+\s*(?:tô|phần|ly|suất|chén|dĩa|cái|chai|lon|hộp|x)\s*(?:bún|phở|cơm|mì|hủ tiếu|bánh mì|trà|nước|cà phê|gà|bò)', msg):
            return True

        return False

    def clean_dish_phrase(self, phrase: str) -> str:
        """Cleans conversational filler phrases (e.g. 'có trong chi nhánh đó', 'của quán') from dish names."""
        p = phrase.lower().strip()
        # Remove leading quantity phrases
        p = re.sub(r'^\d+\s*(?:tô|phần|ly|suất|chén|dĩa|cái|chai|lon|hộp|gói|suat|phan)?\s*', '', p)
        # Remove trailing/leading conversational fillers
        fillers = [
            r'\bcó trong chi nhánh đó\b', r'\bở chi nhánh đó\b', r'\bcủa chi nhánh đó\b',
            r'\bcó trong quán đó\b', r'\bở quán đó\b', r'\bcủa quán đó\b',
            r'\bcó trong quán\b', r'\bcủa quán\b', r'\bở quán\b',
            r'\bcó trong chi nhánh\b', r'\bcủa chi nhánh\b', r'\bở chi nhánh\b',
            r'\bở đây\b', r'\btại quán\b', r'\bchi nhánh đó\b', r'\bquán đó\b',
            r'\bgiúp mình\b', r'\bgiúp tôi\b', r'\bgiúp em\b', r'\bcho mình\b', r'\bcho tôi\b',
            r'\bnhé\b', r'\bnha\b', r'\bạ\b', r'\bđược không\b', r'\bđi\b', r'\bđó\b', r'\bnày\b'
        ]
        for filler in fillers:
            p = re.sub(filler, '', p)
        return p.strip()

    def score_dish_match(self, raw_phrase: str, clean_phrase: str, dish_name: str) -> float:
        """Scores candidate dish against raw and cleaned user request."""
        d = dish_name.lower().strip()
        raw = raw_phrase.lower().strip()
        clean = clean_phrase.lower().strip()

        # 1. Exact match with raw phrase (e.g. "phở ngon" == "phở ngon")
        if raw == d:
            return 100.0
        # 2. Exact match with clean phrase (e.g. "phở" == "phở")
        if clean == d:
            return 95.0
        # 3. Raw phrase is prefix/substring of dish name (e.g. "phở ngon" in "phở ngon gia truyền")
        if raw in d:
            return 80.0 + (len(raw) / max(len(d), 1)) * 15.0
        if d in raw:
            return 75.0 + (len(d) / max(len(raw), 1)) * 15.0
        # 4. Clean phrase is prefix/substring
        if clean in d:
            penalty = 0.0
            if ("xào" in d and "xào" not in raw) or ("cuốn" in d and "cuốn" not in raw) or ("chiên" in d and "chiên" not in raw):
                penalty = 20.0
            return 50.0 + (len(clean) / max(len(d), 1)) * 10.0 - penalty
        # 5. Token overlap
        raw_tokens = set(raw.split())
        d_tokens = set(d.split())
        common = raw_tokens.intersection(d_tokens)
        if common:
            return 30.0 + len(common) * 10.0
        return 0.0

    async def parse_and_build_order_draft(
        self,
        message: str,
        branch_id: Optional[str] = None,
        menu_context: Optional[List[MenuContextItem]] = None,
        user_location: Optional[Dict[str, float]] = None,
        chat_history: Optional[List[Any]] = None
    ) -> Tuple[str, Optional[OrderDraft], List[str], List[BranchRecommendationCard]]:
        """
        Parses food ordering utterances:
        - If branch is specified or explicitly named in query/history: Builds a precise OrderDraft.
        - If query is generic (no specific store chosen): Returns matching branch recommendations
          with options to view menu, customize toppings/size, or add to cart.
        """
        msg = message.lower().strip()
        has_active_branch = bool(branch_id and branch_id != "00000000-0000-0000-0000-000000000000" and branch_id != "all")

        # 1. If branch_id was explicitly provided, use it directly
        selected_branch_id = branch_id if has_active_branch else None

        # 2. Check if user mentioned a specific branch in the message
        all_branch_candidates = vector_store.search_hybrid(
            query_vector=[0.0] * 1536,
            doc_type="menu_item",
            top_k=100,
            filters={"is_available": True}
        )

        distinct_branches = {}
        for b_cand in all_branch_candidates:
            b_id = b_cand["metadata"].get("branch_id")
            b_name = b_cand["metadata"].get("branch_name", "")
            if b_id and b_name and b_id not in distinct_branches:
                distinct_branches[b_id] = b_name

        if not selected_branch_id:
            # Sort branches by name length descending to match specific names first
            sorted_branches = sorted(distinct_branches.items(), key=lambda x: len(x[1]), reverse=True)
            for b_id, b_name in sorted_branches:
                b_name_lower = b_name.lower().strip()
                clean_b_name = re.sub(r'^(dinex|quán|tiệm|chi nhánh)\s+', '', b_name_lower).strip()
                if (b_name_lower and b_name_lower in msg) or (clean_b_name and len(clean_b_name) >= 3 and clean_b_name in msg):
                    selected_branch_id = b_id
                    break

        # 3. Check if previous message explicitly indicated a branch context (e.g. "quán đó", "ở đây", "đầu tiên")
        if not selected_branch_id and chat_history:
            is_referring_to_history = any(ref in msg for ref in ["quán đó", "quan do", "chi nhánh đó", "chi nhanh do", "ở đây", "o day", "đầu tiên", "dau tien", "quán 1", "quan 1"])
            if is_referring_to_history:
                for hist_msg in reversed(chat_history):
                    hist_content = getattr(hist_msg, "content", None) or (hist_msg.get("content") if isinstance(hist_msg, dict) else str(hist_msg))
                    if hist_content:
                        hist_text = hist_content.lower()
                        for b_id, b_name in sorted(distinct_branches.items(), key=lambda x: len(x[1]), reverse=True):
                            b_name_lower = b_name.lower().strip()
                            clean_b_name = re.sub(r'^(dinex|quán|tiệm|chi nhánh)\s+', '', b_name_lower).strip()
                            if (b_name_lower and b_name_lower in hist_text) or (clean_b_name and len(clean_b_name) >= 3 and clean_b_name in hist_text):
                                selected_branch_id = b_id
                                break
                        if selected_branch_id:
                            break

        # If NO branch is selected and no branch named in message, return branch recommendations so customer can explore & customize
        if not selected_branch_id:
            reply_text, cards, _ = await rag_search_service.search_branches_and_dishes(
                query=message,
                user_location=user_location,
                top_k=5
            )
            distinct_stores = list(dict.fromkeys([c.storeName for c in cards]))[:3]
            chips = [f"Đặt tại quán {s}" for s in distinct_stores]
            chips.append("Xem quán gần nhất")

            friendly_reply = (
                "Dạ! Tôi đã tìm thấy các quán có món ngon phù hợp với yêu cầu của bạn dưới đây.\n\n"
                "👉 Bạn có thể bấm **'Xem menu & Topping'** tại quán bạn thích để chọn Size, Topping và Ghi chú theo sở thích, "
                "hoặc chọn chi nhánh bên dưới để tôi lên đơn cho bạn nhé!"
            )
            return friendly_reply, None, chips, cards

        # If branch IS known: parse items and build draft for this specific branch
        clean_msg = re.sub(r'^(cho tôi|cho mình|cho em|cho anh|cho chị|lấy cho mình|lấy cho tôi|đặt cho tôi|đặt cho mình|đặt|mua|gọi|order)\s+', '', msg)
        split_phrases = re.split(r'[,+&]|\bvà\b|\bvới\b|\bthêm\b', clean_msg)

        items: List[OrderDraftItem] = []
        subtotal = 0.0
        selected_branch_name = None

        # Fetch menu candidates for this specific branch
        branch_candidates = vector_store.search_hybrid(
            query_vector=[0.0] * 1536,
            doc_type="menu_item",
            top_k=50,
            filters={"branch_id": selected_branch_id, "is_available": True}
        )

        requested_dishes: List[str] = []
        for phrase in split_phrases:
            phrase = phrase.strip()
            if not phrase:
                continue

            # Extract quantity
            qty = 1
            qty_match = re.search(r'(\d+)\s*(?:tô|phần|ly|suất|chén|dĩa|cái|chai|lon|hộp|x)?', phrase)
            if qty_match:
                try:
                    extracted_qty = int(qty_match.group(1))
                    if 1 <= extracted_qty <= 50:
                        qty = extracted_qty
                except ValueError:
                    pass

            # Extract note
            note = None
            note_match = re.search(r'(ít đá|không hành|nhiều ớt|ít ngọt|không đá|ít đường|nhiều nước béo|tái|chín|nóng|lạnh)', phrase)
            if note_match:
                note = note_match.group(1)

            raw_dish_phrase = re.sub(r'^\d+\s*(?:tô|phần|ly|suất|chén|dĩa|cái|chai|lon|hộp|x)?\s*', '', phrase)
            raw_dish_phrase = re.sub(r'(ít đá|không hành|nhiều ớt|ít ngọt|không đá|ít đường|nhiều nước béo|tái|chín|nóng|lạnh)', '', raw_dish_phrase).strip()

            clean_phrase = self.clean_dish_phrase(phrase)
            if not clean_phrase:
                clean_phrase = raw_dish_phrase

            if not raw_dish_phrase and not clean_phrase:
                continue

            requested_dishes.append(raw_dish_phrase or clean_phrase)
            
            # Score each branch candidate
            scored_cands = []
            for cand in branch_candidates:
                meta = cand["metadata"]
                dish_name = meta.get("name", "")
                score = self.score_dish_match(raw_dish_phrase, clean_phrase, dish_name)
                if score >= 30.0:
                    scored_cands.append((meta, score))

            scored_cands.sort(key=lambda x: x[1], reverse=True)

            if scored_cands:
                best = scored_cands[0][0]
                item_price = best.get("price", 0.0)
                item_name = best.get("name", "Món ăn")
                item_id = best.get("menu_item_id") or "00000000-0000-0000-0000-000000000000"
                selected_branch_name = best.get("branch_name")

                item_total = item_price * qty
                subtotal += item_total
                items.append(OrderDraftItem(
                    menuItemId=item_id,
                    name=item_name,
                    quantity=qty,
                    price=item_price,
                    notes=note
                ))


        if not items:
            branch_name = selected_branch_name or "quán"
            for cand in branch_candidates:
                b_name = cand["metadata"].get("branch_name")
                if b_name:
                    branch_name = b_name
                    break

            dish_req_display = f"món '{requested_dishes[0]}'" if requested_dishes else "món bạn yêu cầu"
            reply = (
                f"Dạ, chi nhánh **{branch_name}** hiện không có {dish_req_display}. "
                f"Bạn có thể xem thực đơn các món đang có của quán, hoặc để tôi tìm chi nhánh khác có món này cho bạn nhé!"
            )
            chips = [
                f"Tìm {requested_dishes[0]} ở quán khác" if requested_dishes else "Tìm quán khác",
                f"Xem menu {branch_name}",
                "Món bán chạy của quán"
            ]
            return reply, None, chips, []

        draft = OrderDraft(
            branchId=selected_branch_id,
            items=items,
            subtotal=subtotal,
            discountAmount=0.0,
            finalAmount=subtotal
        )

        formatted_total = f"{int(subtotal):,}đ".replace(",", ".")
        item_summary = ", ".join([f"{it.quantity}x {it.name}" + (f" ({it.notes})" if it.notes else "") for it in items])
        store_text = f" tại quán **{selected_branch_name}**" if selected_branch_name else ""

        reply = (
            f"Dạ, tôi đã lên phiếu đơn hàng tạm tính cho bạn{store_text} gồm:\n"
            f"• {item_summary}\n"
            f"👉 **Tổng cộng tạm tính:** {formatted_total}\n\n"
            f"Bạn có thể bấm **'Tùy chỉnh'** để chọn thêm Topping/Size hoặc bấm **'Xác nhận & Nhận mã QR'** để thanh toán nhé!"
        )

        return reply, draft, ["Xác nhận đặt hàng", "Tùy chỉnh thêm topping", "Thêm 1 ly trà đá"], []

order_agent_service = OrderAgentService()


