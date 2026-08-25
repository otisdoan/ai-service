import re
import math
from typing import Dict, Any, List, Optional, Tuple
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store
from app.models.schemas import BranchRecommendationCard

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates distance in kilometers between two GPS coordinates using Haversine formula."""
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c

class RagSearchService:
    def __init__(self):
        pass

    def check_greeting_or_identity(self, query: str) -> Optional[Tuple[str, List[str]]]:
        """
        Handles standard greetings, identity questions, and polite conversational phrases without returning food cards.
        """
        q = query.lower().strip()
        clean_q = re.sub(r'[?!.,;:]', '', q).strip()

        # 1. Greetings
        greetings = [
            "xin chào", "chào", "hello", "hi", "alo", "hey", "good morning", "good evening",
            "chào bạn", "chào bot", "chào shop", "chào ad", "chào em", "chào anh", "chào chị"
        ]
        if clean_q in greetings or any(clean_q.startswith(g + " ") for g in greetings if len(g) > 2):
            reply = (
                "Dạ xin chào bạn! Tôi là Trợ lý AI của DineX. Tôi có thể hỗ trợ bạn:\n\n"
                "• Tìm kiếm món ăn & gợi ý quán ngon theo khẩu vị, mức giá hoặc khu vực gần bạn.\n"
                "• Hỗ trợ lên đơn đặt món nhanh và thanh toán trực tiếp qua mã QR PayOS.\n\n"
                "Bạn đang muốn tìm món ăn gì hay cần gợi ý quán ngon nào hôm nay?"
            )
            chips = ["Gợi ý món dưới 50k", "Tìm quán phở ngon", "Món ngon gần tôi", "Chi nhánh đang giảm giá"]
            return reply, chips

        # 2. Identity / Capabilities
        identities = [
            "bạn là ai", "bạn là gì", "mày là ai", "ai đấy", "bạn tên gì", "bot là ai", "ai thế",
            "hướng dẫn", "bạn giúp được gì", "bạn làm được gì", "chức năng của bạn", "trợ giúp", "help"
        ]
        if any(iden in clean_q for iden in identities):
            reply = (
                "Dạ! Tôi là Trợ lý Ảo AI DineX, người bạn đồng hành ẩm thực của bạn. 🍜\n\n"
                "Tôi có thể giúp bạn:\n"
                "1. Tìm quán & món ăn theo tiêu chí: mức giá (dưới 50k), quận huyện (Quận 1, Bình Thạnh), món ăn (phở bò, trà sữa).\n"
                "2. Gợi ý các quán gần vị trí hiện tại của bạn nhất.\n"
                "3. Lên phiếu đơn hàng và sinh mã QR thanh toán PayOS trực tiếp trong khung chat.\n\n"
                "Bạn hãy thử nhắn món ăn hoặc câu hỏi bạn muốn nhé!"
            )
            chips = ["Tìm quán gần Quận 1", "Món ngon dưới 50k", "Cho mình 2 tô phở bò", "Trà sữa khuyến mãi"]
            return reply, chips

        # 3. Thanks
        thanks = ["cảm ơn", "cảm ơn bạn", "thank you", "thanks", "cám ơn", "tks", "ok cảm ơn", "thank"]
        if any(clean_q == th or clean_q.startswith(th + " ") for th in thanks):
            reply = "Dạ không có gì ạ! Rất vui được hỗ trợ bạn. Chúc bạn có một bữa ăn thật ngon miệng cùng DineX nhé! ❤️"
            chips = ["Gợi ý món ăn ngon", "Tìm quán gần tôi", "Món đang giảm giá"]
            return reply, chips

        return None

    def is_food_or_dinex_domain(self, query: str) -> bool:
        """
        Validates if the user's message is related to food, drinks, pricing, restaurants, ordering, or DineX.
        Uses exact word token matching to prevent false positives from sub-strings.
        """
        q = query.lower().strip()
        words = set(re.findall(r'[\w\d]+', q))

        # 1. Multi-word food / drink terms
        multi_word_food = [
            "phở bò", "phở gà", "bún bò", "bún chả", "bún đậu", "bún riêu", "bún thịt nướng",
            "cơm tấm", "cơm sườn", "cơm gà", "bánh mì", "trà sữa", "trà đào", "cà phê",
            "nước ép", "sinh tố", "gà rán", "mì ý", "mì xào", "đồ nướng", "bò kho", "hải sản",
            "đồ ăn", "thức ăn", "món ăn", "đồ uống", "thức uống", "ăn vặt", "tráng miệng",
            "giải khát", "món mặn", "món chay"
        ]
        if any(term in q for term in multi_word_food):
            return True

        # 2. Single-word food / drink keywords (exact word token match)
        single_word_food = {
            "phở", "bún", "cơm", "mì", "mi", "hủ tiếu", "hu tieu", "miến", "cháo", "bánh", "gỏi", "lẩu", "nướng",
            "thịt", "gà", "bò", "heo", "lợn", "cá", "tôm", "mực", "trứng", "chả", "nem",
            "ốc", "ếch", "vịt", "canh", "súp", "xôi", "sườn", "pizza", "burger", "pasta",
            "spaghetti", "dimsum", "sushi", "tokbokki", "kimbap", "trà", "cafe", "cf",
            "nước", "sữa", "chè", "kem", "matcha", "soda", "bia", "rượu", "cocktail",
            "topping", "ăn", "uống", "đói", "thèm", "món", "ẩm thực", "chay", "rau", "nấm",
            "sáng", "trưa", "tối", "khuya"
        }
        if any(w in words for w in single_word_food):
            return True

        # 3. Price / Budget keywords
        if re.search(r'\d+\s*(?:k|nghìn|ngàn|vnd|đ|đồng)\b', q):
            return True
        price_words = {"giá", "rẻ", "tiền", "sinh viên", "tiết kiệm", "bình dân", "mắc", "đắt", "budget"}
        if any(w in words for w in price_words) or "bao nhiêu" in q:
            return True

        # 4. Restaurant / Location / Delivery / Quality keywords
        dinex_phrases = [
            "quán ăn", "quán nước", "nhà hàng", "cửa hàng", "chi nhánh", "thực đơn",
            "gần đây", "gần nhất", "khu vực", "đánh giá", "bán chạy", "nổi tiếng",
            "khuyến mãi", "giảm giá", "giao hàng", "lên đơn", "đặt món"
        ]
        if any(p in q for p in dinex_phrases):
            return True

        dinex_single_words = {
            "quán", "tiệm", "brand", "dinex", "menu", "combo", "set", "voucher", "deal",
            "freeship", "ship", "order", "quận", "huyện"
        }
        if any(w in words for w in dinex_single_words):
            return True

        # 5. Check if query matches any known database entity
        for doc in vector_store.documents.values():
            name = doc.metadata.get("name", "").lower()
            b_name = doc.metadata.get("branch_name", "").lower()
            if (name and name in q) or (b_name and b_name in q):
                return True

        return False

    def parse_query_constraints(self, query: str) -> Dict[str, Any]:
        """
        Extracts semantic query keywords and hard constraints from Vietnamese natural language.
        """
        q = query.lower().strip()
        filters: Dict[str, Any] = {
            "is_available": True,
            "is_sold_out": False
        }
        
        # 1. Price extraction
        # e.g., "dưới 50k", "< 50k", "dưới 50.000", "dưới 50000đ", "tầm 40k"
        under_price_match = re.search(r'(?:dưới|nhỏ hơn|<|tầm|khoảng|<=)\s*(\d+(?:[.,]\d+)?)\s*(k|nghìn|ngàn|vnd|đ)?', q)
        if under_price_match:
            num = float(under_price_match.group(1).replace(',', '.'))
            unit = under_price_match.group(2)
            if unit in ['k', 'nghìn', 'ngàn'] or num < 1000:
                max_price = num * 1000
            else:
                max_price = num
            filters["max_price"] = max_price

        above_price_match = re.search(r'(?:trên|lớn hơn|>|>=)\s*(\d+(?:[.,]\d+)?)\s*(k|nghìn|ngàn|vnd|đ)?', q)
        if above_price_match:
            num = float(above_price_match.group(1).replace(',', '.'))
            unit = above_price_match.group(2)
            if unit in ['k', 'nghìn', 'ngàn'] or num < 1000:
                min_price = num * 1000
            else:
                min_price = num
            filters["min_price"] = min_price

        # "Rẻ", "giá sinh viên" heuristic
        if "rẻ" in q or "sinh viên" in q or "tiết kiệm" in q:
            if "max_price" not in filters:
                filters["max_price"] = 45000.0

        # 2. Location / District extraction
        districts = [
            "quận 1", "quận 2", "quận 3", "quận 4", "quận 5", "quận 6", "quận 7", "quận 8",
            "quận 9", "quận 10", "quận 11", "quận 12", "bình thạnh", "gò vấp", "phú nhuận",
            "tân bình", "tân phú", "thủ đức", "bình tân", "bình chánh", "nhà bè", "hóc môn", "củ chi"
        ]
        for dist in districts:
            if dist in q:
                filters["district"] = dist
                break

        # 3. Rating / Best seller
        if "ngon nhất" in q or "đánh giá cao" in q or "top" in q or "5 sao" in q:
            filters["min_rating"] = 4.5

        # 4. Extract specific dish/category topic keyword
        food_topics = [
            "phở bò", "phở gà", "phở", "bún bò huế", "bún bò", "bún chả", "bún đậu", "bún riêu", "bún thịt nướng", "bún",
            "cơm tấm", "cơm sườn", "cơm gà", "cơm chiên", "cơm", "bánh mì", "bánh xèo", "bánh cuốn", "bánh bao", "bánh tráng", "bánh",
            "hủ tiếu nam vang", "hủ tiếu", "hu tieu", "miến", "cháo", "gỏi cuốn", "gỏi",
            "trà sữa", "trà đào", "trà tắc", "trà chanh", "trà", "cà phê", "cafe", "cf",
            "nước ép", "sinh tố", "lẩu", "đồ nướng", "gà rán", "mì ý", "mì xào", "mì", "chè", "kem", "nem nướng", "bò kho"
        ]
        for topic in food_topics:
            if topic in q:
                filters["dish_topic"] = topic
                break

        return filters

    def find_mentioned_branch(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Finds branch metadata if the query explicitly mentions a branch name or alias.
        """
        q = query.lower().strip()
        distinct_branches: Dict[str, Dict[str, Any]] = {}
        for doc in vector_store.documents.values():
            b_id = str(doc.metadata.get("branch_id"))
            b_name = doc.metadata.get("branch_name")
            if b_id and b_name and b_id not in distinct_branches:
                distinct_branches[b_id] = {
                    "id": b_id,
                    "name": b_name,
                    "address": doc.metadata.get("address") or "",
                    "district": doc.metadata.get("district") or "",
                    "latitude": doc.metadata.get("latitude"),
                    "longitude": doc.metadata.get("longitude")
                }

        sorted_branches = sorted(distinct_branches.values(), key=lambda x: len(x["name"]), reverse=True)
        # 1. Exact / Substring match on full branch name
        for b in sorted_branches:
            b_name_lower = b["name"].lower()
            if b_name_lower in q:
                return b

        # 2. Cleaned branch name without prefix "DineX ", "Quán ", "Chi nhánh "
        for b in sorted_branches:
            clean_b_name = re.sub(r'^(dinex|quán|tiệm|chi nhánh)\s+', '', b["name"].lower()).strip()
            if clean_b_name and len(clean_b_name) >= 3 and clean_b_name in q:
                return b

        return None

    async def search_branches_and_dishes(
        self,
        query: str,
        active_branch_id: Optional[str] = None,
        user_location: Optional[Dict[str, float]] = None,
        top_k: int = 5
    ) -> Tuple[str, List[BranchRecommendationCard], List[str]]:
        """
        Executes multi-criteria RAG search and returns textual summary + interactive branch cards + dynamic chips.
        Supports:
        1. Querying all dishes of a specific branch (e.g. "Chi nhánh Phở Quận 1 bán những món nào?")
        2. Querying specific dishes at a branch
        3. Multi-criteria search across all branches
        """
        # 1. Check greeting / identity / polite conversational phrases
        greeting_result = self.check_greeting_or_identity(query)
        if greeting_result:
            reply_text, chips = greeting_result
            return reply_text, [], chips

        # 2. Check domain relevance: Is this related to food, drinks, pricing, or DineX?
        if not self.is_food_or_dinex_domain(query):
            out_of_domain_reply = (
                "Dạ, hiện tại tôi là Trợ lý Ẩm thực DineX nên chỉ có thể hỗ trợ các thông tin về món ăn, "
                "thức uống, giá cả, gợi ý quán ngon và hỗ trợ đặt hàng thôi ạ. 😊\n\n"
                "Bạn có thể thử hỏi tôi tìm món ngon, tìm quán gần đây hoặc nhờ tôi lên đơn đặt món nhé!"
            )
            default_chips = [
                "Món ngon gần tôi",
                "Món ăn dưới 50k",
                "Chi nhánh đang giảm giá",
                "Tìm quán phở ngon"
            ]
            return out_of_domain_reply, [], default_chips

        # 3. Check if query specifically names a branch
        mentioned_branch = self.find_mentioned_branch(query)
        target_branch_id = mentioned_branch["id"] if mentioned_branch else (
            active_branch_id if active_branch_id and active_branch_id != "00000000-0000-0000-0000-000000000000" and active_branch_id != "all" else None
        )
        
        target_branch_name: Optional[str] = mentioned_branch["name"] if mentioned_branch else None
        target_branch_addr: Optional[str] = mentioned_branch.get("address") if mentioned_branch else None
        if target_branch_id and not target_branch_name:
            for doc in vector_store.documents.values():
                if str(doc.metadata.get("branch_id")) == str(target_branch_id):
                    target_branch_name = doc.metadata.get("branch_name")
                    target_branch_addr = doc.metadata.get("address")
                    break

        # Parse constraints on the query (stripping branch name if present so branch name words like "Cơm gà" don't become restrictive filters)
        query_for_constraints = query
        if mentioned_branch:
            b_name = mentioned_branch["name"]
            query_for_constraints = re.sub(re.escape(b_name), "", query_for_constraints, flags=re.IGNORECASE)
            clean_b_name = re.sub(r'^(dinex|quán|tiệm|chi nhánh)\s+', '', b_name, flags=re.IGNORECASE).strip()
            if clean_b_name:
                query_for_constraints = re.sub(re.escape(clean_b_name), "", query_for_constraints, flags=re.IGNORECASE)
            query_for_constraints = re.sub(r'\b(quán|tiệm|chi nhánh|nhà hàng|cửa hàng)\b', '', query_for_constraints, flags=re.IGNORECASE).strip()

        filters = self.parse_query_constraints(query_for_constraints if query_for_constraints else query)
        dish_topic = filters.get("dish_topic")

        q_lower = query.lower().strip()
        is_asking_branch_menu = bool(
            mentioned_branch and (
                re.search(r'(?:bán\s+(?:những\s+|các\s+)?(?:món\s+)?(?:gì|nào|cái\s+gì)|có\s+(?:những\s+|các\s+)?(?:món\s+)?(?:gì|nào|cái\s+gì)|thực\s+đơn|menu|danh\s+sách\s+món|bảng\s+giá|danh\s+mục)', q_lower) or
                dish_topic is None
            )
        )

        user_lat = user_location.get("lat") or user_location.get("latitude") if user_location else None
        user_lng = user_location.get("lng") or user_location.get("longitude") if user_location else None

        # 4. If user is asking for the complete menu of a specific branch
        if is_asking_branch_menu and target_branch_id:
            branch_items = [
                doc.metadata for doc in vector_store.documents.values()
                if str(doc.metadata.get("branch_id")) == str(target_branch_id) and doc.metadata.get("is_available") is not False
            ]

            cards: List[BranchRecommendationCard] = []
            for meta in branch_items:
                price = meta.get("price", 0.0)
                price_text = f"{int(price):,}đ".replace(",", ".")

                dist_km: Optional[float] = None
                b_lat = meta.get("latitude")
                b_lng = meta.get("longitude")
                if user_lat is not None and user_lng is not None and b_lat and b_lng:
                    try:
                        dist_km = calculate_distance(user_lat, user_lng, float(b_lat), float(b_lng))
                    except Exception:
                        dist_km = None

                if dist_km is not None:
                    distance_str = f"{dist_km:.1f} km"
                    delivery_time = f"{max(15, int(dist_km * 7 + 10))}-{max(25, int(dist_km * 7 + 20))} phút"
                else:
                    distance_str = meta.get("district") or "1.5 km"
                    delivery_time = "20-30 phút"

                tag = "Món bán chạy" if meta.get("sold_count", 0) > 50 else "Được yêu thích"

                cards.append(BranchRecommendationCard(
                    branchId=str(meta.get("branch_id", "")),
                    menuItemId=str(meta.get("menu_item_id", "")),
                    storeName=meta.get("branch_name", target_branch_name or "DineX"),
                    dishName=meta.get("name", "Món ăn"),
                    priceAmount=price,
                    priceText=price_text,
                    rating=meta.get("rating", 5.0),
                    reviews=meta.get("review_count", 100),
                    deliveryTime=delivery_time,
                    distance=distance_str,
                    imageUrl=meta.get("image_url", ""),
                    tag=tag
                ))

            if cards:
                addr_text = f" ({target_branch_addr})" if target_branch_addr else ""
                reply = (
                    f"Dạ, chi nhánh **{target_branch_name}**{addr_text} hiện đang phục vụ {len(cards)} món ngon sau đây:\n\n"
                    f"👉 Bạn có thể bấm **'Thêm vào giỏ'** hoặc nhắn tên món để tôi lên đơn đặt ngay nhé!"
                )
                chips = [
                    f"Đặt món tại {target_branch_name}",
                    f"Món bán chạy của {target_branch_name}",
                    "Xem quán gần tôi nhất"
                ]
                return reply, cards, chips

        # 5. Standard multi-criteria search
        if target_branch_id:
            filters["branch_id"] = target_branch_id

        # Generate embedding vector for the query
        query_vector = await embedding_service.get_embedding(query)

        # Retrieve matching candidates
        results = vector_store.search_hybrid(
            query_vector=query_vector,
            doc_type="menu_item",
            top_k=50,
            filters=filters
        )

        candidates_with_meta: List[Tuple[Dict[str, Any], Optional[float]]] = []
        for res in results:
            meta = res["metadata"]
            dish_name = meta.get("name", "")
            cat_name = meta.get("category_name", "")

            # If user searched for a dish topic (e.g. "phở"), ensure the candidate actually matches
            if dish_topic:
                topic_words = [w for w in dish_topic.split() if len(w) >= 2]
                name_match = all(w in dish_name.lower() for w in topic_words)
                cat_match = all(w in cat_name.lower() for w in topic_words)
                if not name_match and not cat_match:
                    continue

            dist_km: Optional[float] = None
            b_lat = meta.get("latitude")
            b_lng = meta.get("longitude")
            if user_lat is not None and user_lng is not None and b_lat and b_lng:
                try:
                    dist_km = calculate_distance(user_lat, user_lng, float(b_lat), float(b_lng))
                except Exception:
                    dist_km = None

            candidates_with_meta.append((meta, dist_km))

        # Sort: if query explicitly asks for nearest or user GPS is given, prioritize distance
        if "gần" in query.lower() or "quanh đây" in query.lower() or "gần nhất" in query.lower():
            candidates_with_meta.sort(key=lambda x: (x[1] if x[1] is not None else 9999.0))
        elif user_lat is not None and user_lng is not None:
            # Sort by balance of price and distance
            candidates_with_meta.sort(key=lambda x: (x[1] if x[1] is not None else 9999.0, x[0].get("price", 0)))

        cards: List[BranchRecommendationCard] = []
        for meta, dist_km in candidates_with_meta:
            dish_name = meta.get("name", "")
            price = meta.get("price", 0.0)
            price_text = f"{int(price):,}đ".replace(",", ".")
            
            # Real dynamic distance
            if dist_km is not None:
                distance_str = f"{dist_km:.1f} km"
                delivery_time = f"{max(15, int(dist_km * 7 + 10))}-{max(25, int(dist_km * 7 + 20))} phút"
            else:
                distance_str = meta.get("district") or "1.5 km"
                delivery_time = "20-30 phút"

            # Smart Tagging
            tag = "Gợi ý hàng đầu"
            if "max_price" in filters and price <= filters["max_price"]:
                tag = "Giá tốt nhất"
            elif dist_km is not None and dist_km <= 2.0:
                tag = "Gần bạn"
            elif meta.get("sold_count", 0) > 50:
                tag = "Bán chạy"
            elif meta.get("rating", 5.0) >= 4.8:
                tag = "Được yêu thích"

            card = BranchRecommendationCard(
                branchId=str(meta.get("branch_id", "")),
                menuItemId=str(meta.get("menu_item_id", "")),
                storeName=meta.get("branch_name", "DineX Branch"),
                dishName=dish_name,
                priceAmount=price,
                priceText=price_text,
                rating=meta.get("rating", 4.8),
                reviews=meta.get("review_count", 120),
                deliveryTime=delivery_time,
                distance=distance_str,
                imageUrl=meta.get("image_url", ""),
                tag=tag
            )
            cards.append(card)
            if len(cards) >= top_k:
                break

        # Generate dynamic contextual reply and chips
        reply = self._build_reply_text(query, filters, cards, target_branch_name)
        dynamic_chips = self.generate_suggestive_chips(query, dish_topic, cards, target_branch_name)
        return reply, cards, dynamic_chips

    def _build_reply_text(
        self,
        query: str,
        filters: Dict[str, Any],
        cards: List[BranchRecommendationCard],
        active_branch_name: Optional[str] = None
    ) -> str:
        dish_topic = filters.get("dish_topic")

        if not cards:
            if active_branch_name:
                dish_display = f"món '{dish_topic.title()}'" if dish_topic else "món bạn yêu cầu"
                return f"Dạ, chi nhánh **{active_branch_name}** hiện không có {dish_display}. Bạn có thể xem thực đơn các món đang có của quán, hoặc chọn tìm kiếm ở các chi nhánh khác nhé!"

            dish_display = f"món '{dish_topic.title()}'" if dish_topic else "món"
            reply = f"Dạ, hiện tại hệ thống chưa tìm thấy chi nhánh nào có {dish_display} phù hợp với tiêu chí của bạn"
            if "max_price" in filters:
                max_str = f"{int(filters['max_price']):,}".replace(",", ".") + "đ"
                reply += f" dưới {max_str}"
            if "district" in filters:
                reply += f" tại khu vực {filters['district'].title()}"
            reply += ". Bạn có thể thử tìm với mức giá khác hoặc chọn các món gợi ý phổ biến bên dưới nhé!"
            return reply

        count = len(cards)
        reply = f"Dạ, tôi đã tìm thấy {count} chi nhánh có món ngon phù hợp với yêu cầu của bạn"
        if "max_price" in filters:
            min_str = f"{int(cards[0].priceAmount):,}".replace(",", ".") + "đ"
            max_str = f"{int(filters['max_price']):,}".replace(",", ".") + "đ"
            reply += f" (giá từ {min_str}, dưới {max_str})"
        if "district" in filters:
            reply += f" tại khu vực {filters['district'].title()}"
        reply += ". Mời bạn xem chi tiết hoặc bấm 'Thêm vào giỏ' để đặt món ngay nhé:"
        return reply

    def generate_suggestive_chips(
        self,
        query: str,
        dish_topic: Optional[str],
        cards: List[BranchRecommendationCard],
        active_branch_name: Optional[str] = None
    ) -> List[str]:
        if not cards and active_branch_name:
            chips = []
            if dish_topic:
                chips.append(f"Tìm quán khác có {dish_topic}")
            chips.append(f"Xem menu {active_branch_name}")
            chips.append("Món ngon gần tôi")
            return chips

        chips = []
        distinct_stores = list(dict.fromkeys([c.storeName for c in cards]))
        if distinct_stores:
            chips.append(f"Xem menu quán {distinct_stores[0]}")
        
        if dish_topic:
            chips.append(f"Đặt 1 {dish_topic}")
            chips.append(f"{dish_topic.title()} gần tôi nhất")
            chips.append(f"Có món gì rẻ dưới 40k?")
        else:
            chips.append("Món ngon gần tôi")
            chips.append("Món đang giảm giá")
            chips.append("Quán đánh giá cao")
        
        return chips[:4]

rag_search_service = RagSearchService()

