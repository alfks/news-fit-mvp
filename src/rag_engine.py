import chromadb
from sentence_transformers import SentenceTransformer

class RagEngine:
    def __init__(self, db_path="./data/ground_truth_db"):
        # DB 경로가 없으면 자동으로 생성됨
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection("news_data")
        self.embedder = SentenceTransformer("jhgan/ko-sbert-nli")

    def retrieve_context(self, query_text, target_trojan_type, topic):
        """
        인자 설명:
        - query_text: 검색할 텍스트 (기사 원문 일부)
        - target_trojan_type: 검색할 반대 논리 타입 ('progressive_quote' 등)
        - topic: 검색할 주제 ('minimum_wage' 등)
        """
        query_vec = self.embedder.encode([query_text]).tolist()
        
        # 1. Fact 검색 (주제에 맞는 팩트만)
        facts = self.collection.query(
            query_embeddings=query_vec,
            n_results=1,
            where={
                "$and": [
                    {"type": "fact"},
                    {"topic": topic}  # 주제가 일치하는 것만 검색
                ]
            }
        )
        
        # 2. Trojan 검색 (주제에 맞고 + 지정된 반대 타입인 것만)
        trojans = self.collection.query(
            query_embeddings=query_vec,
            n_results=1,
            where={
                "$and": [
                    {"type": target_trojan_type}, # FrameAnalyzer가 정해준 타입
                    {"topic": topic}             # 주제가 일치하는 것만 검색
                ]
            }
        )
        
        # --- 안전하게 데이터 꺼내기 ---
        if facts['documents'] and facts['documents'][0]:
            fact_text = facts['documents'][0][0]
        else:
            fact_text = f"관련 팩트 없음 (주제: {topic})"

        if trojans['documents'] and trojans['documents'][0]:
            trojan_text = trojans['documents'][0][0]
        else:
            trojan_text = f"관련 반론 없음 (타입: {target_trojan_type})"
        # ----------------------------------------
        
        return {
            "fact_anchor": fact_text,
            "trojan_horse": trojan_text
        }