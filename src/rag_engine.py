# src/rag_engine.py
import chromadb
from sentence_transformers import SentenceTransformer

class RagEngine:
    def __init__(self, db_path="./data/ground_truth_db"):
        # DB 경로가 없으면 자동으로 생성됨
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection("news_data")
        self.embedder = SentenceTransformer("jhgan/ko-sbert-nli")

    def retrieve_context(self, query_text, trojan_keyword, user_bias):
        query_vec = self.embedder.encode([query_text]).tolist()
        
        # 1. Fact 검색
        facts = self.collection.query(
            query_embeddings=query_vec,
            n_results=1,
            where={"type": "fact"} 
        )
        
        # 2. Trojan 검색
        target_bias_tag = "progressive_quote" if user_bias == "conservative" else "conservative_quote"
        
        trojans = self.collection.query(
            query_embeddings=self.embedder.encode([trojan_keyword]).tolist(),
            n_results=1,
            where={"type": target_bias_tag}
        )
        
        # --- [수정된 부분] 안전하게 데이터 꺼내기 ---
        # facts['documents']가 존재하고, 그 안의 첫 번째 리스트가 비어있지 않은지 확인해야 함
        if facts['documents'] and facts['documents'][0]:
            fact_text = facts['documents'][0][0]
        else:
            fact_text = "관련 팩트 없음 (DB가 비어있거나 매칭되지 않음)"

        if trojans['documents'] and trojans['documents'][0]:
            trojan_text = trojans['documents'][0][0]
        else:
            trojan_text = "관련 반론 없음"
        # ----------------------------------------
        
        return {
            "fact_anchor": fact_text,
            "trojan_horse": trojan_text
        }