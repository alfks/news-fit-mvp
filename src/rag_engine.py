# src/rag_engine.py
import chromadb
from sentence_transformers import SentenceTransformer

class RagEngine:
    def __init__(self, db_path="./data/ground_truth_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection("news_data")
        self.embedder = SentenceTransformer("jhgan/ko-sbert-nli")

    def retrieve_context(self, query_text, trojan_keyword, user_bias):
        """
        두 가지 종류의 데이터를 동시에 검색 (Fact + Trojan)
        """
        query_vec = self.embedder.encode([query_text]).tolist()
        
        # 1. Fact 검색 (할루시네이션 방지용) - 편향 상관없이 팩트만
        facts = self.collection.query(
            query_embeddings=query_vec,
            n_results=1,
            where={"type": "fact"} 
        )
        
        # 2. Trojan 검색 (사용자 성향의 반대 논리 찾기)
        # 사용자가 '보수'라면 -> DB에서 '진보'측 코멘트('progressive_quote') 검색
        target_bias_tag = "progressive_quote" if user_bias == "conservative" else "conservative_quote"
        
        trojans = self.collection.query(
            query_embeddings=self.embedder.encode([trojan_keyword]).tolist(),
            n_results=1,
            where={"type": target_bias_tag}
        )
        
        return {
            "fact_anchor": facts['documents'][0][0] if facts['documents'] else "관련 팩트 없음",
            "trojan_horse": trojans['documents'][0][0] if trojans['documents'] else "관련 반론 없음"
        }