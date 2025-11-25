import json
import os
import shutil
import chromadb
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. 경로 설정
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR)) # news-fit-mvp/

# 읽을 데이터 (방금 변환한 파일)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "ground_truth.json")

# DB 저장 경로
DB_PATH = os.path.join(PROJECT_ROOT, "data", "ground_truth_db")

def main():
    print("🚀 DB 구축 스크립트 시작")
    print(f"📁 읽을 데이터: {DATA_PATH}")
    print(f"💾 DB 저장소:  {DB_PATH}")

    # -------------------------------------------------------
    # 2. 기존 DB 초기화
    # -------------------------------------------------------
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print("🧹 기존 DB 삭제 완료")

    # -------------------------------------------------------
    # 3. JSON 데이터 로드
    # -------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        print("❌ 데이터 파일 없음. create_rag_data.py를 먼저 실행하세요.")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        rag_data = json.load(f)
    
    print(f"📂 데이터 로드 성공: {len(rag_data)}건")

    # -------------------------------------------------------
    # 4. 임베딩 및 적재
    # -------------------------------------------------------
    print("⏳ 임베딩 모델 로딩 중...")
    embedder = SentenceTransformer("jhgan/ko-sbert-nli")
    
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection("news_data")

    print("💾 DB 적재 중...")
    
    # 배치 처리 (데이터가 많을 수 있으므로)
    batch_size = 50
    for i in range(0, len(rag_data), batch_size):
        batch = rag_data[i : i + batch_size]
        
        ids = [item["id"] for item in batch]
        documents = [item["text"] for item in batch]
        metadatas = [item["metadata"] for item in batch]
        embeddings = embedder.encode(documents).tolist()

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        print(f"   ... {i + len(batch)}/{len(rag_data)} 완료")

    print("✅ DB 구축 완료!")

if __name__ == "__main__":
    main()