import json
import os
import shutil
import chromadb
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. 경로 설정 (변경된 JSON 위치 반영)
# ==========================================
# 현재 스크립트 위치: .../data/ground_truth/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 상위 폴더: .../data/
DATA_ROOT_DIR = os.path.dirname(BASE_DIR)

# 1) 데이터 읽을 곳: .../data/ground_truth.json
DATA_PATH = os.path.join(DATA_ROOT_DIR, "ground_truth.json")

# 2) DB 저장할 곳: .../data/ground_truth_db (그대로 유지)
DB_PATH = os.path.join(DATA_ROOT_DIR, "ground_truth_db")

def main():
    print(f"🚀 DB 구축 스크립트 시작")
    print(f"📁 읽을 JSON 파일: {DATA_PATH}")
    print(f"💾 DB 저장 경로:   {DB_PATH}")

    # -------------------------------------------------------
    # 2. 기존 DB 초기화
    # -------------------------------------------------------
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print(f"🧹 기존 DB 삭제 완료")
    
    # -------------------------------------------------------
    # 3. JSON 데이터 로드
    # -------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        print(f"❌ [오류] 데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
        print("👉 같은 폴더의 'create_rag_data.py'를 먼저 실행해주세요.")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        rag_data = json.load(f)
    
    print(f"📂 데이터 로드 성공: {len(rag_data)}건")

    # -------------------------------------------------------
    # 4. 임베딩 모델 및 ChromaDB 로드
    # -------------------------------------------------------
    print("⏳ 임베딩 모델 로딩 중... (jhgan/ko-sbert-nli)")
    try:
        embedder = SentenceTransformer("jhgan/ko-sbert-nli")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection("news_data")

    # -------------------------------------------------------
    # 5. 데이터 벡터화 및 적재
    # -------------------------------------------------------
    print("💾 데이터를 벡터화하여 DB에 적재 중...")
    
    ids = [item["id"] for item in rag_data]
    documents = [item["text"] for item in rag_data]
    metadatas = [item["metadata"] for item in rag_data]
    
    embeddings = embedder.encode(documents).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print(f"✅ DB 구축 완료! 총 {len(documents)}개의 데이터가 저장되었습니다.")

if __name__ == "__main__":
    main()