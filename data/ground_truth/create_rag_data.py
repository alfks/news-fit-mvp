import json
import os

# ==========================================
# 1. 경로 설정
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR)) # news-fit-mvp/

# 크롤링된 데이터 경로 (crawler.py가 저장한 곳)
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "input", "articles_naver.json")

# 최종 저장 경로 (RAG용)
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "ground_truth.json")

print(f"📂 읽을 파일: {RAW_DATA_PATH}")
print(f"💾 저장할 파일: {OUTPUT_PATH}")

# ==========================================
# 2. 분류 규칙 (언론사별 성향 매핑)
# ==========================================
# 스트레이트 뉴스 (Fact)
FACT_MEDIA = ["연합뉴스", "YTN", "KBS", "SBS", "MBC"]

# 보수 성향 (Opinion - Conservative)
CONS_MEDIA = ["조선일보", "중앙일보", "동아일보", "한국경제", "매일경제"]

# 진보 성향 (Opinion - Progressive)
PROG_MEDIA = ["한겨레", "경향신문", "오마이뉴스", "프레시안"]

def classify_article(article):
    media = article.get("media_outlet", "")
    content = article.get("content", "")
    
    # 너무 짧은 기사는 제외
    if not content or len(content) < 50:
        return None

    metadata = {
        "source": media,
        "topic": "minimum_wage", # 지금은 최저임금 주제로 고정
        "date": article.get("published_date", "")
    }

    # 1. Fact 분류
    if media in FACT_MEDIA:
        metadata["type"] = "fact"
        return metadata

    # 2. Trojan(Opinion) 분류
    elif media in CONS_MEDIA:
        metadata["type"] = "conservative_quote"
        return metadata
        
    elif media in PROG_MEDIA:
        metadata["type"] = "progressive_quote"
        return metadata
    
    return None # 분류 불가

# ==========================================
# 3. 데이터 변환 실행
# ==========================================
if not os.path.exists(RAW_DATA_PATH):
    print(f"❌ [오류] 크롤링된 데이터가 없습니다: {RAW_DATA_PATH}")
    print("👉 먼저 크롤러를 실행해서 데이터를 수집해주세요.")
    exit()

with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

articles = raw_data.get("articles", [])
rag_data = []

print(f"🔄 총 {len(articles)}개 기사 처리 중...")

for article in articles:
    meta = classify_article(article)
    if meta:
        # RAG 데이터 포맷으로 변환
        rag_item = {
            "id": article["article_id"],
            "text": article["content"][:1000], # 너무 길면 자름 (임베딩 제한 고려)
            "metadata": meta
        }
        rag_data.append(rag_item)

# ==========================================
# 4. 저장
# ==========================================
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(rag_data, f, ensure_ascii=False, indent=2)

print(f"✅ 변환 완료! 총 {len(rag_data)}개의 RAG용 데이터 생성.")
print(f"   - Fact: {len([d for d in rag_data if d['metadata']['type']=='fact'])}개")
print(f"   - Conservative: {len([d for d in rag_data if d['metadata']['type']=='conservative_quote'])}개")
print(f"   - Progressive: {len([d for d in rag_data if d['metadata']['type']=='progressive_quote'])}개")