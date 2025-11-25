import json
import os

# ==========================================
# 1. 경로 설정 (상위 폴더인 data/에 저장)
# ==========================================
# 현재 스크립트 위치: .../data/ground_truth/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 목표 폴더: 상위 폴더인 .../data/
DATA_DIR = os.path.dirname(BASE_DIR)

# 최종 파일 경로: .../data/ground_truth.json
FILE_NAME = "ground_truth.json"
SAVE_PATH = os.path.join(DATA_DIR, FILE_NAME)

print(f"📂 현재 스크립트 위치: {BASE_DIR}")
print(f"💾 데이터 생성 위치:   {SAVE_PATH}")

# ==========================================
# 2. 데이터 정의 (최저임금 이슈, 태그: minimum_wage)
# ==========================================
rag_data = [
    # --- 1. Fact 데이터 (공통 팩트) ---
    {
        "id": "fact_mw_01",
        "text": "2024년도 최저임금이 올해보다 2.5% 오른 시간당 9,860원으로 최종 확정되었다. 월급으로 환산하면 209시간 기준 206만 740원이다.",
        "metadata": {
            "type": "fact", 
            "topic": "minimum_wage",
            "source": "연합뉴스"
        }
    },
    {
        "id": "fact_mw_02",
        "text": "최저임금위원회는 밤샘 논의 끝에 표결을 통해 9,860원 안을 의결했다. 이는 역대 최장 심의 기간인 110일을 기록한 결과다.",
        "metadata": {
            "type": "fact", 
            "topic": "minimum_wage",
            "source": "연합뉴스"
        }
    },

    # --- 2. Trojan: 진보/노동계 논거 (보수 사용자에게 보여줄 내용) ---
    {
        "id": "opinion_prog_mw",
        "text": "노동계는 최근의 가파른 물가 상승률과 공공요금 인상을 고려할 때, 2.5% 인상은 실질임금이 삭감되는 것과 다를 바 없어 생존권이 위협받는다고 주장한다.",
        "metadata": {
            "type": "progressive_quote", 
            "topic": "minimum_wage",
            "source": "민주노총성명"
        }
    },

    # --- 3. Trojan: 보수/경영계 논거 (진보 사용자에게 보여줄 내용) ---
    {
        "id": "opinion_cons_mw",
        "text": "소상공인연합회는 '현재도 한계 상황인 자영업자들에게 이번 인상은 폐업을 강요하는 사망 선고'라며, 고용 축소가 불가피하다고 호소했다.",
        "metadata": {
            "type": "conservative_quote", 
            "topic": "minimum_wage",
            "source": "소상공인연합회"
        }
    }
]

# ==========================================
# 3. JSON 파일로 저장
# ==========================================
with open(SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(rag_data, f, ensure_ascii=False, indent=2)

print(f"✅ 데이터 파일 생성 완료! ({SAVE_PATH})")