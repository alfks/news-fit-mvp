from transformers import pipeline

class FrameAnalyzer:
    def __init__(self):
        # Zero-shot 모델 로드
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        # [1단계] 주제 분류용 키워드
        self.topic_keywords = {
            "medical": ["medical", "doctor", "hospital", "student", "school", "2000"], # 의대 관련
            "wage": ["wage", "salary", "labor", "money", "9860"] # 최저임금 관련
        }

        # [2단계] 주제별 프레임 후보군 (분리!)
        self.frames_by_topic = {
            "medical": [
                "healthcare reform necessity",       # 찬성
                "concerns about education quality"   # 반대
            ],
            "wage": [
                "guarantee of workers' livelihood",  # 진보
                "economic burden for business"       # 보수
            ]
        }

        # 반대 논리 매핑 (트로이 목마용)
        self.counter_map = {
            "healthcare reform necessity": "deterioration of medical education quality",
            "concerns about education quality": "shortage of doctors statistics",
            "guarantee of workers' livelihood": "small business labor cost burden",
            "economic burden for business": "low income worker cost of living crisis"
        }

        # 한국어 매핑
        self.kor_map = {
            "healthcare reform necessity": "의료 개혁 필요성 (찬성)",
            "concerns about education quality": "교육 질 저하 우려 (반대)",
            "guarantee of workers' livelihood": "노동자 생존권 보장 (진보)",
            "economic burden for business": "소상공인 경영 부담 (보수)"
        }

    def analyze(self, text):
        """
        1. 주제(Topic)를 먼저 하드코딩으로 감지
        2. 해당 주제의 프레임만 후보로 놓고 제로샷 분류 수행
        """
        
        # 1. 주제 감지 (간단한 키워드 매칭이 딥러닝보다 빠르고 정확함)
        topic = "unknown"
        if any(k in text for k in ["의대", "의사", "정원", "2000", "2천"]):
            topic = "medical"
        elif any(k in text for k in ["최저임금", "시급", "9860", "월급", "노동"]):
            topic = "wage"
            
        # 주제를 못 찾았으면 기본값(최저임금)으로 설정 (데모용 안전장치)
        if topic == "unknown":
            topic = "wage"

        # 2. 해당 주제의 프레임 후보만 가져오기
        candidate_labels = self.frames_by_topic[topic]
        
        # 3. 제로샷 분류 실행
        result = self.classifier(text[:512], candidate_labels)
        source_frame_en = result['labels'][0]
        
        # 4. 결과 반환
        source_frame_kr = self.kor_map.get(source_frame_en, source_frame_en)
        
        # 트로이 목마 검색어 결정 (반대 논리를 가져오도록 매핑)
        # 주제(Topic) 정보도 함께 넘겨줘야 RAG가 정확히 검색함
        target_trojan_keyword = self.counter_map.get(source_frame_en, "general opposing view")
        
        # RAG Engine이 topic을 필요로 하므로 함께 반환
        rag_topic_tag = "medical_reform" if topic == "medical" else "minimum_wage"
        
        return {
            "source_frame": source_frame_kr,
            "trojan_search_keyword": target_trojan_keyword,
            "topic": rag_topic_tag  # <--- [중요] RAG 검색 시 필터링에 사용
        }