from transformers import pipeline

class FrameAnalyzer:
    def __init__(self):
        # Zero-shot 모델 로드
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        # --- [통합 후보군] ---
        self.candidate_labels = [
            # [주제 1: 의대 증원]
            "healthcare reform necessity",       # 찬성
            "concerns about education quality",  # 반대
            
            # [주제 2: 최저임금]
            "guarantee of workers' livelihood",  # 진보
            "economic burden for business",      # 보수
        ]

        # --- [핵심 수정] 반대 논리 매핑 (검색어가 아니라 'DB 태그'로 변경) ---
        # 원문 프레임 -> (반대편) DB 메타데이터 타입
        self.counter_map = {
            # 의대 증원 (찬성 -> 반대 자료 찾기)
            "healthcare reform necessity": "progressive_quote",      # 의협/과학계 입장
            "concerns about education quality": "conservative_quote", # 정부/OECD 입장
            
            # 최저임금 (진보 -> 보수 자료 찾기)
            "guarantee of workers' livelihood": "conservative_quote", # 경영계 입장
            "economic burden for business": "progressive_quote"       # 노동계 입장
        }

        # --- [UI 표시용] 한국어 매핑 ---
        self.kor_map = {
            "healthcare reform necessity": "의료 개혁 필요성 (찬성)",
            "concerns about education quality": "교육 질 저하 우려 (반대)",
            "guarantee of workers' livelihood": "노동자 생존권 보장 (진보)",
            "economic burden for business": "소상공인 경영 부담 (보수)"
        }

    def analyze(self, text):
        """
        기사의 현재 프레임(Source Frame)을 진단하고, 
        RAG 검색을 위한 타겟 타입(Target Type)과 주제(Topic)를 반환함.
        """
        
        # 1. 주제 감지 (키워드 매칭)
        topic = "unknown"
        if any(k in text for k in ["의대", "의사", "정원", "2000", "2천"]):
            topic = "medical"
        elif any(k in text for k in ["최저임금", "시급", "9860", "월급", "노동"]):
            topic = "wage"
            
        if topic == "unknown":
            topic = "wage" # 기본값

        # 2. 해당 주제의 프레임 후보만 가져오기
        # (단순화를 위해 여기선 전체 후보를 쓰지만, 이전 코드처럼 나누셔도 됩니다)
        
        # 3. 제로샷 분류 실행
        result = self.classifier(text[:512], self.candidate_labels)
        source_frame_en = result['labels'][0] # 1등 프레임
        
        # 4. 한국어 변환
        source_frame_kr = self.kor_map.get(source_frame_en, source_frame_en)
        
        # 5. 트로이 목마 타겟 결정 (DB 태그 반환)
        target_trojan_type = self.counter_map.get(source_frame_en, "conservative_quote")
        
        # 6. RAG용 주제 태그 결정
        rag_topic_tag = "medical_reform" if topic == "medical" else "minimum_wage"
        
        # src/frame_analyzer.py 반환값 확인
        return {
            "source_frame": source_frame_kr,
            "target_trojan_type": target_trojan_type,  
            "topic": rag_topic_tag      
}