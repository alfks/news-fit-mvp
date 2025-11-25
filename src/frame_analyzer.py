from transformers import pipeline

class FrameAnalyzer:
    def __init__(self):
        # Zero-shot 모델 로드
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        # --- [통합 후보군] 의대 증원 + 최저임금 프레임 모두 포함 ---
        self.candidate_labels = [
            # [주제 1: 의대 증원]
            "healthcare reform necessity",       # 의료 개혁 필요성 (정부/찬성)
            "concerns about education quality",  # 교육 질 저하 우려 (의료계/반대)
            
            # [주제 2: 최저임금]
            "guarantee of workers' livelihood",  # 노동자 생존권 보장 (노동계/진보)
            "economic burden for business",      # 소상공인 경영 부담 (경영계/보수)
        ]

        # --- [트로이 목마 매핑] 반대 논리 검색어 설정 ---
        self.counter_map = {
            # 의대 증원 관련
            "healthcare reform necessity": "deterioration of medical education quality",
            "concerns about education quality": "shortage of doctors statistics",
            
            # 최저임금 관련 (핵심 수정 부분)
            "guarantee of workers' livelihood": "small business labor cost burden", 
            "economic burden for business": "low income worker cost of living crisis"
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
        반대 프레임(Counter Frame)을 도출함.
        """
        # 1. 프레임 분류 (텍스트 앞부분 300자만 사용해 속도 향상)
        result = self.classifier(text[:300], self.candidate_labels)
        
        source_frame_en = result['labels'][0] # 가장 확률 높은 프레임 (1등)
        
        # 2. 한국어 변환
        source_frame_kr = self.kor_map.get(source_frame_en, source_frame_en)

        # 3. 트로이 목마 검색어 결정
        target_trojan_keyword = self.counter_map.get(source_frame_en, "general opposing view")
        
        return {
            "source_frame": source_frame_kr,
            "trojan_search_keyword": target_trojan_keyword
        }