# src/frame_analyzer.py
from transformers import pipeline

class FrameAnalyzer:
    def __init__(self):
        # MVP용: Zero-shot Classification 모델 사용 (빠른 구현)
        # 실제로는 BERTopic이나 학습된 분류기 권장
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        # 정의된 프레임 리스트 (예시)
        self.candidate_labels = [
            "economic burden for business",  # 소상공인/기업 부담 (보수)
            "livelihood of workers",         # 노동자 생계 (진보)
            "national security",             # 안보 우려
            "diplomatic relationship"        # 외교 관계
        ]

    def analyze(self, text):
        """
        기사의 현재 프레임(Source Frame)을 진단하고, 
        반대 프레임(Counter Frame)을 도출함.
        """
        # 1. 프레임 분류
        result = self.classifier(text[:512], self.candidate_labels)
        source_frame = result['labels'][0] # 가장 확률 높은 프레임
        
        # 2. 반대 프레임 매핑 (Rule-based for MVP)
        counter_frame_map = {
            "economic burden for business": "livelihood of workers",
            "livelihood of workers": "economic burden for business",
            # ... 추가 매핑
        }
        
        target_trojan_frame = counter_frame_map.get(source_frame, "general facts")
        
        return {
            "source_frame": source_frame,
            "trojan_search_keyword": target_trojan_frame
        }