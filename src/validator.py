# src/validator.py
from transformers import pipeline

class NewtonValidator:
    def __init__(self):
        # 감성 분석 모델 (SIA 역할)
        self.sentiment_pipe = pipeline("text-classification", model="matthewburke/korean_sentiment")
        # 정치 편향 모델 (MVP에선 키워드 기반이나 간단한 분류기 사용)
        
    def validate(self, text, target_bias):
        # 1. 감성 격앙도 측정
        sent = self.sentiment_pipe(text[:512])[0]
        intensity = sent['score'] if sent['label'] == 'LABEL_0' else 0.2 # 부정일 때 점수 높음
        
        # 2. 편향도 측정 (Dummy Logic for MVP)
        # 실제로는 학습된 KoBERT 모델의 output probability 사용
        bias_score = 0.8 if target_bias == "conservative" else -0.7
        
        return {
            "bias_score": bias_score,
            "intensity": intensity,
            "passed": intensity < 0.9 # 격앙도가 90% 미만이면 통과
        }