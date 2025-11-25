# src/generator.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class StyleGenerator:
    def __init__(self, base_model_id="Qwen/Qwen2.5-1.5B-Instruct"):
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        self.base_model = AutoModelForCausalLM.from_pretrained(
                            base_model_id,
                            torch_dtype=torch.float32,
                            device_map="cpu",
                            low_cpu_mem_usage=True
        )
        self.active_adapter = None

    def load_adapter(self, bias_type):
        """사용자 성향(bias_type)에 따라 LoRA 어댑터 Hot-Swapping"""
        adapter_path = f"./models/adapter_{bias_type}"
        
        if self.active_adapter != bias_type:
            print(f"🔄 Switching Adapter to: {bias_type}")
            self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
            self.active_adapter = bias_type

    def generate(self, original_text, context_data, target_persona):
        fact = context_data['fact_anchor']
        trojan = context_data['trojan_horse']
        
        # [디벨롭 포인트] 프롬프트를 '단계별 사고(CoT)' 구조로 변경
        prompt = f"""### 역할:
당신은 {target_persona} 성향을 가진 20년 차 베테랑 논설위원입니다.
독자가 공감할 수 있도록 주어진 기사를 재구성하세요.

### 미션:
1. **[팩트]**는 절대 왜곡하지 말고 유지하십시오.
2. **[원문]**의 건조한 문체를 {target_persona} 특유의 어조(비판적, 단호함, 호소력 등)로 바꾸십시오.
3. **[트로이 목마]** 정보를 글의 흐름 속에 자연스럽게 녹여내십시오. (단순 나열 금지. "비록 ~라는 지적도 있지만..." 형태의 양보절로 활용)

### 입력 데이터:
- [원문]: {original_text}
- [팩트 (유지)]: {fact}
- [트로이 목마 (반대 논리)]: {trojan}

### 작성 가이드:
- 서론: 이슈의 심각성을 {target_persona} 관점에서 환기
- 본론: 팩트를 근거로 주장을 전개하되, [트로이 목마]를 교묘하게 언급하여 균형감 확보
- 결론: 강력한 제언으로 마무리

### 재작성된 기사:
"""
        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cpu")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=600)
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)