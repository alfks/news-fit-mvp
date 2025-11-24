# src/generator.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class StyleGenerator:
    def __init__(self, base_model_id="Qwen/Qwen2.5-1.5B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder="./offload", # [핵심] 부족한 메모리를 하드디스크로 대체
            low_cpu_mem_usage=True
        )
        self.active_adapter = None

    def load_adapter(self, bias_type):
        """사용자 성향(bias_type)에 따라 LoRA 어댑터 Hot-Swapping"""
        adapter_path = f"./models/adapter_{bias_type}"
        
        if self.active_adapter != bias_type:
            print(f"🔄 Switching Adapter to: {bias_type}")
            # 기존 어댑터 해제 후 새 어댑터 병합 로직 (PeftModel 활용)
            self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
            self.active_adapter = bias_type

    def generate(self, original_text, context_data, target_persona):
        """프롬프트 조립 및 생성"""
        
        fact = context_data['fact_anchor']
        trojan = context_data['trojan_horse']
        
        prompt = f"""
### 역할:
당신은 {target_persona} 성향의 뉴스 에디터입니다. 
독자가 읽기 편하도록 아래 지침에 따라 기사를 재작성하세요.

### 지침:
1. [팩트]는 절대 왜곡하지 말고 그대로 유지할 것. (Fact Anchoring)
2. [트로이 목마] 내용을 기사 중간에 "일각에서는 ~라는 의견도 있다" 형태로 자연스럽게 인용할 것.
3. 전체적인 어조는 {target_persona} 스타일에 맞출 것.

### 입력 데이터:
- 원문: {original_text}
- [팩트]: {fact}
- [트로이 목마]: {trojan}

### 재작성된 기사:
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=600)
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)