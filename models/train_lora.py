import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
PERSONA_TYPE = "conservative"  # 'conservative' 또는 'progressive'
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = f"./data/raw/raw_text_for_lora/train_{PERSONA_TYPE}.txt"
OUTPUT_DIR = f"./models/adapter_{PERSONA_TYPE}"

# GPU 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 실행 장치: {device.upper()}")

# ==========================================
# 2. 데이터 로드 및 전처리
# ==========================================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {DATA_PATH}")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.read().split('\n') if len(line) > 10]

print(f"✅ 데이터 로드 완료: {len(lines)} 문장")

alpaca_prompt = """### 지시:
당신은 {persona} 성향의 뉴스 에디터입니다. 이 문장을 {persona} 관점의 사설조로 재작성하세요.

### 입력:
{}

### 응답:
{}"""

def formatting_prompts_func(examples):
    instruction = "보수" if PERSONA_TYPE == "conservative" else "진보"
    texts = []
    for output in examples["text"]:
        # 입력(Input)은 비워두고("{}"), 실제 사설 내용(output)을 학습
        text = alpaca_prompt.format("{}", output, persona=instruction)
        texts.append(text)
    return { "text" : texts }

dataset = Dataset.from_dict({"text": lines})
dataset = dataset.map(formatting_prompts_func, batched=True)

# ==========================================
# 3. 모델 및 토크나이저 로드
# ==========================================
print("⏳ 모델 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token 

bnb_config = None
if device == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config if device == "cuda" else None,
    device_map={"": "cuda"} if device == "cuda" else "cpu", # GPU 강제 할당
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True
)

# ==========================================
# 4. LoRA 설정
# ==========================================
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, peft_config)

# ==========================================
# 5. 학습 실행 (Trainer)
# ==========================================
# [핵심 수정] 최신 trl 문법 적용
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    max_length=512,         
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=5,
    max_steps=30,
    save_steps=50,
    fp16=(device == "cuda"),
    use_cpu=(device == "cpu"),
    report_to="none",
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer, 
    args=sft_config,    
)

print(f"🚀 학습 시작 ({PERSONA_TYPE} 성향)...")
trainer.train()

# ==========================================
# 6. 저장
# ==========================================
print(f"💾 모델 저장 중... -> {OUTPUT_DIR}")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("🎉 학습 완료!")