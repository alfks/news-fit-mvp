import streamlit as st
from src.frame_analyzer import FrameAnalyzer
from src.rag_engine import RagEngine
from src.generator import StyleGenerator
from src.validator import NewtonValidator

# 0. 초기화 (캐싱)
@st.cache_resource
def load_modules():
    return FrameAnalyzer(), RagEngine(), StyleGenerator(), NewtonValidator()

analyzer, rag, generator, validator = load_modules()

st.title("📰 News Fit MVP")
st.caption("인지 편의성 기반 뉴스 재구성 서비스")

# --- [Input 2: User Preference] ---
with st.sidebar:
    st.header("1. 사용자 설정")
    user_bias = st.select_slider(
        "당신의 정치 성향은?",
        options=["progressive", "neutral", "conservative"],
        value="conservative"
    )
    st.info(f"선택된 페르소나: {user_bias.upper()}")

# --- [Input 1: Source Data] ---
st.header("2. 기사 원문 입력")
original_text = st.text_area("분석할 기사를 입력하세요.", height=200)

if st.button("News Fit 변환 시작 🚀"):
    if not original_text:
        st.error("기사를 입력해주세요.")
    else:
        with st.status("News Fit 파이프라인 가동 중...", expanded=True) as status:
            
            # Step 1. 프레임 진단
            st.write("🧠 Module 1: 프레임 진단 중...")
            frame_result = analyzer.analyze(original_text)
            st.success(f"진단 완료: 원문은 '{frame_result['source_frame']}' 프레임입니다.")
            
            # Step 2. 문맥 확보 (RAG)
            st.write("🔍 Module 2: Fact & Trojan 검색 중...")
            context_data = rag.retrieve_context(
                query_text=original_text[:100], 
                trojan_keyword=frame_result['trojan_search_keyword'],
                user_bias=user_bias
            )
            st.json(context_data) # 검색된 팩트/반론 보여주기
            
            # Step 3. 생성 (LoRA)
            st.write("✍️ Module 3: 맞춤형 기사 생성 중...")
            generator.load_adapter(user_bias) # 어댑터 교체
            final_news = generator.generate(original_text, context_data, user_bias)
            
            # Step 4. 검증 (Newton Index)
            st.write("⚖️ Module 4: 뉴턴 지수 검증 중...")
            val_result = validator.validate(final_news, user_bias)
            
            status.update(label="변환 완료!", state="complete", expanded=False)

        # --- [Final Output] ---
        st.divider()
        st.subheader("📝 변환된 뉴스 브리핑")
        
        # 기사 내용 표시
        st.markdown(f"> {final_news}")
        
        # 뉴턴 지수 대시보드
        col1, col2 = st.columns(2)
        col1.metric("정치 편향도 (목표 달성)", f"{val_result['bias_score']}")
        col2.progress(val_result['intensity'], text="감정 격앙도")
        
        # 트로이 목마 하이라이트
        st.warning(f"🐴 트로이 목마 작동됨: 반대 진영의 논거 '{context_data['trojan_horse'][:30]}...'가 포함되었습니다.")