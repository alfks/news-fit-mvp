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
        # 1. 파이프라인 실행 (상태창 표시)
        with st.status("News Fit 파이프라인 가동 중...", expanded=True) as status:
            
            # Step 1. 프레임 진단
            st.write("🧠 Module 1: 프레임 진단 중...")
            frame_result = analyzer.analyze(original_text)
            st.success(f"진단 완료: 원문은 '{frame_result['source_frame']}' 프레임입니다.")
            
            # Step 2. 문맥 확보 (RAG)
            st.write("🔍 Module 2: Fact & Trojan 검색 중...")
            context_data = rag.retrieve_context(
                query_text=original_text[:100],
                target_trojan_type=frame_result['target_trojan_type'], 
                topic=frame_result['topic']                
            )
            st.json(context_data) # 검색된 팩트/반론 보여주기
            
            # Step 3 & 4. 생성 및 검증 루프
            max_retries = 2
            final_news = "생성 실패 (재시도 횟수 초과)"
            final_val_result = None
            is_success = False

            for attempt in range(max_retries):
                st.write(f"🔄 생성 시도 중... ({attempt + 1}/{max_retries})")
                
                # 생성
                generator.load_adapter(user_bias) 
                current_news = generator.generate(original_text, context_data, user_bias)
                
                # 검증
                val_result = validator.validate(current_news, user_bias)
                
                # 결과 저장 (실패하더라도 일단 저장)
                final_news = current_news
                final_val_result = val_result

                if val_result['passed']:
                    is_success = True
                    st.success(f"✅ 검증 통과! (시도 {attempt + 1}회 만에 성공)")
                    break
                else:
                    st.warning(f"⚠️ 검증 미달 (격앙도: {val_result['intensity']:.2f})... 다시 생성합니다.")

            if not is_success:
                st.error("최대 시도 횟수를 초과했습니다. 가장 마지막 결과를 출력합니다.")
            
            # 상태창 닫기 (완료)
            status.update(label="변환 완료!", state="complete", expanded=False)

        # 2. 결과 출력 (상태창 바깥)
        st.divider()
        st.subheader("📝 변환된 뉴스 브리핑")
        
        # 기사 내용 표시
        st.markdown(f"> {final_news}")
        
        # 뉴턴 지수 대시보드
        col1, col2 = st.columns(2)
        if final_val_result:
            col1.metric("정치 편향도", f"{final_val_result['bias_score']}")
            col2.progress(final_val_result['intensity'], text=f"감정 격앙도 ({final_val_result['intensity']*100:.0f}%)")
        
        # [수정됨] 트로이 목마 경고창 조건부 표시
        trojan_text = context_data.get('trojan_horse', "")
        if "관련 반론 없음" not in trojan_text and len(trojan_text) > 5:
            st.warning(f"🐴 트로이 목마 작동됨: 반대 진영의 논거 '{trojan_text[:30]}...'가 포함되었습니다.")