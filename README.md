# 📰 News Fit (뉴스 핏)
### : 인지 편의성 기반 AI 뉴스 재구성 서비스
**(Cognitive Ease-based News Re-framing Service)**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?style=flat-square&logo=streamlit)
![Unsloth](https://img.shields.io/badge/LoRA-Unsloth-green?style=flat-square)
![RAG](https://img.shields.io/badge/RAG-ChromaDB-yellow?style=flat-square)

> **"Fact는 그대로, View는 내 입맛대로."**
> 
> **News Fit**은 사용자의 정치 성향과 감정 상태에 맞춰 기사의 **프레임(Frame)**과 **어조(Tone)**를 재구성하여, 뉴스 회피 현상을 해소하고 정보 불균형을 해결하는 AI 뉴스 에디터입니다.

---

## 🧐 Project Background (기획 배경)

현대 사회의 많은 독자들은 **"나와 맞지 않는 기사가 주는 스트레스"** 때문에 뉴스를 아예 보지 않는 **뉴스 회피(News Avoidance)** 현상을 겪고 있습니다. 기존의 추천 알고리즘은 편향된 기사만 보여주어 **필터 버블(Filter Bubble)**을 심화시킬 뿐입니다.

**News Fit**은 이 문제를 해결하기 위해 다음과 같은 접근을 시도합니다:

1.  **Personalization (맞춤 변환):** 읽기 싫은 뉴스를 사용자가 선호하는 문체로 변환하여 진입 장벽을 낮춥니다.
2.  **Fact Anchoring (팩트 고정):** RAG 기술을 통해 원문의 핵심 팩트는 100% 보존합니다.
3.  **Trojan Horse Strategy (트로이 목마):** 편안한 문체 속에 **반대 진영의 핵심 논거**를 은밀하게 포함하여 균형 잡힌 시각을 유도합니다.

---

## 🏗️ System Architecture (시스템 구조)

본 프로젝트는 **진단(Diagnosis) → 재료 확보(Retrieval) → 생성(Generation) → 검증(Validation)**의 4단계 파이프라인으로 구성됩니다.

```mermaid
graph TD
    User_Input(["📄 기사 원문"]) --> Module_1
    User_Pref(["🎚️ 사용자 성향"]) -.-> Module_2 & Module_3
    
    subgraph "Module 1: Diagnosis"
    Module_1["🧠 프레임 진단 (Frame Analysis)"]
    end
    
    Module_1 --> Module_2
    
    subgraph "Module 2: Retrieval (RAG)"
    Module_2["🔍 지능형 문맥 확보"]
    DB[("🗄️ Ground Truth DB")] <--> Module_2
    note["Fact (Anchor)<br/>+ Trojan (Counter-Logic)"]
    end
    
    Module_2 --> Module_3
    
    subgraph "Module 3: Generation (LoRA)"
    Module_3["✍️ 스타일 변환 (LLM + LoRA)"]
    Adapter_A["🔴 보수 Adapter"] -.-> Module_3
    Adapter_B["🔵 진보 Adapter"] -.-> Module_3
    end
    
    Module_3 --> Module_4
    
    subgraph "Module 4: Validation"
    Module_4{"⚖️ 뉴턴 지수 측정"}
    Module_4 -- Pass --> Output(["📰 맞춤형 뉴스 브리핑"])
    Module_4 -- Fail --> Module_3
    end
