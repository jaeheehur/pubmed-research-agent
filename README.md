# PubMed Research Agent

의료 문헌 검색 및 Entity Extraction을 위한 AI Agent

GGUF 양자화 모델과 Metal GPU 가속으로 Mac에서 빠른 의료 entity 추출을 제공합니다.

## 빠른 시작

### 1. 의존성 설치

```bash
# 가상 환경 활성화
conda activate pubmed_py312

# 패키지 설치
pip install -r requirements.txt

# macOS: Metal GPU 가속 지원 (권장)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# Linux/Windows: CUDA GPU 가속
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python

# CPU만 사용 (느림)
pip install llama-cpp-python
```

### 2. GGUF 모델 다운로드

```bash
# 설치된 모델 확인 또는 새로운 모델 다운로드
python download_gguf_model.py
```

**권장 모델**: JSL-MedLlama-3-8B Q4_K_M (~5GB, 3-5초/abstract)

### 3. 앱 실행

```bash
streamlit run app.py
```

## 사용 가능한 모델

### GGUF 모델 (권장 - Mac에서 빠름)
- **JSL-MedLlama-3-8B Q6_K**: 최고 정확도 (~6.6GB)
- **JSL-MedLlama-3-8B Q4_K_M**: 균형잡힌 선택 ✅ (~5GB)
- **기타**: BioMistral, Llama-3.2, TinyLlama

### Rule-based (가장 빠름)
- 키워드 및 정규식 기반 extraction
- Config 파일 기반 demographics 추출

## 성능 비교 (Mac M2 Pro)

| 모델 유형 | 속도 | 정확도 | 권장 |
|-----------|------|--------|------|
| **GGUF (Q4)** | 3-5초 ⚡ | ⭐⭐⭐⭐ | ✅ 균형잡힌 선택 |
| **GGUF (Q6)** | 20-30초 | ⭐⭐⭐⭐⭐ | 높은 정확도 필요시 |
| Rule-based | 0.1초 🚀 | ⭐⭐⭐ | 빠른 탐색용 |

## 주요 기능

1. **PubMed 검색**: NCBI E-utilities API 사용
2. **Entity Extraction**:
   - 약물/의약품 (Drugs)
   - 부작용 (Adverse Events) - 심각도 포함
   - 환자 인구통계 (Demographics):
     - 나이 (Age)
     - 성별 (Gender)
     - 인종/민족 (Race/Ethnicity)
     - 임신 여부 (Pregnancy Status) 🆕
     - BMI 🆕
     - 샘플 크기 (Sample Size)
   - 질병/증상 (Diseases)
3. **시각화**: Plotly 차트
4. **내보내기**: JSON, 텍스트 리포트

## 파일 구조

```
.
├── app.py                      # Streamlit 웹 인터페이스
├── agent_gguf.py              # GGUF 기반 agent (Metal GPU 가속)
├── download_gguf_model.py     # GGUF 모델 다운로드/선택
├── list_installed_models.py   # 설치된 모델 확인
├── config/                    # 설정 파일
│   ├── entity_extraction.prompt      # LLM extraction prompt
│   └── demographics_config.json      # Demographics 추출 규칙
├── tools/                     # PubMed 검색 도구
├── utils/                     # Entity extraction
└── requirements.txt
```

## 문제 해결

### GGUF 모델을 찾을 수 없음
```bash
python download_gguf_model.py
```

### llama-cpp-python 설치 오류
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

### Metal 경고 메시지 (`skipping kernel_* not supported`)
**정상 동작**: 이 메시지는 경고일 뿐, 에러가 아닙니다.
- Mac Metal GPU가 BFloat16을 지원하지 않아 Float32로 자동 fallback
- 성능에 영향 없음
- 앱은 이미 이 경고를 자동으로 숨김 처리

### Context window 경고 (`n_ctx_per_seq < n_ctx_train`)
**정상 동작**: 메모리 절약을 위해 컨텍스트 창을 줄임
- 원래: 8192 토큰
- 현재: 4096 토큰 (충분함, 대부분 abstract는 1000 토큰 이하)

### 모델이 느림
- GGUF Q4 또는 Q2 모델 사용
- Rule-based extraction 사용

## 테스트 및 디버깅

### GGUF 모델 테스트
```bash
python test_gguf_extraction.py
```

테스트 결과는 `logs/` 디렉토리에 자동 저장됩니다:
- `logs/gguf_test_*.log` - 실행 로그
- `logs/gguf_test_*.json` - 모델 응답 및 추출 결과
