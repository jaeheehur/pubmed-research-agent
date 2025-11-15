# PubMed Research Agent

의료 문헌 검색 및 Entity Extraction을 위한 AI Agent

## 빠른 시작

### 1. 환경 설정

```bash
conda activate pubmed_py312
```

### 2. GGUF 모델 다운로드 (Mac에서 빠른 추론)

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

### Transformers 모델 (느림)
- Kimi-K2-Thinking
- JSL-MedLlama-3-8B-v2.0

### Rule-based (가장 빠름)
- 키워드 기반 extraction

## 성능 비교 (Mac M2 Pro)

| 모델 유형 | 속도 | 정확도 | 권장 |
|-----------|------|--------|------|
| **GGUF (Q4)** | 3-5초 ⚡ | ⭐⭐⭐⭐ | ✅ |
| Transformers | 30-60초 🐌 | ⭐⭐⭐⭐⭐ | ❌ |
| Rule-based | 0.1초 🚀 | ⭐⭐ | 빠른 탐색용 |

## 주요 기능

1. **PubMed 검색**: NCBI E-utilities API 사용
2. **Entity Extraction**:
   - 약물/의약품
   - 부작용 (Adverse Events)
   - 환자 인구통계
   - 질병/증상
3. **시각화**: Plotly 차트
4. **내보내기**: JSON, 텍스트 리포트

## 파일 구조

```
.
├── app.py                      # Streamlit 웹 인터페이스
├── agent.py                    # Transformers 기반 agent
├── agent_gguf.py              # GGUF 기반 agent (빠름)
├── download_gguf_model.py     # GGUF 모델 다운로드/선택
├── list_installed_models.py   # 설치된 모델 확인
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
