# GGUF 모델 문제 해결 완료 ✅

## 문제 요약

1. **Segmentation Fault**: `streamlit run app.py` 실행 시 segfault 발생
2. **Rule-based Fallback**: GGUF 모델이 항상 rule-based extraction으로 폴백됨

## 해결된 내용

### 1. Segmentation Fault 해결

**파일**: `agent_gguf.py:36-43`

```python
self.llm = Llama(
    model_path=model_path,
    n_ctx=2048,      # 4096 → 2048 (메모리 사용량 감소)
    n_gpu_layers=n_gpu_layers,
    verbose=False,
    n_threads=2,     # 4 → 2 (CPU 부하 감소)
    n_batch=512      # 추가: 배치 크기 제한
)
```

### 2. 모델 타입 자동 감지

**파일**: `agent_gguf.py:47-55`

```python
# Mistral과 Llama 모델을 자동으로 감지하여 적절한 프롬프트 형식 사용
if 'biomistral' in model_path_lower or 'mistral' in model_path_lower:
    self.model_type = 'mistral'
elif 'llama' in model_path_lower or 'medllama' in model_path_lower:
    self.model_type = 'llama'
```

### 3. Mistral 전용 프롬프트 개선

**파일**: `agent_gguf.py:108-136`

- BioMistral에 최적화된 `[INST] ... [/INST]` 형식 사용
- JSON 스키마를 명시적으로 제공
- 더 짧고 직접적인 프롬프트로 개선

### 4. JSON Repair 기능 추가

**파일**: `agent_gguf.py:216-265`

```python
def _try_fix_json(self, json_str: str) -> Optional[str]:
    """
    Incomplete JSON 자동 수정:
    - 누락된 ] } 자동 추가
    - 누락된 필드에 기본값 추가
    - JSON 파싱 성공률 대폭 향상
    """
```

### 5. GGUF 추출 로직 개선

**파일**: `agent_gguf.py:179-198`

```python
# 이전: drugs OR adverse_events OR diseases 중 하나라도 있어야 함
# 현재: 어느 하나라도 추출되면 GGUF 결과 사용
if entities_dict is not None:
    entities_result = self._dict_to_entities(entities_dict)
    # 완전히 비어있을 때만 rule-based로 폴백
    if (len(entities_result.drugs) == 0 and
        len(entities_result.adverse_events) == 0 and
        len(entities_result.diseases) == 0):
        return self._extract_rule_based(text)
    return entities_result
```

### 6. TinyLlama 제거

**파일**: `app.py:274-276`

```python
# TinyLlama는 의료 도메인에 부적합하므로 목록에서 제외
if 'tinyllama' in gguf["display_name"].lower():
    continue
```

## 테스트 결과

### Real PubMed Search 테스트

```bash
python test_real_search.py
```

**결과**:
- Article 1: ✅ 1 drugs, 3 adverse events, 1 disease
- Article 2: ✅ 1 drug (JSON repair로 성공)
- Article 3: ✅ 12 drugs (대량 추출 성공)

**모든 추출이 GGUF 모델 사용, rule-based fallback 없음!**

## 사용 가능한 GGUF 모델

1. **BioMistral-7B** (추천)
   - 파일: `BioMistral-7B.Q4_K_M.gguf` (4.07 GB)
   - 의료 도메인 특화 모델
   - 빠른 추론 속도

2. **JSL-MedLlama-3-8B**
   - 파일: `JSL-MedLlama-3-8B-v2.0-Q4_K_M.gguf` (4.58 GB)
   - 의료 도메인 특화 모델
   - 더 큰 모델로 정확도 향상 가능

3. **JSL-MedLlama-3-8B (Q6)**
   - 파일: `JSL-MedLlama-3-8B-v2.0-Q6_K.gguf` (6.14 GB)
   - 가장 높은 정확도 (더 큰 파일 크기)

## 앱 실행 방법

### 방법 1: 안전한 실행 스크립트 (권장)

```bash
./run_app.sh
```

### 방법 2: 직접 실행

```bash
/opt/anaconda3/envs/pubmed_py312/bin/streamlit run app.py
```

또는

```bash
streamlit run app.py
```

## 앱 사용법

1. **모델 선택**: 사이드바에서 GGUF 모델 선택
   - BioMistral-7B 또는 JSL-MedLlama 선택
   - Rule-based (Fast) - GGUF 없이 키워드 기반 추출

2. **검색**: PubMed 쿼리 입력 후 검색

3. **결과 확인**:
   - 📄 Articles: 추출된 entity가 하이라이트된 초록
   - 🧬 Entities: 시각화된 entity 통계
   - 💾 Export: JSON/텍스트 다운로드

## 성능 특징

### GGUF 모델 장점
- ✅ AI 기반 entity 추출
- ✅ 컨텍스트 이해
- ✅ 복잡한 의학 용어 처리
- ✅ Metal GPU 가속 (Mac)

### Rule-based 장점
- ✅ 매우 빠른 속도
- ✅ 메모리 사용량 적음
- ✅ 예측 가능한 결과

## 문제 해결

### Segmentation Fault 재발 시
```bash
# agent_gguf.py 에서 파라미터 조정:
n_ctx=1024        # 더 작게
n_gpu_layers=0    # CPU만 사용
n_threads=1       # 스레드 감소
```

### GGUF 추출이 여전히 rule-based로 폴백되는 경우
```bash
# 로그 확인:
LOG_LEVEL=DEBUG streamlit run app.py

# "Successfully extracted with GGUF" 메시지 확인
# 없다면 JSON 파싱 실패 원인 확인
```

## 개선 사항 (선택사항)

1. **더 큰 context window**: `n_ctx`를 4096으로 증가 (메모리 충분시)
2. **더 많은 GPU layers**: `n_gpu_layers`를 33 (전체)로 증가
3. **더 긴 generation**: `max_tokens`를 3072로 증가

## 참고 파일

- `test_real_search.py`: 실제 PubMed 검색 테스트
- `test_medical_model.py`: 의료 모델 테스트
- `test_gguf_extraction.py`: GGUF 모델 진단
- `run_app.sh`: 안전한 앱 실행 스크립트

---

**최종 확인일**: 2025-11-17
**상태**: ✅ 모든 문제 해결 완료
