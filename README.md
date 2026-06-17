# Zero-Shot Performance and Limitations Analysis of Korean Sentiment Recognition in Multilingual LLM

## 소개

자연어처리2026
---
## 쓰인 라이브러리들

~
~~~
# 라이브러리를 다 쓰기엔 어려우니 ~
# wsl 환경, 리눅스 환경, colab A100 에서 돌리는 것을 권장합니다.
!pip install cuml cudf transformers scikit-learn huggingfacehub vllm lm-eval-harness
~~~
---
## 주요 코드들(연구에 실제로 쓰인 것들)

### textcleaning_sentiment_analysis: 문장 부호 제거 및 MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7을 활용하여 감성 점수 부여(-1(부정) ~ 1(긍정))

### model_pipeline: vllm과 lm-eval-harness를 활용한 제로샷 감성 분석 파이프라인

### VariousGraphs: 그래프 및 샘플링

### Shap_Analysis: Shap 결과 수행

### 데이터 폴더(xlsx, pkl, json, yaml)
아래 드라이브에 모든 데이터가 보관되어 있습니다. 파일의 사이즈가 큰 관계로 깃헙에 직접 커밋하는 대신 드라이브 링크를 올려두겠습니다.
**https://drive.google.com/drive/u/0/folders/1WDpZNOU2LlekVgcsp1lAIUf5OphZOfCH**
---
## 보조 코드들

### VectorizationAndClustering(1): 벡터화 및 클러스터링 결과를 바탕으로 샘플링을하려 했으나 vLLM 기반 환경에서 뛰어나게 잘 작동하여 실제로 쓰이지 않음

### 자연어처리이현준_: 데이터 탐색 결과를 위해 쓰려했으나 폐기
---
### 방법론
### 1. 수집한 데이터에 "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"으로 감정분석 시행
### 2. 감정분석 값이 입력된 데이터를 감정의 세기(abs(추론_감정_점수))에 따라 3개의 데이터로 분리(기준: 삼분위수)
### 3. 분리한 데이터를 각각 LLM(Exaone 4.0 1.2B, Gemma 4 E2B, Qwen 3.5 2B)에 넣고 제로샷 감성분석 수행후 정답인 것과 오답인 것으로 분류
### 4. 각 데이터를 nlp과 데이터상의 특징(ex. 상황및 맥락 레이블)으로 특징 분석(품사 태그 등 다양한 방식 사용)
### 5. nlp 분석 결과를 llmshap의 sliding window base shapley value으로 검증(random sampling으로 각각 1000개 정도 추출)

---
### Overleaf(LaTex)
### https://www.overleaf.com/5969293489qkqgppvtbsxw#454709
