# Zero-Shot Performance and Limitations Analysis of Korean Sentiment Recognition in Multilingual LLM

## 소개

**2026년 1학기 자연어처리기반사회분석 수업의 다국어 LLM의 한국어 감정 인식 제로샷 성능 및 한계 분석의 레포지토리 입니다.**

---
## 실행 환경 권장
이 프로젝트는 **Google Colab A100 GPU** 환경에 최적화되어 있습니다.
- **Google Colab A100 GPU 환경을 권장합니다.**
- RAPIDS 라이브러리(`cuml`, `cudf`)를 활용한 고속 클러스터링/차원 축소와 `vLLM` 및 `lm-eval-harness`를 이용한 LLM 제로샷 추론을 실행하기 위해서는 A100 환경이 필수적입니다.
- A100의 CUDA 버전에 맞추어 vLLM과 huggingface hub 라이브
- WSL 환경 혹은 Linux GPU 서버 환경에서도 구동 가능하지만, 라이브러리 의존성 문제 해결을 위해 **Google Colab A100 GPU**에서 구동하는 것을 강력하게 권장합니다.


### 쓰인 라이브러리들

~
~~~
# 라이브러리를 다 쓰기엔 어려우니 ~
# wsl 환경, 리눅스 환경, colab A100 에서 돌리는 것을 권장합니다.
!pip install cuml cudf transformers scikit-learn huggingfacehub vllm lm-eval-harness
~~~
---
## 파일 및 폴더 설명

### textcleaning_sentiment_analysis.ipynb
문장 부호 제거 및 MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7을 활용하여 감성 점수(-1(부정) ~ 1(긍정)) 부여.

### model_pipeline_fixed.ipynb
vllm과 lm-eval-harness를 활용한 제로샷 감성 분석 파이프라인.

### VariousGraphs.ipynb
그래프와 표 및 HDBSCAN를 활용한 클러스터링과 Kiwi 토크나이저를 활용한 태그 분석 시행.

### VectorizationAndClustering.ipynb
벡터화 및 클러스터링 결과를 바탕으로 샘플링을하려 했으나 vLLM 기반 환경에서 뛰어나게 잘 작동하여 샘플링을 사용하지 않음. 대신 이 코드에서 MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7을 활용하여 부여한 감성 점수에 따라 삼분위수로 나누어 감성_세기(상, 중, 하로 구분)라는 레이블을 달음. 

### 자연어처리이현준.ipynb
데이터 탐색에 쓰임.

### resultwithtag.xlsx
각 문장의 태그의 수(Kiwi 토크나이저의 분류에서 크게 묶어 사용)까지 기입된 최종 데이터.

### figures 폴더
그래프 png 파일들이 저장되어 있음.

### paper 폴더
연구에 참고한 선행연구 pdf들이 저장되어 있음.

---
## 연구에 사용된 모든 데이터(xlsx, pkl, json, yaml)가 저장된 드라이브
아래 드라이브에 연구에 사용된 모든 데이터가 보관되어 있습니다. 파일의 사이즈가 큰 관계로 깃헙에 직접 커밋하는 대신 아래 드라이브 링크를 올려두겠습니다.

**https://drive.google.com/drive/u/0/folders/1WDpZNOU2LlekVgcsp1lAIUf5OphZOfCH**

---
### 방법론
### 1. 수집한 데이터에 "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"으로 감정분석 시행
### 2. 감정분석 값이 입력된 데이터를 감정의 세기(abs(추론_감정_점수))에 따라 3개의 데이터로 분리(기준: 삼분위수)
### 3. 분리한 데이터를 각각 LLM(Exaone 4.0 1.2B, Gemma 4 E2B, Qwen 3.5 2B)에 넣고 제로샷 감성분석 수행후 정답인 것과 오답인 것으로 분류
### 4. 각 데이터를 nlp 분석기법과 데이터상의 특징(ex. HDBSCAN 클러스터링)으로 분석(품사 태그 등 다양한 방식 사용)

---
### Overleaf(LaTex)
### https://www.overleaf.com/5969293489qkqgppvtbsxw#454709
