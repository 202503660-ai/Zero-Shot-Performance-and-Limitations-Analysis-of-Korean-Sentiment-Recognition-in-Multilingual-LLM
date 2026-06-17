# Zero-Shot Performance and Limitations Analysis of Korean Sentiment Recognition in Multilingual LLM

## 소개

**2026년 1학기 자연어처리기반사회분석 수업의 다국어 LLM의 한국어 감정 인식 제로샷 성능 및 한계 분석의 레포지토리 입니다.**

---
## 쓰인 라이브러리들 및 실행 환경 필수 요구 사항

### 1. 실행 환경 요구 사항 (Google Colab A100 GPU 필수)
본 프로젝트는 **Python 3**, **허깅페이스 토큰**,**GPU 가속(CUDA)** 및 **대용량 VRAM(A100 40GB 이상)** 환경이 강제되는 작업들로 구성되어 있습니다.
- **Google Colab A100 GPU 환경에서의 실행이 무조건적(필수)으로 요구됩니다.**
- **Python 3 (구글 코랩 기본 런타임 환경인 Python 3.10 이상)** 버전에서 구동이 필수적이며, 이전 Python 버전과의 호환성은 보장하지 않습니다.
- Gemma와 Exaone의 접근을 위해 키를 발급받아야하며 키를 발급받고 난 후 **허깅페이스 토큰**을 생성하여 로그인 해야합니다.
- **필수 지정 사유**
  - 본 프로젝트에 사용된 `vllm`, `huggingface_hub`, 그리고 `cuda` 간의 매우 복잡한 패키지 의존성 및 버전 호환성 문제로 인해, Colab A100 환경이 제공하는 CUDA 런타임 환경이 아니면 설치/빌드 시 심각한 패키지 충돌이나 런타임 오류가 발생하여 실행이 불가능합니다.
  - 또한, vLLM을 활용한 로컬 다국어 거대 언어 모델(LLM) 제로샷 추론(`EXAONE-4.0-1.2B`, `Qwen3.5-2B`, `gemma-4-E2B-it`) 및 GPU 가속 기반 RAPIDS 라이브러리(`cuml`, `cudf`)를 활용한 고속 클러스터링/차원 축소 작업은 대용량 VRAM을 소모하므로, T4, V100 GPU나 일반 로컬 CPU/GPU 환경에서는 메모리 부족(OOM) 오류가 발생합니다.
  - Gemma와 Exaone의 경우 키가 없을 시 접근을 못할 수 있습니다.

### 라이브러리 설치 안내 및 버전 정보
**저장소내의 노트북 코드에 각 코드를 위한 라이브러리 설치코드가 있으니 그것을 활용하는 것을 추천합니다.**

아래 명령어를 사용하여 분석 및 모델 평가에 사용된 모든 외부 라이브러리를 설치할 수 있습니다. 완벽한 결과 재현을 위해 테스트가 완료된 아래 버전들의 설치를 강력히 권장합니다.
~~~
# 1) 기본 패키지 및 딥러닝/텍스트 마이닝 라이브러리 설치
!pip install vllm==0.6.3 lm-eval==0.4.4 scikit-learn==1.3.2 openpyxl==3.1.5 pyyaml==6.0.1 seaborn==0.13.2 gdown==5.2.0 ray==2.35.0 transformers==4.45.2 kiwipiepy==0.23.2 sentence-transformers==3.0.1 hdbscan==0.8.37 umap-learn==0.5.7 wordcloud==1.9.4 statsmodels==0.14.4 koreanize-matplotlib==0.1.1 cupy-cuda12x==12.2.0

# 2) GPU 가속 RAPIDS 라이브러리 (cudf, cuml) 설치 (Colab A100 환경 최적 설치 명령어)
!pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12==24.4.1 cuml-cu12==24.4.1
```
~~~
### 프로젝트 사용 라이브러리 상세 버전 정보 및 용도

| 라이브러리명 (Package) | 가져오기 이름 (Import) | 테스트 버전 | 주요 용도 및 역할 |
| :--- | :--- | :--- | :--- |
| **vllm** | `vllm` | `0.6.3` | 다국어 LLM(Exaone, Qwen, Gemma)의 고속 로컬 제로샷 추론 구동 및 GPU 자원 관리 |
| **lm-eval** | `lm_eval` | `0.4.4` | `lm-eval-harness` 기반 커스텀 제로샷 감성 분석 평가 파이프라인 구축 및 실행 |
| **transformers** | `transformers` | `4.45.2` | `mDeBERTa-v3` 기반 감성 점수 추론 파이프라인 생성 및 토크나이저 로드 |
| **scikit-learn** | `sklearn` | `1.3.2` | 모델 평가 지표(Accuracy, F1-Score) 계산 및 TF-IDF 텍스트 임베딩, 코사인 거리 연산 |
| **openpyxl** | `openpyxl` | `3.1.5` | 엑셀 데이터 파일(`.xlsx`) 읽기/쓰기 지원 |
| **pyyaml** | `yaml` | `6.0.1` | lm-eval용 커스텀 감성분석 평가 태스크 `.yaml` 설정 파일 처리 |
| **seaborn** | `seaborn` | `0.13.2` | 혼동 행렬(Confusion Matrix) 및 데이터 통계 분포 시각화 차트 생성 |
| **gdown** | `gdown` | `5.2.0` | 구글 드라이브로부터 대용량 원본 데이터셋 및 설정 파일 다운로드 |
| **ray** | `ray` | `2.35.0` | vLLM 가중치 병렬 로드 및 Ray 분산 자원 관리 |
| **kiwipiepy** | `kiwipiepy` | `0.23.2` | 한국어 형태소 분석기(Kiwi) 기반 명사/동사/조사 등 품사(POS) 태그 추출 및 빈도 분석 |
| **sentence-transformers** | `sentence_transformers` | `3.0.1` | SBERT 기반 한국어 문장 임베딩(Vectorization) 수행 |
| **hdbscan** | `hdbscan` | `0.8.37` | 차원 축소된 문장 벡터의 고밀도 군집 분석(HDBSCAN 클러스터링) 수행 (CPU 폴백용) |
| **umap-learn** | `umap` | `0.5.7` | 고차원 텍스트 벡터의 차원 축소(UMAP) 수행 (CPU 폴백용) |
| **wordcloud** | `wordcloud` | `1.9.4` | 품사 태그 분석 결과 빈도 시각화를 위한 워드클라우드 생성 |
| **statsmodels** | `statsmodels` | `0.14.4` | 데이터 통계적 경향성 및 정량 분석 |
| **koreanize-matplotlib** | `koreanize_matplotlib` | `0.1.1` | Matplotlib 그래프 시각화 시 한국어 폰트 깨짐 방지 및 나눔글꼴 세팅 자동화 |
| **cupy** | `cupy` | `12.2.0` | GPU 가속 기반 다차원 배열 연산 및 GPU 메모리 제어 |
| **cudf (RAPIDS)** | `cudf` | `24.4.1` | RAPIDS GPU 가속 기반 고속 데이터프레임 처리 (Colab A100 필수) |
| **cuml (RAPIDS)** | `cuml` | `24.4.1` | RAPIDS GPU 가속 기반 UMAP 차원 축소 및 HDBSCAN/KMeans 군집화 고속 연산 (Colab A100 필수) |
| **pandas** | `pandas` | `2.2.2` | 전체 대화 데이터셋 핸들링, 전처리 데이터 병합 및 정제 |
| **numpy** | `numpy` | `2.0.2` | 수치형 다차원 배열 연산 및 행렬 대수 연산 |
| **matplotlib** | `matplotlib` | `3.8.4` | UMAP 산점도 및 통계 분석 시각화 그래프 드로잉 |
| **scipy** | `scipy` | `1.11.4` | 임베딩 행렬 거리 연산 및 통계 계산 보조 |
| **torch** | `torch` | `2.4.0` | PyTorch 딥러닝 프레임워크 (vLLM 구동 및 HuggingFace 모델 로딩 CUDA 백엔드) |
| **tqdm** | `tqdm` | `4.67.3` | 데이터 가공 및 모델 제로샷 추론 진행률(Progress Bar) 출력 |
| **huggingface_hub** | `huggingface_hub` | `0.24.6` | 허깅페이스 API 로그인 및 로컬 모델 가중치(EXAONE 등) 호출 및 다운로드 |

> [!NOTE]
> 파이썬 내장 표준 라이브러리(`collections`, `itertools`, `gc`, `os`, `re`, `sys`, `time`, `subprocess`, `string`)는 별도의 설치 과정 없이 즉시 임포트하여 사용됩니다.
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

### .gitattributes
Git LFS 사용을 위한 git 설정이 들어있음.

---
## 연구에 사용된 모든 데이터(xlsx, pkl, json, yaml)가 저장된 드라이브 및 데이터셋 소개
아래 드라이브에 연구에 사용된 모든 데이터가 보관되어 있습니다. 파일의 사이즈가 큰 관계로 깃헙에 직접 커밋하는 대신 아래 드라이브 링크를 올려두겠습니다.

**https://drive.google.com/drive/u/0/folders/1WDpZNOU2LlekVgcsp1lAIUf5OphZOfCH**

### 사용한 데이터셋 소개 (`resultwithtag.xlsx` 기준)
본 프로젝트는 다국어 거대 언어 모델(LLM)의 한국어 감성 인식 제로샷 성능 평가 및 통계적 한계 분석을 위해 구축된 전처리/가공 데이터셋을 사용합니다.

- **데이터셋 출처**: **[AI-Hub 감성 대화 말뭉치](https://aihub.or.kr/)**를 원천 데이터로 삼고 있습니다.
- **데이터셋 규모**: 총 **58,271건**의 대화/문장 레코드로 구성되어 있습니다.
- **데이터 가공 작업**:
  1. **감성 점수 부여**: 문장별 정량적 감성 강도를 산출하고자 NLI 모델(`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`)을 적용하여 `-1.0`(부정) ~ `1.0`(긍정) 사이의 실수값(`추론_감성_점수`)을 수록했습니다.
  2. **감성 세기 범주화**: 감성 점수의 절대값(`|추론_감성_점수|`)에 대해 삼분위수를 기준점으로 삼아 감성 세기를 '상', '중', '하' 3개 그룹으로 라벨링했습니다.
  3. **형태소 형태 분석**: `kiwipiepy`(Kiwi 토크나이저)를 통해 명사, 용언, 부사, 조사 등 주요 품사 태그의 출현 빈도를 개별 컬럼으로 집계하여 병합했습니다.
  4. **LLM 추론 성능 비교**: 다국어 LLM(`EXAONE-4.0-1.2B`, `Qwen3.5-2B`, `gemma-4-E2B-it`)의 제로샷 추론 정답(감정_대분류) 여부와 오답 여부를 각각 매핑하여 수록했습니다.

#### 데이터셋 주요 컬럼 상세 가이드

* **연령 / 성별 / 상황키워드 / 신체질환**: 대화 화자의 인구통계학적 정보 및 상황 메타데이터 (예: 청소년, 여성, 대인관계, 무질환 등)
* **감정_대분류 / 감정_소분류**: 대화 원본에 매핑된 실제 정답 감정 레이블 (예: 불안, 두려운 등)
* **사람문장(1 ~ 3) / 시스템문장(1 ~ 3)**: 원천 데이터인 3-Turn 사용자 및 시스템 응답 대화 텍스트
* **사람문장_통합 / 전처리된_문장**: 형태소 분석 및 기호 정제 가공을 거친 단일 텍스트 데이터 (LLM 추론 입력값)
* **추론_감성_점수 / 전처리된_추론_감성_점수**: mDeBERTa-v3 NLI 모델로 구한 긍/부정 감성 수치 (-1.0 ~ 1.0)
* **추론_감성_점수_절댓값**: 감성 강도 분석에 사용되는 감성 점수의 절대치
* **감성_세기**: 삼분위수 기반의 감성 점수 절대값 그룹 ('상', '중', '하')
* **LLM 추론 결과 및 정답 여부** (`Exaone 추론`, `Exaone 정답유무`, `Qwen 추론`, `Qwen 정답유무`, `Gemma 추론`, `Gemma 정답유무`): 각 언어 모델의 예측 라벨 및 실제 정답(감정_대분류)과의 매칭 결과
* **품사 태그 수 (Kiwi 형태소 분류 컬럼들)**: `명사`, `수사`, `용언`, `관형사`, `부사`, `감탄사`, `조사`, `어미`, `접두사`, `접미사`, `어근`, `부호`, `외국어`, `숫자`, `특수문자` 등 각 대화문에서 분리된 형태소의 출현 횟수 (다변량 분석 및 군집 분석 피처로 활용)

---
### 방법론
### 1. 수집한 데이터에 "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"으로 감정분석 시행
### 2. 감정분석 값이 입력된 데이터를 감정의 세기(abs(추론_감정_점수))에 따라 3개의 데이터로 분리(기준: 삼분위수)
### 3. 분리한 데이터를 각각 LLM(Exaone 4.0 1.2B, Gemma 4 E2B, Qwen 3.5 2B)에 넣고 제로샷 감성분석 수행후 정답인 것과 오답인 것으로 분류
### 4. 각 데이터를 nlp 분석기법과 데이터상의 특징(ex. HDBSCAN 클러스터링)으로 분석(품사 태그 등 다양한 방식 사용)

---
### Overleaf(LaTex)
### https://www.overleaf.com/5969293489qkqgppvtbsxw#454709
