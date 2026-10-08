# AI 기본법 QA 백엔드 (RAG Mini Project)

**인공지능기본법을 잘 모르는 사용자도 질문으로 쉽게 이해할 수 있는, 근거 기반 QA API**

법률 PDF를 법률 구조(장·절·조·항·호·목)에 맞게 청킹해 Qdrant에 저장하고,
Hybrid 검색 + Reranking으로 관련 조문을 찾은 뒤, **찾은 조문만 근거로** 답변과 출처를 돌려준다.

- 데이터: 인공지능 발전과 신뢰 기반 조성 등에 관한 기본법 (법률 제21311호, 2026. 7. 21. 시행)
- 결과물: `POST /ask` → `{"answer": ..., "sources": [{"article": "제2조 제4호", ...}]}`

---

## 1. 전체 흐름

```text
[Offline] 문서를 검색 가능한 지식으로 만든다            [Online] 질문에 근거를 찾아 답한다

PDF                                                    질문
 ↓ loader.py      Docling으로 읽기 + 정리               ↓ retriever.py   Dense + BM25 → RRF (후보 20개)
 ↓ chunker.py     조/항/호 구조 기반 청킹 (174개)       ↓ reranker.py    Cross-Encoder로 재정렬 (상위 5개)
 ↓ vectorstore.py 임베딩 → Qdrant 저장                  ↓ pipeline.py    Context 구성 → LLM 근거 기반 답변
Qdrant (ai_basic_law 컬렉션)                            ↓ app/main.py    POST /ask
                                                       답변 + 출처
```

## 2. 폴더 구조

```text
├── data/ai_basic_law/          법령 PDF (시행일별로 보관, 최신본을 기본으로 사용)
├── rag/
│   ├── loader.py               PDF → 구조 단위로 줄바꿈된 텍스트 + 법령 metadata
│   ├── chunker.py              조 단위 청킹, 긴 조는 항 → 호 → (목) 단위로 분할
│   ├── vectorstore.py          QdrantVectorStore 저장(build) / 불러오기(load)
│   ├── retriever.py            dense / bm25 / hybrid(RRF) / multi_query 검색
│   ├── reranker.py             Qwen3-Reranker-0.6B로 후보 재정렬
│   └── pipeline.py             검색 → Rerank → Small-to-Big → Grounded Answer
├── app/main.py                 FastAPI 서버 (POST /ask)
├── eval/
│   ├── golden_set.jsonl        평가용 질문 30개 (정답 조문, 참고 답변)
│   ├── evaluate.py             검색/답변 평가
│   └── results/                평가 결과 JSON (실행할 때마다 저장)
├── common/                     설정, LLM/임베딩, Qdrant 클라이언트
├── docker-qdrant/              Qdrant 실행용 docker-compose
└── notebook/qdrant_test.ipynb  Qdrant + RAG 기본 실습 노트북
```

## 3. 실행 방법

```bash
# 0) 패키지 설치
uv sync

# 1) .env 작성 (프로젝트 루트)
#    LLM_API_KEY=...
#    LLM_BASE_URL=...

# 2) Qdrant 실행
cd docker-qdrant && docker compose up -d && cd ..

# 3) 문서 → Qdrant 저장 (최초 1회, 청킹을 바꾼 뒤에도 다시 실행. 약 20초)
uv run python -m rag.vectorstore

# 4) API 서버 실행 (기동에 약 15초: Reranker 모델 로딩)
uv run uvicorn app.main:app --reload
#    → http://localhost:8000/docs 에서 직접 질문해 볼 수 있다.

# 5) 평가
uv run python -m eval.evaluate --retrieval-only   # 검색 평가만 (약 30초)
uv run python -m eval.evaluate                    # 검색 + 답변 평가 (약 5분)
```

각 모듈은 단독으로 실행해 중간 결과를 확인할 수 있다.

```bash
uv run python -m rag.loader       # 법령 metadata와 본문 앞부분
uv run python -m rag.chunker      # 청크 수, 토큰 분포, 누락 점검
uv run python -m rag.retriever    # 검색 방식별 정답 순위 비교
uv run python -m rag.reranker     # Rerank 전후 순위 비교
uv run python -m rag.pipeline "고영향 인공지능이란 무엇인가요?"
```

### API 예시

```bash
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "국가인공지능전략위원회 위원장은 누구야?"}'
```

```json
{
  "answer": "국가인공지능전략위원회의 위원장은 대통령입니다.[S1] 위원장은 위원회를 대표하고 사무를 총괄하며,[S3] 회의를 소집하고 그 의장이 됩니다.[S5]",
  "sources": [
    {"article": "제7조 제3항", "title": "국가인공지능전략위원회", "content": "제7조(국가인공지능전략위원회)\n③ 위원회의 위원장은 대통령이 되고, ..."},
    {"article": "제7조 제5항", "title": "국가인공지능전략위원회", "content": "..."},
    {"article": "제7조 제10항", "title": "국가인공지능전략위원회", "content": "..."}
  ],
  "is_answerable": true
}
```

법에 없는 내용을 물으면 지어내지 않고 거절한다.

```json
{"answer": "제공된 법률 조문만으로는 답변하기 어렵습니다.", "sources": [], "is_answerable": false}
```

---

## 4. 설계 포인트

### 4.1 문서 로딩 (loader.py)
Docling의 Markdown 변환을 그대로 쓰면 법률 정보가 깨져서, Docling이 인식한 **원본 텍스트(`orig`)를 직접 정리**했다.

| 문제 | 해결 |
|---|---|
| Markdown 변환이 호 번호를 다시 매김 (목 `가.` → `5. 가.`) | 원본 텍스트(`item.orig`) 사용 |
| 페이지 경계에서 문장이 잘림 | 구조 기호(제N조, ①, 1., 가.)로 시작하지 않는 줄은 앞 줄에 붙임 |
| 최신본에서 머리말(법령명·시행일)이 제2조 중간에 끼어듦 | 위치가 아니라 패턴으로 머리말을 골라내 metadata로 분리 |
| 가지번호 호(`2의2.`)와 두 항(②③)이 한 줄로 붙음 | 줄 중간의 항/가지번호 기호에서 줄을 나눔 |

### 4.2 청킹 (chunker.py): 조 단위 + 긴 조만 분할
수업의 HybridChunker(토큰 기준)로 자르면 51개 중 48개 청크에 "제N조"가 없어 **출처를 표시할 수 없었다.**
그래서 법률 구조를 기준으로 직접 나눴다.

1. 조 하나 = 청크 하나 (기본)
2. 500토큰을 넘는 조는 항 단위로, 항이 없으면 호 단위로 분할
3. 그래도 긴 호에 목이 있으면 "호 본문"과 "호 본문 + 목 목록"으로 분할
4. 나눈 조각 앞에는 `제N조(제목)`과 상위 문장을 붙여 문맥을 유지

모든 청크에는 `citation`(예: `제8조 제1항 제2호의2`), `chapter`, `article`, `paragraph`, `item` 등의 metadata가 붙는다.
`check_chunks()`로 원문의 모든 줄이 청크에 들어갔는지 자동으로 점검한다.

### 4.3 검색 (retriever.py, reranker.py)
- **Dense**: Qdrant 코사인 유사도 (text-embedding-3-small)
- **BM25**: 한국어는 조사가 붙어서("인공지능이란" ≠ "인공지능은") 띄어쓰기 대신 **두 글자(bigram) 단위**로 토큰화했다.
- **Hybrid**: Dense와 BM25의 순위를 RRF로 결합
- **Rerank**: Hybrid 후보 20개를 `Qwen/Qwen3-Reranker-0.6B`(Cross-Encoder)로 다시 평가해 상위 5개 선택

### 4.4 답변 (pipeline.py)
- 검색된 청크에 `[S1] 제7조 제3항 (국가인공지능전략위원회)`처럼 번호를 붙여 Context를 만든다.
- LLM은 `RagAnswer(answer, source_ids, is_answerable)` 형식으로 답한다. 출처는 LLM이 쓴 조문 번호가 아니라 **코드가 원본 metadata에서 가져온다.**
- **Small-to-Big**: 짧은 "호 본문" 청크가 검색되면, 답변에는 같은 호의 "목 목록 포함" 청크를 넘긴다.
- 법이 세부 사항을 "대통령령으로 정한다"고 위임한 경우, 거절하지 않고 위임 사실을 설명한다.

---

## 5. 평가 결과와 개선 과정

`eval/golden_set.jsonl`의 30문항(정답 조문이 있는 28문항 + 답변불가 2문항), K=5

### 5.1 검색 방식 비교

| 방식 | Hit@5 | Recall@5 | MRR |
|---|:-:|:-:|:-:|
| dense | 0.893 | 0.857 | 0.736 |
| bm25 | 0.929 | 0.929 | 0.792 |
| hybrid (RRF) | 0.929 | 0.911 | 0.824 |
| **hybrid + rerank** | **1.000** | **1.000** | **0.893** |

법률 질문은 "국내대리인", "과태료"처럼 용어가 정확히 일치하는 경우가 많아 BM25가 Dense보다 강했고,
둘을 결합한 뒤 Reranker로 다시 정렬했을 때 가장 좋았다.

### 5.2 답변 품질 (hybrid + rerank, LLM-as-a-Judge)

| 지표 | 결과 |
|---|:-:|
| 정답성 (1~5) | 4.96 |
| 관련성 (1~5) | 5.00 |
| 근거 충실성 (1~5) | 5.00 |
| Hallucination 비율 | 0% |
| 답변불가 질문 거절 | 2 / 2 |

### 5.3 개선 사례

**① 청킹 개선: "고영향 인공지능이란?"이 검색되지 않던 문제**

정답인 제2조 제4호는 정의 문장 뒤에 11개 영역 목록(「에너지법」, 「먹는물관리법」 …)이 붙은 914토큰짜리 청크였다.
임베딩이 "법률 목록" 쪽 의미로 희석되어 Dense 검색에서 173개 중 134등이었다.

| 청크 모양 | 질문과의 유사도 |
|---|:-:|
| 정의 문장 + 목 11개 (기존) | 0.29 |
| 정의 문장만 | 0.56 |

→ "호 본문"과 "호 본문 + 목 목록"으로 청크를 나눈 결과

| | dense | bm25 | hybrid | multi_query |
|---|:-:|:-:|:-:|:-:|
| 수정 전 순위 | 134등 | 21등 | 순위 밖 | 순위 밖 |
| 수정 후 순위 | 2등 | 1등 | 1등 | 2등 |

그 대신 답변에서 영역 목록이 빠지는 부작용이 생겨, **Small-to-Big**(검색은 작은 청크, 답변은 큰 청크)으로 보완했다.

**② Reranker: 정답이 아래쪽에 있던 문제**

"위원장은 누구야?" 질문에서 Hybrid는 정답(제7조 제3항)을 4등으로 찾았다. Reranker가 "위원장"을 언급하는 여러 항 중
"대통령이 된다"라고 직접 답하는 항을 1등으로 올렸다. (MRR 0.824 → 0.893)

**③ 프롬프트: 법이 위임한 내용을 통째로 거절하던 문제**

"누적 연산량 기준은 구체적으로 얼마야?"(q30)에 "답변하기 어렵습니다"라고만 답했다.
"대통령령에 위임된 것도 답이다"라는 규칙을 추가해, 위임 사실과 출처(제32조)를 설명하도록 바꿨다. (정답성 3 → 5)

---

## 6. 한계와 주의사항

- **평가 점수가 실제보다 높게 나왔을 수 있다.** Golden Set 질문을 법 조문을 보고 만들었고, 채점(Judge)도 답변과 같은 LLM이 해서 후하게 나오는 경향이 있다. Judge 점수는 실행마다 ±1점 정도 흔들린다. 팀 검토와 질문 보강(목표 40~50문항)이 필요하다.
- **Multi Query는 결과가 매번 다르다.** LLM이 만드는 검색 질문이 실행마다 달라서, 평가에서는 기본 방식에서 제외했다.
- **API 호출 제한(분당 30회).** 한도를 넘으면 에러 없이 약 60초 멈췄다가 재시도한다. 질문 임베딩은 캐시하고 묶어서 보내며, 평가는 호출 사이에 2.5초 간격을 둔다. 질문 하나에 임베딩 1회와 LLM 1회를 쓴다.
- **PDF 줄바꿈 때문에 단어 사이에 공백이 남는다**("말 한다", "인공지 능"). 임베딩에는 영향이 적고, BM25는 bigram 토큰화로 영향을 줄였다.
- **법령이 개정되면** 새 PDF를 `data/ai_basic_law/`에 넣고 `rag/loader.py`의 `DEFAULT_PDF_PATH`를 바꾼 뒤 `uv run python -m rag.vectorstore`를 다시 실행한다. 판본이 바뀌면 PDF 레이아웃이 달라질 수 있으니 `uv run python -m rag.chunker`의 점검 결과를 꼭 확인한다.

---

## 부록. 개발 환경 설정 (uv)

`notebook/`과 `app/`, `rag/`가 프로젝트 루트의 `.venv` 하나를 공유한다.

```bash
uv sync                         # pyproject.toml / uv.lock 기준으로 환경 복원
uv add 패키지명                  # 패키지 추가
uv run jupyter lab              # 노트북 실행
uv run python -m ipykernel install --user --name rag-minipjt --display-name "rag-minipjt"
```

`common/`, `app/`, `rag/`는 `pyproject.toml`의 hatch 설정으로 프로젝트 자체를 editable 패키지로 설치해,
노트북이 `notebook/` 폴더에서 실행되어도 `from common... import`, `from rag... import`가 동작한다.

```toml
[tool.hatch.build.targets.wheel]
packages = ["common", "app", "rag"]
```

- 소스를 수정한 뒤에는 `uv sync` 없이 노트북 커널만 재시작하면 된다.
- `.env`를 수정한 경우에도 커널을 재시작해야 반영된다. (`common/config.py`가 import될 때 값을 읽기 때문)
- Git에는 `pyproject.toml`, `uv.lock`을 포함하고 `.venv/`, `.env`는 제외한다.
