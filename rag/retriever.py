"""질문과 관련된 법률 Chunk를 찾는다. (Online 단계의 첫 번째)

수업 노트북과 같은 구조
  05_retrieval_engineering: dense_search → bm25_search → rrf → hybrid_search
  04_multi_query:           LLM으로 검색 질문 여러 개 만들기 → 각각 검색 → 결과 합치기

pipeline에서는 retrieve(question, method=...) 하나만 부르면 된다.

실행 (검색 방식별 결과 비교)
  uv run python -m rag.retriever
"""

import re
from functools import lru_cache

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from common.ai_model import get_llm_model
from common.config import QDRANT_COLLECTION
from rag.vectorstore import load_vector_store


# ---------------------------------------------------------------------------
# 1. Dense 검색 (의미 검색)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_vector_store():
    """Qdrant 연결을 한 번만 만들고 재사용한다.

    @lru_cache: 처음 호출한 결과를 기억해 두었다가, 다음 호출부터는 다시 만들지 않고 그대로 돌려준다.
    """
    return load_vector_store()


# 이미 임베딩한 질문의 벡터를 기억해 두는 저장소 {질문: 벡터}
_query_vectors: dict[str, list[float]] = {}


def embed_queries(queries: list[str]) -> list[list[float]]:
    """질문들을 임베딩한다.

    MonoRouter는 분당 30회까지만 호출할 수 있어서 호출 수를 줄인다.
    - 이미 임베딩한 질문은 다시 요청하지 않는다. (hybrid, multi_query가 같은 질문을 또 검색할 때)
    - 새 질문 여러 개는 API 한 번에 묶어서 보낸다. (OpenAI 임베딩은 문서용/질문용 결과가 같다)
    """
    new_queries = [q for q in dict.fromkeys(queries) if q not in _query_vectors]  # 중복 제거
    if new_queries:
        vectors = get_vector_store().embeddings.embed_documents(new_queries)
        _query_vectors.update(zip(new_queries, vectors))
    return [_query_vectors[q] for q in queries]


def dense_search(query: str, k: int = 5) -> list[Document]:
    # 질문을 임베딩으로 바꾼 뒤, 코사인 유사도가 높은 Chunk k개를 찾는다.
    # (수업 코드의 similarity_search(query)와 같지만, 임베딩을 직접 만들어 재사용한다)
    query_vector = embed_queries([query])[0]
    return get_vector_store().similarity_search_by_vector(query_vector, k=k)


# ---------------------------------------------------------------------------
# 2. BM25 검색 (키워드 검색)
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """BM25용 토큰화: 공백과 기호를 지우고 두 글자씩 자른다. (bigram)

    수업 코드는 띄어쓰기 단위(\\w+)로 나눴지만, 한국어는 조사가 붙어서
    "인공지능이란"과 "인공지능은"이 다른 단어가 되어 버린다.
    두 글자씩 자르면 둘 다 "인공", "공지", "지능"을 포함하므로 서로 맞출 수 있다.
    PDF 줄바꿈 때문에 생긴 공백("인공지 능")도 지운 뒤 자르므로 영향을 받지 않는다.

    예) "고영향 인공지능이란" → ["고영", "영향", "향인", "인공", "공지", "지능", "능이", "이란"]
    """
    text = re.sub(r"[^\w]", "", text.lower())  # 글자/숫자가 아닌 것(공백, 기호)을 지운다.
    return [text[i:i + 2] for i in range(len(text) - 1)]


@lru_cache(maxsize=1)
def get_bm25() -> tuple[BM25Okapi, list[Document]]:
    """Qdrant에 저장된 Chunk를 모두 가져와 BM25 검색기를 만든다.

    Dense와 BM25가 같은 문서 집합을 쓰도록 Qdrant에서 가져온다. (수업 코드의 vector_store.get()과 같은 역할)
    """
    points, _ = get_vector_store().client.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=10_000,       # 전체를 한 번에 가져온다. (현재 173개)
        with_payload=True,
    )
    corpus_docs = [
        Document(page_content=p.payload["page_content"], metadata=p.payload["metadata"])
        for p in points
    ]
    corpus_docs.sort(key=lambda doc: doc.metadata["chunk_id"])  # 순서를 고정해 결과가 매번 같게 한다.

    bm25 = BM25Okapi([tokenize(doc.page_content) for doc in corpus_docs])
    return bm25, corpus_docs


def bm25_search(query: str, k: int = 5) -> list[Document]:
    bm25, corpus_docs = get_bm25()
    # 질문도 Chunk와 같은 방식으로 토큰화한 뒤, BM25 점수 상위 k개를 가져온다.
    return bm25.get_top_n(tokenize(query), corpus_docs, n=k)


# ---------------------------------------------------------------------------
# 3. RRF + Hybrid 검색
# ---------------------------------------------------------------------------
def rrf(*results: list[Document], k: int = 60) -> list[Document]:
    """여러 검색 결과의 순위를 RRF(Reciprocal Rank Fusion)로 합친다.

    수업 코드는 rrf(dense_docs, bm25_docs)처럼 두 개만 받았지만,
    Multi Query에서도 쓰기 위해 결과 목록을 몇 개든 받을 수 있게 했다.
    """
    scores = {}     # chunk_id별 RRF 점수
    documents = {}  # chunk_id로 원래 Document를 다시 찾기 위한 저장소

    for result in results:
        for rank, doc in enumerate(result, start=1):
            chunk_id = doc.metadata["chunk_id"]
            if chunk_id not in scores:
                scores[chunk_id] = 0
                documents[chunk_id] = doc
            # RRF 공식: 검색기마다 1 / (k + 순위)를 더한다. 여러 검색기에서 상위권이면 점수가 커진다.
            scores[chunk_id] += 1 / (k + rank)

    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    return [documents[chunk_id] for chunk_id in ranked_ids]


def hybrid_search(query: str, k: int = 5, candidates: int = 10) -> list[Document]:
    """Dense와 BM25에서 후보를 candidates개씩 가져와 RRF로 합친 뒤 상위 k개를 쓴다."""
    dense_docs = dense_search(query, k=candidates)
    bm25_docs = bm25_search(query, k=candidates)
    return rrf(dense_docs, bm25_docs)[:k]


# ---------------------------------------------------------------------------
# 4. Multi Query 검색
# ---------------------------------------------------------------------------
MULTI_QUERY_PROMPT = """
다음 질문을 인공지능기본법 조문 검색에 사용할 검색 질문 {n}개로 바꿔주세요.

규칙:
- 원래 질문의 의미를 유지하세요.
- 새로운 주제를 추가하지 마세요.
- 서로 다른 표현으로 작성하세요. 법률 용어로 바꾼 표현도 하나 포함하세요.
- 한 줄에 질문 하나만 작성하세요.
- 번호는 붙이지 마세요.

원본 질문:
{question}
"""


def generate_queries(question: str, n: int = 3) -> list[str]:
    """LLM에게 같은 의미의 검색 질문 n개를 만들게 한다."""
    llm = get_llm_model(max_tokens=300)
    response = llm.invoke(MULTI_QUERY_PROMPT.format(n=n, question=question))
    # 응답을 줄 단위로 나누고 빈 줄은 버린다.
    return [line.strip() for line in response.content.splitlines() if line.strip()][:n]


def multi_query_search(question: str, k: int = 5, candidates: int = 10) -> list[Document]:
    """원본 질문 + LLM이 만든 질문 3개로 각각 Hybrid 검색한 뒤 RRF로 합친다."""
    queries = [question] + generate_queries(question)  # 원본 질문도 버리지 않고 함께 쓴다.
    embed_queries(queries)  # 질문 4개를 API 한 번에 임베딩해 둔다. (아래 검색에서 재사용)
    results = [hybrid_search(query, k=candidates, candidates=candidates) for query in queries]
    return rrf(*results)[:k]


# ---------------------------------------------------------------------------
# pipeline에서 부르는 입구
# ---------------------------------------------------------------------------
SEARCH_METHODS = {
    "dense": dense_search,
    "bm25": bm25_search,
    "hybrid": hybrid_search,
    "multi_query": multi_query_search,
}


def retrieve(question: str, k: int = 5, method: str = "hybrid") -> list[Document]:
    """질문과 관련된 Chunk k개를 찾는다. method: dense / bm25 / hybrid / multi_query"""
    return SEARCH_METHODS[method](question, k=k)


# ---------------------------------------------------------------------------
# 동작 확인용 질문 (정식 평가는 eval/에서 Golden Set으로 한다)
# ---------------------------------------------------------------------------
SAMPLE_QUESTIONS = [
    ("고영향 인공지능이란 무엇인가요?", "제2조 제4호"),
    ("이 법의 목적은?", "제1조"),
    ("생성형 인공지능이란?", "제2조 제5호"),
    ("국가인공지능전략위원회 위원장은 누구야?", "제7조 제3항"),
    ("과태료는 최대 얼마야?", "제43조"),
    ("인공지능 기본계획은 몇 년마다 세워?", "제6조 제1항"),
    ("해외 AI 회사는 국내대리인을 둬야 해?", "제36조"),
    ("생성형 AI 결과물에 표시를 해야 하나요?", "제31조"),
]


def citation_matches(citation: str, answer: str) -> bool:
    """Chunk의 citation이 정답 조문과 맞는지 판단한다. (법률 구조의 포함 관계를 양방향으로 본다)

    - Chunk가 정답의 일부:   정답 "제43조"      ← Chunk "제43조 제1항"
    - Chunk가 정답을 포함:   정답 "제4조 제1항" ← Chunk "제4조" (짧은 조는 조 전체가 Chunk 하나)
    Golden Set은 항 단위로 정확히 적어 두고, 청킹 방식이 바뀌어도 같은 기준으로 비교하기 위함이다.
    """
    return (
        citation == answer
        or citation.startswith(answer + " ")
        or answer.startswith(citation + " ")
    )


def rank_of(docs: list[Document], answer: str) -> str:
    """정답 조문이 docs에서 몇 등인지 돌려준다. 없으면 '-'."""
    for rank, doc in enumerate(docs, start=1):
        if citation_matches(doc.metadata["citation"], answer):
            return str(rank)
    return "-"


if __name__ == "__main__":
    # 검색 방식별로 정답 조문이 몇 등에 나오는지 비교한다.
    K = 5
    print(f"정답 조문의 순위 (top-{K} 안에 없으면 '-')\n")
    print(f"{'질문':<34}{'정답':<12}" + "".join(f"{m:>12}" for m in SEARCH_METHODS))
    for question, answer in SAMPLE_QUESTIONS:
        ranks = [rank_of(retrieve(question, k=K, method=m), answer) for m in SEARCH_METHODS]
        print(f"{question:<34}{answer:<12}" + "".join(f"{r:>12}" for r in ranks))

    get_vector_store().client.close()
