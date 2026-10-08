"""Retriever가 가져온 후보 Chunk를 Reranker로 다시 줄 세운다.

수업 06_reranking 노트북과 같은 구조
  Retriever 후보 (넉넉하게) → (질문, 문서) 쌍 만들기 → Reranker 점수 → 점수순 정렬 → 최종 Top-k

Retriever와 Reranker의 차이
  - Retriever(임베딩): 질문과 문서를 "따로" 벡터로 바꿔 비교한다. 빠르지만 대략적이다.
  - Reranker(Cross-Encoder): 질문과 문서를 "함께" 읽고 관련성을 판단한다. 정확하지만 느리다.
  → 그래서 Retriever로 후보를 좁힌 다음, 후보만 Reranker로 다시 평가한다.

실행 (Hybrid 결과와 Rerank 결과 비교)
  uv run python -m rag.reranker
"""

from functools import lru_cache

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# 수업과 같은 모델. 첫 실행 때 Hugging Face에서 내려받는다. (GPU가 있으면 자동으로 GPU 사용)
RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """모델을 한 번만 불러오고 재사용한다. (불러오는 데 약 10초 걸린다)"""
    return CrossEncoder(RERANKER_MODEL)


def rerank(query: str, docs: list[Document], top_k: int = 5) -> list[Document]:
    """후보 문서를 질문과의 관련성 점수로 다시 정렬해 상위 top_k개를 돌려준다.

    각 문서의 metadata에 rerank_score를 남긴다. (높을수록 관련 있음)
    """
    if not docs:
        return []

    # 1) 후보 문서마다 같은 질문을 붙여 (질문, 문서) 쌍을 만든다.
    pairs = [(query, doc.page_content) for doc in docs]

    # 2) 쌍마다 관련성 점수를 계산한다.
    scores = get_reranker().predict(pairs)

    # 3) 점수가 높은 순서로 정렬한다.
    reranked = sorted(zip(docs, scores), key=lambda item: float(item[1]), reverse=True)

    # 4) 상위 top_k개를 고르고 점수를 metadata에 남긴다.
    #    원래 Document를 고치지 않고 복사본을 만든다.
    #    (BM25 검색 결과는 메모리에 저장된 문서를 그대로 돌려주므로, 고치면 다음 검색에도 남는다)
    return [
        Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "rerank_score": float(score)},
        )
        for doc, score in reranked[:top_k]
    ]


if __name__ == "__main__":
    from rag.retriever import SAMPLE_QUESTIONS, get_vector_store, hybrid_search, rank_of

    K = 5             # 최종으로 쓸 문서 수
    CANDIDATES = 20   # Reranker에 넘길 후보 수 (수업은 8개. 넉넉할수록 정답이 후보에 들어올 확률이 높다)

    print(f"정답 조문의 순위 (top-{K} 안에 없으면 '-')\n")
    print(f"{'질문':<34}{'정답':<12}{'hybrid':>10}{'hybrid+rerank':>16}")
    for question, answer in SAMPLE_QUESTIONS:
        candidates = hybrid_search(question, k=CANDIDATES, candidates=CANDIDATES)
        before = candidates[:K]                       # Reranker 없이 쓸 때
        after = rerank(question, candidates, top_k=K)  # Reranker로 다시 줄 세운 뒤
        print(f"{question:<34}{answer:<12}{rank_of(before, answer):>10}{rank_of(after, answer):>16}")

    # 한 질문의 상세 결과
    question = "국가인공지능전략위원회 위원장은 누구야?"
    print(f"\n[상세] {question}")
    for rank, doc in enumerate(rerank(question, hybrid_search(question, k=CANDIDATES, candidates=CANDIDATES), top_k=K), 1):
        print(f"  {rank}. score={doc.metadata['rerank_score']:7.3f} | {doc.metadata['citation']} | {doc.page_content[:45]!r}")

    get_vector_store().client.close()
