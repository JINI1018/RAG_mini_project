"""chunker가 만든 Chunk를 임베딩해서 Qdrant에 저장한다. (Offline 단계의 마지막)

수업 코드(02_document_processing의 Chroma.from_documents)와 같은 방식이며,
Chroma 대신 LangChain의 QdrantVectorStore를 사용한다.

- build_vector_store(): Chunk를 임베딩해서 새로 저장한다.
- load_vector_store():  이미 저장된 컬렉션을 불러온다. (retriever에서 사용, 다시 임베딩하지 않음)

실행
  uv run python -m rag.vectorstore
"""

import uuid

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from common.ai_model import get_embedding_model
from common.config import QDRANT_COLLECTION, QDRANT_URL
from common.qdrant import get_qdrant_client
from rag.chunker import chunk_law_document
from rag.loader import load_law_document


def build_vector_store(chunks: list[Document], reset: bool = True) -> QdrantVectorStore:
    """Chunk를 임베딩해서 Qdrant에 저장한다.

    reset=True면 기존 컬렉션을 지우고 새로 만든다. (청킹을 바꾼 뒤에는 True로 다시 실행)
    """
    # Chunk마다 고정 id를 만든다. 같은 Chunk는 항상 같은 id라서 다시 넣어도 중복 저장되지 않는다.
    # (Qdrant의 id는 정수나 UUID만 가능하다)
    ids = [
        str(uuid.uuid5(uuid.NAMESPACE_URL, f"{c.metadata['source']}|{c.metadata['citation']}"))
        for c in chunks
    ]

    return QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        ids=ids,
        url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION,
        force_recreate=reset,  # 수업 코드의 RESET_DB와 같은 역할
        # distance는 기본값이 COSINE이다.
    )


def load_vector_store() -> QdrantVectorStore:
    """저장된 컬렉션을 불러온다. (임베딩을 새로 만들지 않는다)"""
    client = get_qdrant_client()
    exists = client.collection_exists(QDRANT_COLLECTION)
    client.close()

    # 컬렉션이 없으면 먼저 저장하라고 안내하고 중단
    if not exists:
        raise RuntimeError(
            f"'{QDRANT_COLLECTION}' 컬렉션이 없습니다. "
            "먼저 uv run python -m rag.vectorstore 를 실행하세요."
        )

    # 저장할 때와 같은 임베딩 모델을 써야 질문 벡터와 비교할 수 있다.
    return QdrantVectorStore.from_existing_collection(
        embedding=get_embedding_model(),
        url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION,
        # 기본값(True)이면 벡터 차원을 확인하려고 임베딩 API를 1회 호출한다.
        # 저장과 검색이 같은 EMBEDDING_MODEL 설정을 쓰므로 끄고 API 호출을 아낀다. (분당 30회 제한)
        validate_collection_config=False,
    )


if __name__ == "__main__":
    doc = load_law_document()
    chunks = chunk_law_document(doc)
    print(f"Chunk {len(chunks)}개 준비 ({doc.metadata['source']})")

    vector_store = build_vector_store(chunks, reset=True)
    print(f"저장된 Chunk 수: {vector_store.client.count(QDRANT_COLLECTION).count}")

    # 확인용 검색 (Qdrant의 score는 코사인 유사도: 높을수록 비슷하다. Chroma의 distance와 반대)
    query = "고영향 인공지능이란 무엇인가요?"
    print(f"\n[확인용 검색] {query}")
    for chunk, score in vector_store.similarity_search_with_score(query, k=3):
        print(f"  score={score:.4f} | {chunk.metadata['citation']} | {chunk.page_content[:50]!r}")

    vector_store.client.close()  # 연결을 닫는다. (안 닫으면 종료할 때 경고가 뜬다)
