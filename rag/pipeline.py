"""질문 → 검색 → Rerank → 근거 기반 답변까지 RAG 전체 흐름을 연결한다.

수업 노트북과의 관계
  07_context_engineering: 검색된 문서를 Context로 만든다. (근거 추출 단계는 생략, 아래 build_context 참고)
  08_grounded_answer:     Context만 사용하고 [S1]처럼 출처를 표시하며, 근거가 없으면 답하지 않는다.
  템플릿 qdrant_test.ipynb: RagAnswer(Pydantic)로 답변을 구조화해서 받는다.

흐름
  question
    → retrieve (hybrid, 후보 20개)
    → rerank (상위 5개)
    → build_context ([S1] 제7조 제3항 ... 형태)
    → LLM (RagAnswer: answer / source_ids / is_answerable)
    → {"answer": ..., "sources": [{"article": ..., "content": ...}]}

실행
  uv run python -m rag.pipeline
  uv run python -m rag.pipeline "고영향 인공지능이란 무엇인가요?"
"""

import sys

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from common.ai_model import get_llm_model
from rag.reranker import rerank
from rag.retriever import get_bm25, retrieve

NO_ANSWER = "제공된 법률 조문만으로는 답변하기 어렵습니다."


# ---------------------------------------------------------------------------
# 1. 검색: 후보를 넉넉히 가져와 Rerank로 상위 k개를 고른다.
# ---------------------------------------------------------------------------
def retrieve_context(
    question: str,
    k: int = 5,
    method: str = "hybrid",
    use_rerank: bool = True,
    candidates: int = 20,
) -> list[Document]:
    """질문에 답할 근거 Chunk k개를 찾는다."""
    if use_rerank:
        docs = retrieve(question, k=candidates, method=method)
        docs = rerank(question, docs, top_k=k)
    else:
        docs = retrieve(question, k=k, method=method)
    return expand_to_full_item(docs)


def expand_to_full_item(docs: list[Document]) -> list[Document]:
    """Small-to-Big: 검색은 작은 Chunk로, 답변에는 큰 Chunk를 넘긴다.

    chunker는 목 목록이 긴 호를 "호 본문" Chunk와 "호 본문 + 목 목록" Chunk로 나눴다.
    (예: 제2조 제4호 / 제2조 제4호 가목~카목)
    검색에는 짧은 "호 본문"이 잘 걸리지만, 답변에는 목 목록(어떤 영역이 해당하는지)도 필요하다.
    그래서 "호 본문" Chunk가 뽑히면 같은 호의 "목 목록" Chunk로 바꿔서 넘긴다.
    """
    _, all_chunks = get_bm25()  # Qdrant에 저장된 전체 Chunk (retriever가 이미 불러 둔 것)
    full_items = {
        doc.metadata["citation"].rsplit(" ", 1)[0]: doc  # "제2조 제4호 가목~카목" → "제2조 제4호"
        for doc in all_chunks
        if doc.metadata.get("mok")
    }

    expanded = []
    for doc in docs:
        full = full_items.get(doc.metadata["citation"])
        if full is None:
            expanded.append(doc)
        else:
            # 검색 점수 등은 원래 Chunk의 것을 유지하고 본문과 출처만 큰 Chunk로 바꾼다.
            expanded.append(Document(
                page_content=full.page_content,
                metadata={**doc.metadata, "citation": full.metadata["citation"], "mok": full.metadata["mok"]},
            ))
    return expanded


# ---------------------------------------------------------------------------
# 2. Context 만들기
# ---------------------------------------------------------------------------
def build_context(docs: list[Document]) -> str:
    """검색된 Chunk에 [S1], [S2] 번호를 붙여 하나의 Context 문자열로 합친다.

    수업 07번은 여기서 LLM으로 "필요한 근거만 추출"하는 단계를 한 번 더 거쳤다.
    우리 Chunk는 이미 조/항 단위로 작고(중앙값 약 130토큰), LLM 호출을 늘리면
    API 제한(분당 30회)과 응답 시간에 부담이 되어 이 단계는 생략했다.
    """
    blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        blocks.append(f"[S{i}] {meta['citation']} ({meta['article_title']})\n{doc.page_content}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# 3. 근거 기반 답변 (Grounded Answer)
# ---------------------------------------------------------------------------
class RagAnswer(BaseModel):
    """LLM이 채워서 돌려줄 답변 형식 (자유 텍스트 대신 필드로 받으면 출처와 평가를 다루기 쉽다)"""

    answer: str = Field(description="사용자 질문에 대한 답변. 근거 문장 뒤에 [S1]처럼 출처 번호를 표시한다.")
    source_ids: list[str] = Field(description="답변에 실제로 사용한 출처 번호 목록 (예: ['S1', 'S3'])")
    is_answerable: bool = Field(description="주어진 Context만으로 답변할 수 있었는지 여부")


PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """당신은 인공지능기본법을 잘 모르는 사람에게 법 조문을 쉽게 설명하는 법률 QA 어시스턴트다.
아래 Context에 있는 법률 조문만 근거로 답변한다.

규칙:
- Context에 있는 내용만 사용하고, 없는 내용은 추측하거나 지어내지 않는다.
- 사용한 근거 문장 뒤에 [S1], [S2]처럼 출처 번호를 표시한다.
- 법률 용어는 쉬운 말로 풀어서 설명하되, 조문의 의미를 바꾸지 않는다.
- 질문한 세부 사항(기준, 금액, 절차 등)을 조문이 "대통령령으로 정한다", "고시한다"처럼 다른 규정에 맡겨 둔 경우,
  그 자체가 답이다. is_answerable을 true로 하고, 법에는 구체적인 내용이 없으며 어느 규정에 위임되어 있는지를
  출처와 함께 설명한다. 이때 구체적인 수치나 내용을 추측해서 덧붙이지 않는다.
- Context에 질문과 관련된 조문이 아예 없을 때만 is_answerable을 false로 하고, answer에 "{no_answer}"라고 쓴다.""",
    ),
    ("human", "질문:\n{question}\n\nContext:\n{context}"),
])


def generate_answer(question: str, docs: list[Document]) -> dict:
    """검색된 Chunk를 근거로 답변을 만들고 API 응답 형식으로 돌려준다."""
    if not docs:
        return {"answer": NO_ANSWER, "sources": [], "is_answerable": False}

    # MonoRouter에서는 function_calling 방식에서만 구조화 출력이 정상 동작한다. (템플릿 노트북 참고)
    llm = get_llm_model(temperature=0, max_tokens=1000)
    chain = PROMPT | llm.with_structured_output(RagAnswer, method="function_calling")
    result: RagAnswer = chain.invoke({
        "question": question,
        "context": build_context(docs),
        "no_answer": NO_ANSWER,
    })

    if not result.is_answerable:
        return {"answer": NO_ANSWER, "sources": [], "is_answerable": False}

    # LLM이 고른 출처 번호(S1, S3)를 실제 Chunk로 바꾼다. 번호를 안 줬으면 검색된 Chunk 전체를 출처로 쓴다.
    by_id = {f"S{i}": doc for i, doc in enumerate(docs, start=1)}
    used = [by_id[sid] for sid in dict.fromkeys(result.source_ids) if sid in by_id] or docs

    return {
        "answer": result.answer,
        "sources": [
            {
                "article": doc.metadata["citation"],  # 예) 제2조 제4호
                "title": doc.metadata["article_title"],  # 예) 정의
                "content": doc.page_content,
            }
            for doc in used
        ],
        "is_answerable": True,
    }


# ---------------------------------------------------------------------------
# 4. 전체 흐름 (app/main.py의 POST /ask에서 부른다)
# ---------------------------------------------------------------------------
def ask(question: str, k: int = 5, method: str = "hybrid", use_rerank: bool = True) -> dict:
    """질문 하나에 대해 검색 → Rerank → 근거 기반 답변까지 실행한다."""
    docs = retrieve_context(question, k=k, method=method, use_rerank=use_rerank)
    return generate_answer(question, docs)


if __name__ == "__main__":
    questions = sys.argv[1:] or [
        "고영향 인공지능이란 무엇인가요?",
        "생성형 AI로 만든 결과물에는 표시를 해야 하나요?",
        "GPT-5의 벤치마크 점수는 얼마인가요?",  # 법에 없는 내용 → 답변 거절해야 한다.
    ]
    for question in questions:
        response = ask(question)
        print("=" * 80)
        print(f"[질문] {question}")
        print(f"\n[답변] {response['answer']}")
        print("\n[근거]")
        for source in response["sources"]:
            print(f"  - {source['article']} ({source['title']})")
    print("=" * 80)
