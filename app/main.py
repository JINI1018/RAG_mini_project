"""AI 기본법 QA API 서버

실행
  uv run uvicorn app.main:app --reload

확인
  브라우저에서 http://localhost:8000/docs  (Swagger UI에서 직접 질문을 보내 볼 수 있다)
  curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \\
       -d '{"question": "고영향 인공지능이란 무엇인가요?"}'
"""

from contextlib import asynccontextmanager

import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.pipeline import ask
from rag.reranker import get_reranker
from rag.retriever import get_bm25, get_vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버가 켜질 때 한 번 실행된다.

    Qdrant 연결, BM25 준비, Reranker 모델 로딩(약 10초)을 미리 해 두어
    첫 번째 질문이 느려지지 않게 한다. (lru_cache 덕분에 이후에는 다시 만들지 않는다)
    """
    get_vector_store()
    get_bm25()
    get_reranker()
    yield
    # 서버가 꺼질 때: Qdrant 연결을 닫는다.
    get_vector_store().client.close()


app = FastAPI(
    title="AI 기본법 QA API",
    description="인공지능기본법 조문을 근거로 질문에 답하는 RAG 서비스",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# 요청/응답 형식 (Pydantic 모델로 정의하면 FastAPI가 입력 검증과 /docs 문서를 자동으로 만들어 준다)
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(min_length=1, examples=["고영향 인공지능이란 무엇인가요?"])


class Source(BaseModel):
    article: str = Field(description="근거 조문", examples=["제2조 제4호"])
    title: str = Field(description="조문 제목", examples=["정의"])
    content: str = Field(description="검색된 법률 내용")


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    is_answerable: bool = Field(description="법 조문으로 답할 수 있는 질문이었는지")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "RAG API"
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """질문을 받아 AI 기본법 조문을 근거로 답변한다.

    async def가 아니라 def로 만든 이유: 검색과 LLM 호출이 오래 걸리는 동기 코드라서,
    def로 두면 FastAPI가 별도 스레드에서 실행해 다른 요청을 막지 않는다.
    """
    try:
        return ask(request.question)
    except openai.RateLimitError:
        # MonoRouter 호출 제한(분당 30회)에 걸린 경우
        raise HTTPException(status_code=429, detail="요청이 많아 잠시 후(1분) 다시 시도해주세요.")
