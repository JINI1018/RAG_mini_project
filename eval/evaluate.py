"""Golden Set으로 RAG의 검색과 답변을 평가한다.

수업 09_rag_evaluation 구조를 RFP 지표까지 넓혔다.
  Retrieval: Hit@K (수업) + Recall@K + MRR
  Answer:    Faithfulness (수업, LLM-as-a-Judge 1~5점) + 정답성 + 관련성 + Hallucination 여부

실행
  uv run python -m eval.evaluate                    # 검색 + 답변 평가 (약 4~5분, API 약 60회)
  uv run python -m eval.evaluate --retrieval-only   # 검색 평가만 (약 1분, API 1회)
  uv run python -m eval.evaluate --limit 5          # 앞의 5문항만 빠르게 확인

결과는 eval/results/에 JSON으로 저장된다. (개선 전후 비교용)
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from common.ai_model import get_llm_model
from rag.pipeline import build_context, generate_answer, retrieve_context
from rag.retriever import citation_matches, embed_queries, get_vector_store, retrieve

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_SET_PATH = EVAL_DIR / "golden_set.jsonl"
RESULTS_DIR = EVAL_DIR / "results"

# 비교할 검색 방식 (이름: 검색 함수)
RETRIEVAL_CONFIGS = {
    "dense": lambda q, k: retrieve(q, k=k, method="dense"),
    "bm25": lambda q, k: retrieve(q, k=k, method="bm25"),
    "hybrid": lambda q, k: retrieve(q, k=k, method="hybrid"),
    "hybrid+rerank": lambda q, k: retrieve_context(q, k=k, method="hybrid", use_rerank=True),
}


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# API 호출 간격 조절 (MonoRouter: 분당 30회 제한)
# ---------------------------------------------------------------------------
MIN_INTERVAL = 2.5  # LLM 호출 사이 최소 간격(초). 2.5초면 분당 최대 24회라 제한에 걸리지 않는다.
_last_call = 0.0


def throttle() -> None:
    """직전 호출 후 MIN_INTERVAL초가 지나지 않았으면 남은 시간만큼 기다린다."""
    global _last_call
    wait = _last_call + MIN_INTERVAL - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()


# ---------------------------------------------------------------------------
# 1. Retrieval 평가: Hit@K, Recall@K, MRR
# ---------------------------------------------------------------------------
def retrieval_metrics(docs, relevant: list[str]) -> dict:
    """검색 결과 하나를 채점한다.

    Hit@K    : 정답 조문이 하나라도 Top-K 안에 있으면 1, 없으면 0 (수업 09번)
    Recall@K : 정답 조문이 여러 개일 때, 그중 몇 개를 찾았는지 비율
    MRR      : 처음 나온 정답의 순위 역수 (1등=1.0, 2등=0.5, 3등=0.33, 못 찾으면 0)
               → 정답을 "찾았는지"뿐 아니라 "얼마나 위에" 찾았는지까지 본다.
    """
    citations = [doc.metadata["citation"] for doc in docs]
    first_ranks = []  # 정답 조문별로 처음 나온 순위
    for answer in relevant:
        ranks = [i for i, c in enumerate(citations, start=1) if citation_matches(c, answer)]
        if ranks:
            first_ranks.append(ranks[0])

    return {
        "hit": 1 if first_ranks else 0,
        "recall": len(first_ranks) / len(relevant),
        "rr": 1 / min(first_ranks) if first_ranks else 0.0,
        "retrieved": citations,
    }


def evaluate_retrieval(items: list[dict], k: int) -> dict:
    """검색 방식별로 Golden Set 전체를 채점한다. (정답 조문이 없는 문항은 제외)"""
    targets = [item for item in items if item["relevant"]]

    # 질문 전체를 API 한 번에 임베딩해 둔다. (이후 모든 검색 방식이 재사용)
    embed_queries([item["question"] for item in targets])

    results = {}
    for name, search in RETRIEVAL_CONFIGS.items():
        per_question = {}
        for item in targets:
            docs = search(item["question"], k)
            per_question[item["id"]] = retrieval_metrics(docs, item["relevant"])
        scores = per_question.values()
        results[name] = {
            "hit@k": mean(s["hit"] for s in scores),
            "recall@k": mean(s["recall"] for s in scores),
            "mrr": mean(s["rr"] for s in scores),
            "per_question": per_question,
        }
    return results


# ---------------------------------------------------------------------------
# 2. Answer 평가: LLM-as-a-Judge
# ---------------------------------------------------------------------------
class Judgement(BaseModel):
    correctness: int = Field(description="정답성 1~5: 참고 답변의 핵심 내용과 일치하는가")
    relevance: int = Field(description="관련성 1~5: 질문에 직접 답하고 있는가")
    faithfulness: int = Field(description="근거 충실성 1~5: 답변 내용이 Context에서 뒷받침되는가")
    hallucination: bool = Field(description="Context에 없는 사실(숫자, 기간, 기관, 의무 등)을 지어냈으면 true")
    reason: str = Field(description="채점 이유 한두 문장")


JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """당신은 인공지능기본법 QA 시스템의 답변을 채점하는 평가자다.

채점 기준:
- correctness (1~5): 참고 답변의 핵심 내용과 비교한다.
  5 = 핵심 내용이 모두 정확하다. 3 = 일부 누락되거나 부정확하다. 1 = 틀렸거나 답하지 못했다.
- relevance (1~5): 질문에 직접 답하는가.
  5 = 질문에 정확히 답한다. 3 = 관련은 있지만 초점이 흐리다. 1 = 질문과 무관하다.
- faithfulness (1~5): 답변 내용이 Context에서 뒷받침되는가.
  5 = 주요 내용이 Context에서 직접 뒷받침된다. 3 = 일부는 근거가 불명확하다. 1 = 대부분 근거가 없다.
- hallucination: Context에 없는 사실(숫자, 기간, 기관명, 의무 등)을 지어냈으면 true.
  쉬운 말로 풀어 쓴 것은 지어낸 것이 아니다.""",
    ),
    (
        "human",
        "질문:\n{question}\n\n참고 답변:\n{reference}\n\nContext:\n{context}\n\n평가할 답변:\n{answer}",
    ),
])


def evaluate_answers(items: list[dict], k: int) -> dict:
    """pipeline으로 답변을 만들고 LLM Judge로 채점한다."""
    judge = JUDGE_PROMPT | get_llm_model(temperature=0, max_tokens=500).with_structured_output(
        Judgement, method="function_calling"
    )

    per_question = {}
    for n, item in enumerate(items, start=1):
        print(f"  [{n}/{len(items)}] {item['id']} {item['question']}")
        docs = retrieve_context(item["question"], k=k)
        throttle()
        response = generate_answer(item["question"], docs)
        record = {
            "answer": response["answer"],
            "sources": [s["article"] for s in response["sources"]],
            "is_answerable": response["is_answerable"],
        }

        if not item["answerable"]:
            # 답변 불가 문항: LLM 채점 없이 "거절했는지"만 본다.
            record["refusal_ok"] = not response["is_answerable"]
        else:
            throttle()
            judgement: Judgement = judge.invoke({
                "question": item["question"],
                "reference": item["reference_answer"],
                "context": build_context(docs),
                "answer": response["answer"],
            })
            record.update(judgement.model_dump())

        per_question[item["id"]] = record

    judged = [r for r in per_question.values() if "correctness" in r]
    refusals = [r for r in per_question.values() if "refusal_ok" in r]
    return {
        "correctness": mean(r["correctness"] for r in judged) if judged else None,
        "relevance": mean(r["relevance"] for r in judged) if judged else None,
        "faithfulness": mean(r["faithfulness"] for r in judged) if judged else None,
        "hallucination_rate": mean(r["hallucination"] for r in judged) if judged else None,
        "refusal_accuracy": mean(r["refusal_ok"] for r in refusals) if refusals else None,
        "per_question": per_question,
    }


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def print_retrieval(results: dict, items: list[dict], k: int) -> None:
    print(f"\n[Retrieval] 정답 조문이 있는 {sum(1 for i in items if i['relevant'])}문항, K={k}")
    print(f"{'방식':<16}{'Hit@K':>8}{'Recall@K':>10}{'MRR':>8}")
    for name, r in results.items():
        print(f"{name:<16}{r['hit@k']:>8.3f}{r['recall@k']:>10.3f}{r['mrr']:>8.3f}")

    # 가장 좋은 방식에서도 놓친 문항 (Golden Set 검토나 개선 포인트)
    best = max(results, key=lambda name: results[name]["mrr"])
    missed = [qid for qid, s in results[best]["per_question"].items() if not s["hit"]]
    print(f"\n{best}에서 Top-{k} 안에 정답이 없는 문항: {missed or '없음'}")


def print_answers(results: dict) -> None:
    def fmt(value):
        return "-" if value is None else f"{value:.2f}"

    print("\n[Answer] (hybrid+rerank 검색 결과로 답변 생성)")
    print(f"  정답성        (1~5): {fmt(results['correctness'])}")
    print(f"  관련성        (1~5): {fmt(results['relevance'])}")
    print(f"  근거 충실성    (1~5): {fmt(results['faithfulness'])}")
    print(f"  Hallucination 비율  : {fmt(results['hallucination_rate'])}  (낮을수록 좋다)")
    print(f"  답변불가 거절 정확도: {fmt(results['refusal_accuracy'])}")

    low = [
        (qid, r) for qid, r in results["per_question"].items()
        if r.get("correctness", 5) <= 3 or r.get("hallucination") or r.get("refusal_ok") is False
    ]
    if low:
        print("\n  확인이 필요한 문항:")
        for qid, r in low:
            detail = r.get("reason") or ("거절해야 하는데 답변함" if r.get("refusal_ok") is False else "")
            print(f"   - {qid}: 정답성={r.get('correctness', '-')} hallucination={r.get('hallucination', '-')} | {detail}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Golden Set으로 RAG를 평가한다.")
    parser.add_argument("--k", type=int, default=5, help="Top-K (기본 5)")
    parser.add_argument("--retrieval-only", action="store_true", help="검색 평가만 (LLM 호출 없음)")
    parser.add_argument("--limit", type=int, help="앞에서부터 N문항만 평가")
    args = parser.parse_args()

    items = load_golden_set()[: args.limit]
    print(f"Golden Set {len(items)}문항")

    report = {"created_at": datetime.now().isoformat(timespec="seconds"), "k": args.k, "num_items": len(items)}

    report["retrieval"] = evaluate_retrieval(items, args.k)
    print_retrieval(report["retrieval"], items, args.k)

    if not args.retrieval_only:
        print("\n답변 생성 및 채점 중... (API 제한 때문에 문항당 약 5~8초)")
        report["answer"] = evaluate_answers(items, args.k)
        print_answers(report["answer"])

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out.relative_to(EVAL_DIR.parent)}")

    get_vector_store().client.close()
