"""loader가 만든 법률 본문을 검색용 Chunk로 나눈다.

청킹 전략: "조 단위 + 긴 조만 분할"
1. 기본은 조(article) 하나 = Chunk 하나.
2. 조가 MAX_TOKENS보다 길면 항(①②③) 단위로 나눈다.
   항이 없는 조(예: 제2조 정의)는 호(1. 2. 3.) 단위로 나눈다.
3. 나눈 항이 그래도 길면 그 항을 다시 호 단위로 나눈다.
4. 나눈 호가 그래도 길고 아래에 목(가. 나.)이 있으면 "호 본문" Chunk와 "목 목록" Chunk로 나눈다.
   (목 목록이 길면 임베딩이 목록 쪽 의미로 희석되어 정작 호 본문이 검색되지 않는다)
5. 나눈 조각 앞에는 "제N조(제목)"과 상위 문장(예: "다음 각 호의 사항을 심의한다")을 붙여
   조각만 봐도 무슨 내용인지 알 수 있게 한다.

법률 용어 정리
  장(제1장) > 절(제1절) > 조(제1조) > 항(①) > 호(1.) > 목(가.)
"""

import re
from dataclasses import dataclass, field
from statistics import median

import tiktoken
from langchain_core.documents import Document

from rag.loader import load_law_document

# 이 토큰 수를 넘는 조는 항/호 단위로 나눈다.
MAX_TOKENS = 500

# text-embedding-3-small이 사용하는 토크나이저. 임베딩 모델과 같은 기준으로 길이를 잰다.
_encoding = tiktoken.get_encoding("cl100k_base")

# 줄 종류를 구분하는 패턴 (loader가 구조 단위마다 줄을 바꿔 두었기 때문에 줄 맨 앞만 보면 된다)
CHAPTER = re.compile(r"^제\d+장")                               # 제1장 총칙
SECTION = re.compile(r"^제\d+절")                               # 제1절 인공지능산업 기반 조성
ADDENDUM = re.compile(r"^부칙")                                 # 부칙 <제21311호,2026. 1. 20.>
ARTICLE = re.compile(r"^(제\d+조(?:의\d+)?)\((.+?)\)\s*(.*)")   # 제2조(정의) 이 법에서...
PARAGRAPH = re.compile(r"^[①-⑳]")                              # ① ② ③
ITEM = re.compile(r"^(\d+(?:의\d+)?)\.\s")                      # 1. 2. 2의2.
MOK = re.compile(r"^([가-하])\.\s")                             # 가. 나. 다.


@dataclass
class Article:
    """조 하나를 담는 그릇."""

    article: str        # 제2조 (부칙의 조는 "부칙 제1조")
    title: str          # 정의
    chapter: str        # 제1장 총칙
    section: str        # 제1절 ... (절이 없으면 빈 문자열)
    heading: str        # 제2조(정의)
    body: list[str] = field(default_factory=list)  # 제목 뒤의 내용 줄들


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


# ---------------------------------------------------------------------------
# 1단계: 본문을 조 단위로 묶기
# ---------------------------------------------------------------------------
def parse_articles(text: str) -> list[Article]:
    """본문을 위에서부터 한 줄씩 읽으며 조 단위로 묶는다.

    장/절 제목을 만나면 "지금 몇 장 몇 절인지"를 기억해 두었다가 그 아래 조에 붙인다.
    """
    articles: list[Article] = []
    chapter, section = "", ""
    addendum_heading = ""  # 부칙 제목 줄. 비어 있지 않으면 지금 부칙 안에 있다는 뜻이다.
    current: Article | None = None

    for line in text.split("\n"):
        if CHAPTER.match(line):
            chapter, section, current = line, "", None
            continue
        if SECTION.match(line):
            section, current = line, None
            continue
        if ADDENDUM.match(line):
            chapter, section, current = "부칙", "", None
            addendum_heading = line
            continue

        if m := ARTICLE.match(line):
            number, title, rest = m.groups()
            if addendum_heading:
                # 부칙에도 제1조, 제2조가 있을 수 있어 본문 조와 구분한다.
                number = f"부칙 {number}"
            current = Article(number, title, chapter, section, heading=f"{number}({title})")
            # "제3조(...) ① 인공지능기술과..."처럼 제목 뒤에 이어지는 내용은 첫 줄로 넣는다.
            if rest:
                current.body.append(rest)
            articles.append(current)
            continue

        if current is None:
            if not addendum_heading:
                continue  # 장 제목 앞 등 조에 속하지 않는 줄 (정상 문서에는 없다)
            # 최신본 부칙처럼 조 없이 문단만 있으면 부칙 전체를 하나의 조처럼 다룬다.
            current = Article("부칙", "", chapter, section, heading=addendum_heading)
            articles.append(current)

        current.body.append(line)

    return articles


# ---------------------------------------------------------------------------
# 2단계: 긴 조를 항/호 단위로 나누기
# ---------------------------------------------------------------------------
def _group_by(lines: list[str], start: re.Pattern) -> tuple[list[str], list[list[str]]]:
    """start 패턴으로 시작하는 줄마다 새 묶음을 만든다.

    첫 묶음 앞에 있는 줄(예: "이 법에서 사용하는 용어의 뜻은 다음과 같다.")은 lead로 따로 돌려준다.
    """
    lead: list[str] = []
    groups: list[list[str]] = []
    for line in lines:
        if start.match(line):
            groups.append([line])
        elif groups:
            groups[-1].append(line)  # 호 아래의 목(가. 나.)은 그 호에 붙는다.
        else:
            lead.append(line)
    return lead, groups


def _paragraph_no(line: str) -> int:
    """'②' → 2"""
    return ord(line[0]) - ord("①") + 1


def _item_no(line: str) -> str:
    """'2의2. ...' → '2의2'"""
    return ITEM.match(line).group(1)


def _item_pieces(
    prefix: list[str], item: list[str], paragraph: int | None, max_tokens: int
) -> list[dict]:
    """호 하나를 Chunk 재료로 만든다.

    호가 길고 아래에 목(가. 나.)이 있으면 두 개로 나눈다.
      - 호 본문 Chunk: 제목 + 호 첫 줄만        → "고영향 인공지능이란?" 같은 질문용
      - 목 목록 Chunk: 제목 + 호 첫 줄 + 목 전체 → "어떤 영역이 해당돼?" 같은 질문용

    예) 제2조 제4호(고영향 인공지능)는 목 11개가 대부분이라, 나누기 전에는
        "고영향 인공지능이란?" 질문에서 Dense 134등(173개 중)이었다.
    """
    item_no = _item_no(item[0])
    lines = [*prefix, *item]
    mok_lines = [line for line in item[1:] if MOK.match(line)]

    if count_tokens("\n".join(lines)) <= max_tokens or not mok_lines:
        return [{"paragraph": paragraph, "item": item_no, "lines": lines}]

    first_mok = MOK.match(mok_lines[0]).group(1)
    last_mok = MOK.match(mok_lines[-1]).group(1)
    return [
        {"paragraph": paragraph, "item": item_no, "lines": [*prefix, item[0]]},
        {"paragraph": paragraph, "item": item_no, "mok": f"{first_mok}목~{last_mok}목", "lines": lines},
    ]


def split_article(article: Article, max_tokens: int = MAX_TOKENS) -> list[dict]:
    """조 하나를 Chunk 재료 목록으로 만든다.

    반환값의 각 원소: {"paragraph": 항 번호, "item": 호 번호, "mok": 목 범위, "lines": 본문 줄}
    (해당 없는 값은 None이거나 키가 없다)
    """
    whole = [article.heading, *article.body]
    if count_tokens("\n".join(whole)) <= max_tokens:
        return [{"paragraph": None, "item": None, "lines": whole}]

    pieces = []
    lead, paragraphs = _group_by(article.body, PARAGRAPH)

    if not paragraphs:
        # 항이 없는 조(예: 제2조 정의)는 호 단위로 나눈다.
        lead, items = _group_by(article.body, ITEM)
        if not items:
            return [{"paragraph": None, "item": None, "lines": whole}]  # 더 나눌 단위가 없다.
        for item in items:
            pieces += _item_pieces([article.heading, *lead], item, None, max_tokens)
        return pieces

    for paragraph in paragraphs:
        p_no = _paragraph_no(paragraph[0])
        prefix = [article.heading, *lead]

        if count_tokens("\n".join(prefix + paragraph)) <= max_tokens:
            pieces.append({"paragraph": p_no, "item": None, "lines": prefix + paragraph})
            continue

        # 항이 그래도 길면 호 단위로 한 번 더 나눈다. 항의 첫 문장은 모든 조각에 붙인다.
        p_lead, items = _group_by(paragraph, ITEM)
        if not items:
            pieces.append({"paragraph": p_no, "item": None, "lines": prefix + paragraph})
            continue
        for item in items:
            pieces += _item_pieces([*prefix, *p_lead], item, p_no, max_tokens)

    return pieces


def make_citation(
    article: str, paragraph: int | None, item: str | None, mok: str | None = None
) -> str:
    """출처 표시용 문자열. 예) 제2조 제4호, 제7조 제2항, 제8조 제1항 제2호의2, 제2조 제4호 가목~카목"""
    parts = [article]
    if paragraph:
        parts.append(f"제{paragraph}항")
    if item:
        base, _, branch = item.partition("의")
        parts.append(f"제{base}호" + (f"의{branch}" if branch else ""))
    if mok:
        parts.append(mok)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# 3단계: LangChain Document로 만들기
# ---------------------------------------------------------------------------
def chunk_law_document(doc: Document, max_tokens: int = MAX_TOKENS) -> list[Document]:
    """loader의 Document 하나를 검색용 Chunk Document 리스트로 만든다."""
    chunks = []
    for article in parse_articles(doc.page_content):
        for piece in split_article(article, max_tokens):
            content = "\n".join(piece["lines"])
            metadata = {
                **doc.metadata,  # source, law_name, law_short_name, enforcement_date, law_number
                "chunk_id": len(chunks),
                "chapter": article.chapter,
                "section": article.section,
                "article": article.article,
                "article_title": article.title,
                "paragraph": piece["paragraph"],
                "item": piece["item"],
                "mok": piece.get("mok"),
                "citation": make_citation(
                    article.article, piece["paragraph"], piece["item"], piece.get("mok")
                ),
                "num_tokens": count_tokens(content),
            }
            chunks.append(Document(page_content=content, metadata=metadata))
    return chunks


# ---------------------------------------------------------------------------
# 점검: 청킹하면서 내용이 빠지지 않았는지 확인
# ---------------------------------------------------------------------------
def check_chunks(doc: Document, chunks: list[Document]) -> list[str]:
    """문제를 찾아 메시지 목록으로 돌려준다. 빈 리스트면 이상 없음."""
    problems = []
    lines = doc.page_content.split("\n")

    # 1) 원문의 조가 전부 Chunk에 들어갔는가
    source_articles = {m.group(1) for line in lines if (m := ARTICLE.match(line))}
    chunk_articles = {c.metadata["article"].removeprefix("부칙 ") for c in chunks}
    if missing := source_articles - chunk_articles:
        problems.append(f"Chunk에 없는 조: {sorted(missing)}")

    # 2) 장/절/부칙 제목을 뺀 원문의 모든 줄이 어느 Chunk엔가 들어갔는가
    all_text = "\n".join(c.page_content for c in chunks)
    for line in lines:
        if CHAPTER.match(line) or SECTION.match(line) or ADDENDUM.match(line):
            continue  # 제목 줄은 Chunk 본문 대신 metadata(chapter, section)에 들어간다.
        body = m.group(3) if (m := ARTICLE.match(line)) else line
        if body and body not in all_text:
            problems.append(f"Chunk에 없는 줄: {line[:40]}...")

    # 3) 빈 Chunk가 없는가
    problems += [f"빈 Chunk: {c.metadata['citation']}" for c in chunks if not c.page_content.strip()]
    return problems


if __name__ == "__main__":
    # uv run python -m rag.chunker
    doc = load_law_document()
    chunks = chunk_law_document(doc)

    tokens = [c.metadata["num_tokens"] for c in chunks]
    split_articles = sorted(
        {c.metadata["article"] for c in chunks if c.metadata["paragraph"] or c.metadata["item"]},
        key=lambda a: [int(n) for n in re.findall(r"\d+", a)],
    )
    print(f"Chunk 수: {len(chunks)}")
    print(f"토큰 수: 최소 {min(tokens)} / 중앙값 {median(tokens)} / 최대 {max(tokens)}")
    print(f"MAX_TOKENS({MAX_TOKENS}) 초과: {[c.metadata['citation'] for c in chunks if c.metadata['num_tokens'] > MAX_TOKENS]}")
    print(f"나뉜 조 ({len(split_articles)}개): {split_articles}")

    problems = check_chunks(doc, chunks)
    print("점검 결과:", "이상 없음" if not problems else problems)

    # 예시: "고영향 인공지능" 정의가 담긴 Chunk
    for c in chunks:
        if c.metadata["citation"] == "제2조 제4호":
            print("-" * 80)
            print(c.metadata)
            print(c.page_content)
