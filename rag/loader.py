"""AI 기본법 PDF를 읽어 깨끗한 텍스트(LangChain Document)로 만든다.

처리 순서
1. Docling으로 PDF를 읽는다.
2. Docling이 인식한 텍스트 조각(item)을 원본 그대로(orig) 꺼낸다.
3. 페이지가 바뀌면서 잘린 줄을 앞 줄에 다시 이어 붙인다.
4. 문서 머리말에서 법령명, 법률번호, 시행일을 뽑아 metadata로 만든다.

청킹(조/항 단위로 나누기)은 여기서 하지 않는다. chunker.py가 담당한다.
"""

import re
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_core.documents import Document

# 프로젝트 루트 기준 데이터 경로 (어느 폴더에서 실행해도 같은 파일을 찾도록 절대경로로 만든다)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 법령이 개정되면 새 PDF를 data/ai_basic_law/에 넣고 아래 파일명만 바꾼다.
DEFAULT_PDF_PATH = PROJECT_ROOT / "data" / "ai_basic_law" / "ai_basic_law_20260721.pdf"

# 법률 문서에서 "새 단위가 시작되는 줄"의 패턴
# 이 패턴으로 시작하지 않는 줄은 페이지가 넘어가며 잘린 앞 줄의 나머지로 본다.
STRUCTURE_LINE = re.compile(
    r"^("
    r"제\d+장"                      # 장: 제1장 총칙
    r"|제\d+절"                     # 절: 제1절 인공지능산업 기반 조성
    r"|제\d+조(의\d+)?\("           # 조: 제2조(정의), 제22조의2(...)
    r"|[①-⑳]"                      # 항: ① ② ③
    r"|\d+(의\d+)?\.\s"             # 호: 1. 2. 3. / 가지번호 2의2. 5의3.
    r"|[가-하]\.\s"                 # 목: 가. 나. 다.
    r"|부칙"                        # 부칙
    r"|\[시행"                      # [시행일: ...] 안내문
    r"|※"                          # ※ 안내문
    r")"
)

# 머리말(법령 기본 정보) 줄의 패턴
# Docling이 읽는 순서를 잘못 잡아 머리말을 본문 중간에 끼워 넣는 경우가 있다.
# (2026.7.21. 시행본은 머리말이 제2조 중간에 들어갔다) 그래서 위치가 아니라 모양으로 골라낸다.
HEADER_LINE = re.compile(
    r"^("
    r".+\(\s*약칭:"                 # 법령명 ( 약칭: 인공지능기본법 )
    r"|\[시행 \d"                   # [시행 2026. 7. 21.] [법률 제21311호, ...]  ※ 본문의 "[시행일: ...]"과 구분
    r"|※ 본 법령은"                 # 시행 전 개정사항 안내문
    r"|.+\d{2,3}-\d{3,4}-\d{4}$"    # 소관부처 (담당과) 전화번호
    r")"
)

# 한 줄 안에 붙어 있는 다음 항/가지번호 호의 시작 위치.
# Docling이 두 단위를 한 덩어리로 읽는 경우가 있어서 이 위치에서 줄을 나눈다.
#   예) "...구성할 수 없다.<개정 2026. 1. 20.> ③ 위원회의 위원장은..."
#   예) "...에 관한 사항 6의2. 인공지능기술의..."
# ("제6조의2"처럼 조 번호에는 "조"가 끼어 있어 걸리지 않는다)
INLINE_MARKER = re.compile(r"\s+(?=([①-⑳]|\d+의\d+\.)\s)")

# 제목 줄(장/절/부칙). 제목 뒤에 오는 줄은 잘린 줄이 아니라 새 내용이므로 이어 붙이지 않는다.
HEADING_LINE = re.compile(r"^(제\d+장|제\d+절|부칙)")


def _build_converter() -> DocumentConverter:
    """PDF 변환기를 만든다."""
    pipeline_options = PdfPipelineOptions()

    # 법제처 PDF는 글자를 그대로 담고 있는 텍스트 PDF라 OCR(이미지에서 글자 인식)이 필요 없다.
    pipeline_options.do_ocr = False

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def _extract_lines(pdf_path: Path) -> list[str]:
    """Docling으로 PDF를 읽어 텍스트 조각을 줄 단위 리스트로 꺼낸다."""
    converter = _build_converter()
    docling_doc = converter.convert(str(pdf_path)).document

    lines = []
    for item, _level in docling_doc.iterate_items():
        # 그림(picture) 같은 요소는 text가 없으므로 건너뛴다.
        text = getattr(item, "text", None)
        if not text:
            continue

        # Docling은 "1. ..." 같은 목록 번호를 떼어내고 text에 본문만 남긴다.
        # 법률에서는 그 번호가 "제○호"라는 중요한 정보이므로 원본(orig)을 쓴다.
        original = getattr(item, "orig", None) or text
        lines.append(original.strip())

    return lines


def _split_header(lines: list[str]) -> tuple[list[str], list[str]]:
    """전체 줄을 머리말 줄과 본문 줄로 나눈다."""
    header = [line for line in lines if HEADER_LINE.match(line)]
    body = [line for line in lines if not HEADER_LINE.match(line)]
    return header, body


def _split_inline_markers(lines: list[str]) -> list[str]:
    """한 줄에 붙어 있는 항/가지번호 호를 별도 줄로 나눈다."""
    result = []
    for line in lines:
        # split 패턴에 괄호 그룹이 있으면 그 내용도 결과에 끼어들어서, finditer로 위치만 찾아 자른다.
        start = 0
        for m in INLINE_MARKER.finditer(line):
            result.append(line[start:m.start()])
            start = m.end()
        result.append(line[start:])
    return result


def _merge_broken_lines(lines: list[str]) -> list[str]:
    """페이지 경계에서 잘린 줄을 앞 줄에 이어 붙인다.

    예) "...공공기관으로 하여금 인공지능산업" + "의 진흥에 관한 사업 등에..."
        → "...공공기관으로 하여금 인공지능산업의 진흥에 관한 사업 등에..."
    """
    merged: list[str] = []
    for line in lines:
        is_broken = not STRUCTURE_LINE.match(line)
        after_heading = bool(merged) and HEADING_LINE.match(merged[-1])

        if merged and is_broken and not after_heading:
            # 잘린 지점은 대부분 단어 중간이라 공백 없이 붙인다.
            merged[-1] += line
        else:
            merged.append(line)
    return merged


def _parse_law_info(header_lines: list[str]) -> dict:
    """머리말 줄에서 법령 기본 정보를 뽑는다.

    머리말 예시
      인공지능 발전과 신뢰 기반 조성 등에 관한 기본법 ( 약칭: 인공지능기본법 )
      [시행 2026. 7. 21.] [법률 제21311호, 2026. 1. 20., 일부개정]
    """
    info = {}
    header = "\n".join(header_lines)

    name = re.search(r"^(.+?기본법)\s*\(\s*약칭:\s*(.+?)\s*\)", header, re.MULTILINE)
    if name:
        info["law_name"] = name.group(1).strip()
        info["law_short_name"] = name.group(2).strip()

    enforce = re.search(r"\[시행\s*([\d.\s]+?)\]", header)
    if enforce:
        info["enforcement_date"] = enforce.group(1).strip()

    law_no = re.search(r"법률\s*(제\d+호)", header)
    if law_no:
        info["law_number"] = law_no.group(1)

    return info


def load_law_document(pdf_path: str | Path = DEFAULT_PDF_PATH) -> Document:
    """AI 기본법 PDF를 읽어 LangChain Document 하나로 반환한다.

    - page_content: 법률 구조 단위(장/절/조/항/호/목)로 줄바꿈된 본문 (머리말 제외)
    - metadata: source, law_name, law_short_name, enforcement_date, law_number
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

    header, body = _split_header(_extract_lines(pdf_path))
    lines = _merge_broken_lines(_split_inline_markers(body))

    metadata = {"source": pdf_path.name, **_parse_law_info(header)}
    return Document(page_content="\n".join(lines), metadata=metadata)


if __name__ == "__main__":
    # uv run python -m rag.loader 로 실행하면 결과를 확인할 수 있다.
    doc = load_law_document()
    print(doc.metadata)
    print(f"전체 글자 수: {len(doc.page_content)}")
    print("-" * 80)
    print(doc.page_content[:1500])
