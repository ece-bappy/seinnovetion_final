import json
import logging
import os
import pickle
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ai_grader")

ROOT_DIR = Path(__file__).resolve().parent
BOOK_DIR = ROOT_DIR / "book"
ASSIGNMENTS_DIR = ROOT_DIR / "assignments"
RUBRIC_PATH = ROOT_DIR / "rubric.md"
TASK_PATH = ROOT_DIR / "task.md"
CACHE_DIR = ROOT_DIR / ".cache"
INDEX_CACHE_PATH = CACHE_DIR / "book_index.pkl"

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


class SimpleVectorIndex:
    def __init__(self, chunks: Optional[List[str]] = None, embeddings: Optional[np.ndarray] = None) -> None:
        self.chunks = chunks or []
        self.embeddings = embeddings if embeddings is not None else np.empty((0, 0), dtype=np.float32)

    def add(self, chunk: str, embedding: np.ndarray) -> None:
        self.chunks.append(chunk)
        if self.embeddings.size == 0:
            self.embeddings = embedding.reshape(1, -1).astype(np.float32)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding.astype(np.float32)])

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.embeddings.size == 0 or len(self.chunks) == 0:
            return []
        query_vec = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_vec)
        if np.isclose(norms, 0).any() or query_norm == 0:
            return []
        similarities = (self.embeddings @ query_vec.T).reshape(-1) / (norms * query_norm)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [
            {
                "text": self.chunks[i],
                "score": float(similarities[i]),
                "index": int(i),
            }
            for i in top_indices
        ]


def _get_openai_client() -> Optional[Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def _safe_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def extract_text_from_file(file_path: Path) -> str:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    if suffix in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8")
    raise ValueError(f"Unsupported file type: {suffix}")


def extract_text_from_upload(uploaded_file: Any) -> str:
    if uploaded_file is None:
        return ""

    filename = getattr(uploaded_file, "name", "") or ""
    suffix = Path(filename).suffix.lower()

    if hasattr(uploaded_file, "getvalue"):
        payload = uploaded_file.getvalue()
    else:
        payload = uploaded_file.read()

    if not payload:
        return ""

    if suffix == ".pdf":
        reader = PdfReader(BytesIO(payload))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    if suffix in {".txt", ".md"}:
        return payload.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if not text.strip():
        return []
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= chunk_size:
        return [cleaned]
    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunk = cleaned[start:end]
        chunks.append(chunk)
        if end >= len(cleaned):
            break
        start += chunk_size - overlap
    return chunks


def build_or_load_index(book_dir: Path = BOOK_DIR) -> SimpleVectorIndex:
    CACHE_DIR.mkdir(exist_ok=True)
    if INDEX_CACHE_PATH.exists():
        try:
            with INDEX_CACHE_PATH.open("rb") as handle:
                return pickle.load(handle)
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not load cached index: %s", exc)

    book_files = sorted(book_dir.glob("*"))
    if not book_files:
        raise FileNotFoundError("No book files found in the book directory")

    text_parts: List[str] = []
    for file_path in book_files:
        if file_path.is_file() and file_path.suffix.lower() in {".pdf", ".txt", ".md"}:
            text_parts.append(extract_text_from_file(file_path))

    combined_text = "\n\n".join(part for part in text_parts if part)
    if not combined_text:
        raise ValueError("No readable text found in the book directory")

    chunks = chunk_text(combined_text)
    index = SimpleVectorIndex(chunks=[])

    client = _get_openai_client()
    if client:
        try:
            embeddings = client.embeddings.create(model=EMBEDDING_MODEL, input=chunks)
            for chunk, item in zip(chunks, embeddings.data):
                index.add(chunk, np.array(item.embedding, dtype=np.float32))
        except Exception as exc:  # pragma: no cover
            logger.warning("Embedding generation failed: %s", exc)
            for chunk in chunks:
                index.add(chunk, np.zeros(3, dtype=np.float32))
    else:
        logger.warning("No OpenAI API key found; using a lightweight lexical fallback")
        for chunk in chunks:
            index.add(chunk, np.zeros(3, dtype=np.float32))

    with INDEX_CACHE_PATH.open("wb") as handle:
        pickle.dump(index, handle)
    return index


def parse_rubric(rubric_text: str) -> List[Dict[str, Any]]:
    criteria: List[Dict[str, Any]] = []
    lines = [line.strip() for line in rubric_text.splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|")]
    for line in table_lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        if cells[0].lower() == "criterion" or re.fullmatch(r"[-: ]+", cells[0]):
            continue
        name = cells[0].replace("**", "")
        marks = int(re.sub(r"[^0-9]", "", cells[1])) if re.search(r"\d", cells[1]) else 0
        description = cells[2].replace("**", "")
        criteria.append({"name": name, "marks": marks, "description": description})
    return criteria


def detect_prompt_injection(text: str) -> Tuple[bool, str]:
    lowered = text.lower()
    patterns = [
        r"ignore instructions",
        r"give full marks",
        r"override grading",
        r"pretend to be",
        r"do not follow",
        r"ignore the rubric",
    ]
    for pattern in patterns:
        if re.search(pattern, lowered):
            return True, f"Prompt injection detected via pattern: {pattern}"
    return False, ""


def _extract_json_payload(text: str) -> Dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"```(?:json)?", "", candidate).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
        return {}


def _call_llm(prompt: str, system_prompt: str = "You are a strict grading assistant.") -> str:
    client = _get_openai_client()
    if not client:
        return ""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""
    except Exception as exc:  # pragma: no cover
        logger.warning("LLM call failed: %s", exc)
        return ""


def generate_search_query(criterion: Dict[str, Any], student_answer: str, questions_text: Optional[str] = None) -> str:
    prompt = f"""
You are creating a search query for a course book index.
Criterion: {criterion['name']}
Description: {criterion['description']}
Questions: {questions_text or 'No custom questions provided.'}
Student answer: {student_answer[:2000]}
Return a short search query of 8-15 words that captures the most relevant concepts from the book.
"""
    answer = _call_llm(prompt, system_prompt="You write concise retrieval queries.")
    return answer.strip() or f"{criterion['name']} {criterion['description']}"


def lexical_search(index: SimpleVectorIndex, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    query_terms = {term.lower() for term in re.findall(r"\w+", query) if len(term) > 2}
    if not query_terms or not index.chunks:
        return []

    scored_chunks: List[Tuple[int, int, str]] = []
    for position, chunk in enumerate(index.chunks):
        chunk_lower = chunk.lower()
        overlap = sum(1 for term in query_terms if term in chunk_lower)
        if overlap > 0:
            scored_chunks.append((overlap, position, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return [
        {"text": chunk, "score": float(overlap), "index": int(index_pos)}
        for overlap, index_pos, chunk in scored_chunks[:top_k]
    ]


def heuristic_grade(criterion: Dict[str, Any], student_answer: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    context_text = "\n\n".join(item["text"] for item in retrieved[:3]) if retrieved else ""
    answer_tokens = {token.lower() for token in re.findall(r"\w+", student_answer) if len(token) > 3}
    context_tokens = {token.lower() for token in re.findall(r"\w+", context_text) if len(token) > 3}
    overlap = len(answer_tokens & context_tokens)
    max_marks = criterion["marks"]
    if overlap > 0:
        score = min(max_marks, max(1, int(round(overlap / max(1, len(context_tokens)) * max_marks))))
    else:
        score = 0
    flags = ["Unsupported by Book"] if overlap == 0 else []
    return {
        "score": score,
        "justification": "Heuristic fallback based on overlap with retrieved book context." if overlap > 0 else "No clear support from the book context was found.",
        "book_quotes": [context_text[:180]] if context_text else [],
        "flags": flags,
    }


def grade_criterion(
    criterion: Dict[str, Any],
    student_answer: str,
    index: SimpleVectorIndex,
    questions_text: Optional[str] = None,
) -> Dict[str, Any]:
    search_query = generate_search_query(criterion, student_answer, questions_text=questions_text)
    retrieved: List[Dict[str, Any]] = []

    if getattr(index, "embeddings", None) is not None and index.embeddings.size > 0:
        client = _get_openai_client()
        if client:
            try:
                embedding_response = client.embeddings.create(model=EMBEDDING_MODEL, input=[search_query])
                query_vector = np.array(embedding_response.data[0].embedding, dtype=np.float32)
                retrieved = index.search(query_vector, top_k=3)
            except Exception as exc:  # pragma: no cover
                logger.warning("Embeddings query failed: %s", exc)
                retrieved = lexical_search(index, search_query, top_k=3)
        else:
            retrieved = lexical_search(index, search_query, top_k=3)
    else:
        retrieved = lexical_search(index, search_query, top_k=3)

    context_text = "\n\n".join(item["text"] for item in retrieved[:3]) if retrieved else "No directly retrieved book context was available."

    prompt = f"""
Strictly grade the student's answer against the rubric criterion using only the provided book context.
Criterion Name: {criterion['name']}
Criterion Description: {criterion['description']}
Questions Provided: {questions_text or 'No custom questions provided.'}
Maximum Marks: {criterion['marks']}
Student Answer: {student_answer}
Book Context:
{context_text}

Return JSON with these exact keys:
- score: integer score out of {criterion['marks']}
- justification: one-line reason grounded in the book context
- book_quotes: array of short supporting quotes from the book context
- flags: array of strings, including "Unsupported by Book" when the answer makes claims not covered by the book
"""
    result_text = _call_llm(prompt, system_prompt="You are a strict rubric-based grader. Use only the provided book context.")
    parsed = _extract_json_payload(result_text)
    if not parsed:
        fallback = heuristic_grade(criterion, student_answer, retrieved)
        parsed = {
            "score": fallback["score"],
            "justification": fallback["justification"],
            "book_quotes": fallback["book_quotes"],
            "flags": fallback["flags"],
        }
    score = int(parsed.get("score", 0))
    score = max(0, min(score, criterion["marks"]))
    flags = parsed.get("flags", []) or []
    if not flags and context_text == "No directly retrieved book context was available.":
        flags.append("Unsupported by Book")
    return {
        "criterion": criterion["name"],
        "score": score,
        "max_marks": criterion["marks"],
        "justification": _safe_text(parsed.get("justification")),
        "book_quotes": parsed.get("book_quotes", []) or [],
        "flags": flags,
        "retrieved_context": retrieved,
        "search_query": search_query,
    }


def verify_grade(
    criterion_result: Dict[str, Any],
    student_answer: str,
    context_text: str,
    questions_text: Optional[str] = None,
) -> Dict[str, Any]:
    prompt = f"""
You are a second-pass auditor. Review the proposed grade and decide if it is justified by the book context.
Questions Provided: {questions_text or 'No custom questions provided.'}
Student Answer: {student_answer}
Proposed Grade: {json.dumps(criterion_result, ensure_ascii=False)}
Book Context:
{context_text}

Return JSON with exact keys:
- score: integer
- justification: one-line corrected reason
- book_quotes: array of short supporting quotes
- flags: array of strings
"""
    result_text = _call_llm(prompt, system_prompt="You are a careful auditor for grading quality.")
    parsed = _extract_json_payload(result_text)
    if not parsed:
        return criterion_result
    corrected = criterion_result.copy()
    corrected["score"] = int(parsed.get("score", criterion_result["score"]))
    corrected["justification"] = _safe_text(parsed.get("justification")) or criterion_result["justification"]
    corrected["book_quotes"] = parsed.get("book_quotes", []) or criterion_result["book_quotes"]
    corrected["flags"] = parsed.get("flags", []) or criterion_result["flags"]
    return corrected


def grade_submission(
    student_answer: str,
    rubric_text: str,
    index: SimpleVectorIndex,
    questions_text: Optional[str] = None,
) -> Dict[str, Any]:
    injection, reason = detect_prompt_injection(student_answer)
    if injection:
        return {
            "blocked": True,
            "reason": reason,
            "criteria": [],
            "total_score": 0,
            "feedback": "Automatic grading stopped because the submission attempted to override instructions.",
            "flags": [reason],
        }

    criteria = parse_rubric(rubric_text)
    criterion_results: List[Dict[str, Any]] = []
    for criterion in criteria:
        intermediate = grade_criterion(criterion, student_answer, index, questions_text=questions_text)
        context_text = "\n\n".join(item["text"] for item in intermediate["retrieved_context"][:3]) if intermediate["retrieved_context"] else "No directly retrieved book context was available."
        verified = verify_grade(intermediate, student_answer, context_text, questions_text=questions_text)
        criterion_results.append(verified)

    total_score = sum(item["score"] for item in criterion_results)
    flags = [flag for item in criterion_results for flag in item.get("flags", []) if flag]
    feedback = "The submission was graded against the book context only."
    if flags:
        feedback += " Some claims were unsupported by the book."
    return {
        "blocked": False,
        "reason": "",
        "criteria": criterion_results,
        "total_score": total_score,
        "feedback": feedback,
        "flags": sorted(set(flags)),
    }


def list_assignment_files(assignments_dir: Path = ASSIGNMENTS_DIR) -> List[Path]:
    return sorted([path for path in assignments_dir.iterdir() if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"}])


def grade_assignment_file(file_path: Path, rubric_text: str, index: SimpleVectorIndex) -> Dict[str, Any]:
    student_answer = extract_text_from_file(file_path)
    result = grade_submission(student_answer, rubric_text, index)
    result["source_file"] = str(file_path.name)
    return result


def load_rubric(rubric_path: Path = RUBRIC_PATH) -> str:
    return rubric_path.read_text(encoding="utf-8")


def load_task_description(task_path: Path = TASK_PATH) -> str:
    return task_path.read_text(encoding="utf-8")
