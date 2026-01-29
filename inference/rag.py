import argparse
import importlib.util
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Dict, Tuple


WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    tags: List[str] = field(default_factory=list)


class KnowledgeBase:
    def __init__(self, workspace: str, root: Path) -> None:
        self.workspace = workspace
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.data_path = self.root / "kb.json"
        self.chunks: List[Chunk] = []
        if self.data_path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self.data_path.read_text())
        self.chunks = [Chunk(**chunk) for chunk in data.get("chunks", [])]

    def save(self) -> None:
        payload = {
            "workspace": self.workspace,
            "chunks": [chunk.__dict__ for chunk in self.chunks],
        }
        self.data_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    def ingest_files(self, paths: Iterable[Path], tags: List[str], chunk_size: int) -> None:
        new_chunks: List[Chunk] = []
        for path in paths:
            text = load_text(path)
            for idx, chunk_text in enumerate(chunk_texts(text, chunk_size)):
                chunk_id = f"{path.name}-{idx:04d}"
                new_chunks.append(Chunk(chunk_id=chunk_id, text=chunk_text, source=str(path), tags=tags))
        self.chunks.extend(new_chunks)
        self.save()

    def search(self, query: str, top_k: int) -> List[Tuple[Chunk, float]]:
        if not self.chunks:
            return []
        tf_idf_matrix, vocabulary = build_tfidf_matrix([chunk.text for chunk in self.chunks])
        query_vector = tf_idf_vector(query, vocabulary, len(self.chunks), tf_idf_matrix)
        scores = []
        for idx, doc_vector in enumerate(tf_idf_matrix):
            score = cosine_similarity(query_vector, doc_vector)
            scores.append((self.chunks[idx], score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]


def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        if importlib.util.find_spec("pypdf") is None:
            raise RuntimeError("pypdf is required to parse PDFs. Install it with `pip install pypdf`.")
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix in {".docx", ".doc"}:
        if importlib.util.find_spec("docx") is None:
            raise RuntimeError("python-docx is required to parse Word docs. Install it with `pip install python-docx`.")
        from docx import Document

        doc = Document(str(path))
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_texts(text: str, chunk_size: int) -> Iterable[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    return [normalized[i:i + chunk_size] for i in range(0, len(normalized), chunk_size)]


def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


def build_tfidf_matrix(docs: List[str]) -> Tuple[List[Dict[str, float]], Dict[str, int]]:
    doc_freq: Dict[str, int] = {}
    term_freqs: List[Dict[str, int]] = []
    for doc in docs:
        tokens = tokenize(doc)
        tf: Dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        term_freqs.append(tf)
        for token in set(tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1

    vocabulary = {term: idx for idx, term in enumerate(doc_freq.keys())}
    tf_idf_matrix: List[Dict[str, float]] = []
    doc_count = len(docs)
    for tf in term_freqs:
        weighted: Dict[str, float] = {}
        for term, count in tf.items():
            idf = math.log((doc_count + 1) / (doc_freq[term] + 1)) + 1
            weighted[term] = count * idf
        tf_idf_matrix.append(weighted)
    return tf_idf_matrix, vocabulary


def tf_idf_vector(query: str, vocabulary: Dict[str, int], doc_count: int, matrix: List[Dict[str, float]]) -> Dict[str, float]:
    doc_freq: Dict[str, int] = {}
    for doc in matrix:
        for term in doc.keys():
            doc_freq[term] = doc_freq.get(term, 0) + 1
    query_tf: Dict[str, int] = {}
    for token in tokenize(query):
        if token in vocabulary:
            query_tf[token] = query_tf.get(token, 0) + 1
    query_vector: Dict[str, float] = {}
    for term, count in query_tf.items():
        idf = math.log((doc_count + 1) / (doc_freq.get(term, 0) + 1)) + 1
        query_vector[term] = count * idf
    return query_vector


def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(value * vec_b.get(term, 0.0) for term, value in vec_a.items())
    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage RAG knowledge bases.")
    parser.add_argument("--workspace", required=True, help="Workspace name for the knowledge base.")
    parser.add_argument("--store", default="rag_data", help="Root folder for knowledge bases.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into the workspace.")
    ingest_parser.add_argument("paths", nargs="+", help="Paths to documents (txt, pdf, docx).")
    ingest_parser.add_argument("--tags", default="", help="Comma-separated tags to assign.")
    ingest_parser.add_argument("--chunk-size", type=int, default=1200, help="Chunk size in characters.")

    query_parser = subparsers.add_parser("query", help="Query the workspace.")
    query_parser.add_argument("query", help="User query.")
    query_parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to return.")

    args = parser.parse_args()
    root = Path(args.store) / args.workspace
    kb = KnowledgeBase(args.workspace, root)

    if args.command == "ingest":
        tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
        paths = [Path(path).expanduser() for path in args.paths]
        kb.ingest_files(paths, tags, args.chunk_size)
        print(f"Ingested {len(paths)} file(s) into workspace '{args.workspace}'.")
        return

    if args.command == "query":
        results = kb.search(args.query, args.top_k)
        for chunk, score in results:
            print(f"[{score:.3f}] {chunk.source}#{chunk.chunk_id} tags={','.join(chunk.tags) or '-'}")
            print(chunk.text)
            print("---")


if __name__ == "__main__":
    main()
