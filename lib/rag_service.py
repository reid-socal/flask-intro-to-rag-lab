from __future__ import annotations

import re
from typing import Any

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "get",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "need",
    "of",
    "on",
    "or",
    "our",
    "should",
    "so",
    "the",
    "their",
    "to",
    "use",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
    "you",
    "your",
}


def tokenize(text: str) -> set[str]:
    """Convert text into a set of searchable lowercase tokens."""
    raw_tokens = re.findall(r"[a-z0-9']+", text.lower())
    tokens = set()

    for token in raw_tokens:
        token = token.strip("'")
        if len(token) <= 1:
            continue
        if token in STOPWORDS:
            continue
        tokens.add(token)

    return tokens


def document_search_text(document: dict[str, Any]) -> str:
    """Combine searchable document fields into one text value."""
    title = document.get("title", "")
    category = document.get("category", "")
    tags = " ".join(document.get("tags", []))
    text = document.get("text", "")

    return f"{title} {category} {tags} {text}"


def score_document(query: str, document: dict[str, Any]) -> dict[str, Any]:
    """Score a document using keyword overlap."""
    query_tokens = tokenize(query)
    document_tokens = tokenize(document_search_text(document))
    title_tokens = tokenize(document.get("title", ""))

    matched_terms = query_tokens & document_tokens
    title_matches = query_tokens & title_tokens

    score = len(matched_terms) + (0.5 * len(title_matches))

    return {
        "document": document,
        "score": score,
        "matched_terms": sorted(matched_terms),
    }


def retrieve_context(
    query: str,
    documents: list[dict[str, Any]],
    limit: int = 2,
    minimum_score: float = 1.0,
) -> list[dict[str, Any]]:
    """Select the most relevant documents for the query."""
    scored = [score_document(query, document) for document in documents]
    matches = [match for match in scored if match["score"] >= minimum_score]
    matches.sort(key=lambda match: match["score"], reverse=True)

    return matches[:limit]


def format_context(context_matches: list[dict[str, Any]]) -> str:
    """Format retrieved documents into a context block for the prompt."""
    if not context_matches:
        return "No relevant context was found."

    blocks = []
    for match in context_matches:
        document = match["document"]
        blocks.append(
            f"Source ID: {document['id']}\n"
            f"Title: {document['title']}\n"
            f"Category: {document['category']}\n"
            f"Content: {document['text']}"
        )

    return "\n\n---\n\n".join(blocks)


def build_prompt(query: str, context_matches: list[dict[str, Any]]) -> str:
    """Build a structured prompt with instructions, context, question, and requirements."""
    context = format_context(context_matches)

    return (
        "Instructions:\n"
        "You are an internal company assistant. Use only the information in the "
        "Context section below to answer the question. Do not invent policies, "
        "numbers, deadlines, or other details that are not in the context. If the "
        "context does not contain enough information, say so.\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{query}\n\n"
        "Response requirements:\n"
        "- Answer clearly and concisely using only the provided context.\n"
        "- Do not invent or assume unsupported details.\n"
        "- Mention the Source ID of each document you rely on.\n"
        "- If the context is not sufficient, say that the approved documents do "
        "not contain enough information.\n"
    )


def source_metadata(match: dict[str, Any]) -> dict[str, str]:
    """Return source information that is safe to expose in the API response."""
    document = match["document"]

    return {
        "id": document["id"],
        "title": document["title"],
    }