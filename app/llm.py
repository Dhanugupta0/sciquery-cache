"""LLM client and prompt templates."""

import socket

# Force IPv4 preference to avoid broken IPv6 routes causing connect timeouts
_orig_getaddrinfo = socket.getaddrinfo

def _ipv4_first_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    res = _orig_getaddrinfo(host, port, family, type, proto, flags)
    return sorted(res, key=lambda x: 0 if x[0] == socket.AF_INET else 1)

socket.getaddrinfo = _ipv4_first_getaddrinfo

from langchain_openai import ChatOpenAI

from app.config import (
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_NUMERIC_MODEL,
    LLM_FALLBACK_MODEL,
    get_api_key,
)


def get_llm() -> ChatOpenAI:
    """Create the general LLM client. Called once at startup."""
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=get_api_key(),
        model=LLM_MODEL,
        temperature=0,
        max_tokens=600,
        max_retries=0,
    )


def get_numeric_llm() -> ChatOpenAI:
    """Create the numerical LLM client using LLM_NUMERIC_MODEL."""
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=get_api_key(),
        model=LLM_NUMERIC_MODEL,
        temperature=0,
        max_tokens=800,
        max_retries=0,
    )


def get_fallback_llm() -> ChatOpenAI:
    """Create the fallback LLM client using LLM_FALLBACK_MODEL for 429/404 failover."""
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=get_api_key(),
        model=LLM_FALLBACK_MODEL,
        temperature=0,
        max_tokens=800,
        max_retries=0,
    )


# ---- Prompt templates ----

ANSWER_SYSTEM = """You are a helpful tutor for NCERT Class 10 Science.

RULES:
- Answer using the provided context. Answer from partial context or concept descriptions where possible.
- If the question is covered in the context, answer it clearly for a Class 10 student. Do not decline just because a formal definition is missing; explain what it is from its properties and behavior described.
- If the question asks to name, list, explain, or define something that is not covered or directly supported by the context, you MUST set in_scope to false.
- Write short, clear answers for a Class 10 student. Use simple English.
- Cite the chapter name(s) you used in used_chapters.
- Do NOT say "as I said before" or refer to earlier conversation.
- Do NOT write 'Chapter:' or the chapter name inside your answer text.

CONTEXT:
{context}

Respond in this exact JSON format (no markdown, no extra text):
{{"in_scope": true/false, "answer": "...", "used_chapters": ["..."]}}"""


NUMERIC_SYSTEM = """You are a helpful tutor for NCERT Class 10 Science numerical problems.

RULES:
- Answer using the provided context and textbook principles.
- Write plain text only. Do NOT use LaTeX, do NOT use \\( \\), \\[ \\], or $ symbols. Use simple plain text for formulas, fractions, and units (e.g. 1/v - 1/u = 1/f, cm).
- Write the formula (remember: lens formula is 1/v - 1/u = 1/f; mirror formula is 1/v + 1/u = 1/f).
- State the sign convention.
- Substitute the values with their signs.
- Compute step by step, and check the result before answering.
- Write a clear step-by-step solution for a Class 10 student.
- Cite the chapter name(s) you used in used_chapters.
- Do NOT write 'Chapter:' or the chapter name inside your answer text.

CONTEXT:
{context}

Respond in this exact JSON format (no markdown, no extra text):
{{"in_scope": true/false, "answer": "...", "used_chapters": ["..."]}}"""

ANSWER_HUMAN = "{question}"

REWRITE_SYSTEM = """You are a question rewriter.
Given the conversation history, rewrite the user message into a standalone question.

RULES:
- If the user message is already a self-contained question that does not depend on the previous conversation, return it UNCHANGED.
- Only resolve pronouns (it, its, this, that, etc.) or ellipsis that refer to the previous conversation.
- Output ONLY the standalone question, nothing else.

Conversation history (most recent last):
{history}"""

REWRITE_HUMAN = "Follow-up message: {message}"

STYLE_SYSTEM = """You are a helpful tutor. The student wants you to change HOW the answer is presented.
Rewrite the previous answer according to the student's request.
Keep the same facts. Write for a Class 10 student. Use simple English.

Previous answer:
{previous_answer}

Chapters cited: {citations}"""

STYLE_HUMAN = "Student's request: {request}"
