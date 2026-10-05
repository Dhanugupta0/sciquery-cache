"""LLM client and prompt templates."""

from langchain_openai import ChatOpenAI

from app.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_NUMERIC_MODEL


def get_llm() -> ChatOpenAI:
    """Create the general LLM client. Called once at startup."""
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        temperature=0,
    )


def get_numeric_llm() -> ChatOpenAI:
    """Create the numerical LLM client using LLM_NUMERIC_MODEL."""
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_NUMERIC_MODEL,
        temperature=0,
    )


# ---- Prompt templates ----

ANSWER_SYSTEM = """You are a helpful tutor for NCERT Class 10 Science.

RULES:
- Answer using the provided context. Answer from partial context where possible.
- Set in_scope to false ONLY when the context is clearly unrelated or completely lacks relevant information.
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
