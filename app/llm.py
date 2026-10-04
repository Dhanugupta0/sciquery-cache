"""LLM client and prompt templates."""

from langchain_openai import ChatOpenAI

from app.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL


def get_llm() -> ChatOpenAI:
    """Create the LLM client. Called once at startup."""
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        temperature=0,
    )


# ---- Prompt templates ----

ANSWER_SYSTEM = """You are a helpful tutor for NCERT Class 10 Science.

RULES:
- Answer ONLY using the context below. Never use outside knowledge.
- If the context does not cover the question, set in_scope to false.
- Write short, clear answers for a Class 10 student. Use simple English.
- Cite the chapter name(s) you used.
- Do NOT say "as I said before" or refer to earlier conversation.

CONTEXT:
{context}

Respond in this exact JSON format (no markdown, no extra text):
{{"in_scope": true/false, "answer": "...", "used_chapters": ["..."]}}"""

ANSWER_HUMAN = "{question}"

REWRITE_SYSTEM = """Rewrite the follow-up message into a standalone question using the conversation history.
Output ONLY the rewritten question, nothing else.

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
