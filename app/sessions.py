"""In-memory session store for multi-turn conversations."""

import uuid


class SessionStore:
    """Holds per-session conversation history and the last standalone question."""

    def __init__(self):
        self._sessions: dict[str, dict] = {}

    def create(self) -> str:
        sid = uuid.uuid4().hex[:12]
        self._sessions[sid] = {
            "history": [],                # list of {"role": ..., "content": ...}
            "last_standalone_question": None,
            "last_answer": None,
            "last_citations": [],
        }
        return sid

    def get(self, session_id: str) -> dict | None:
        return self._sessions.get(session_id)

    def add_turn(self, session_id: str, role: str, content: str):
        session = self._sessions[session_id]
        session["history"].append({"role": role, "content": content})
        # Keep only last 10 turns to bound memory
        if len(session["history"]) > 10:
            session["history"] = session["history"][-10:]

    def set_last_standalone(self, session_id: str, question: str):
        self._sessions[session_id]["last_standalone_question"] = question

    def set_last_answer(self, session_id: str, answer: str, citations: list[str]):
        self._sessions[session_id]["last_answer"] = answer
        self._sessions[session_id]["last_citations"] = citations
