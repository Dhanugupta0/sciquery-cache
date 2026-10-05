"""Pipeline integration tests — mock the LLM, exercise the full graph.

Tests:
1. Cache hit → zero LLM calls (mocked LLM's invoke is never called).
2. Out-of-scope question → polite decline, no LLM call.
3. Multi-turn follow-up → rewrite resolves pronouns using history.
"""

import json
import os
import sys
import tempfile
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def pipeline_env():
    """Initialize the pipeline once with a temp cache DB."""
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_db = f.name
    f.close()

    with patch("app.cache.store.CACHE_DB_PATH", tmp_db):
        from app.graph import init_pipeline, run_pipeline
        from app.sessions import SessionStore

        sessions = SessionStore()
        init_pipeline(sessions=sessions)

        yield {
            "run": run_pipeline,
            "sessions": sessions,
            "tmp_db": tmp_db,
        }

    if os.path.exists(tmp_db):
        os.unlink(tmp_db)


class TestCacheHitZeroLLM:
    """A cache hit must NOT call the LLM at all."""

    def test_cache_hit_zero_llm_calls(self, pipeline_env):
        import app.graph as graph_mod
        run = pipeline_env["run"]
        sessions = pipeline_env["sessions"]

        real_llm = graph_mod._llm

        # Step 1: Mock LLM for the first turn to populate cache
        mock_llm_step1 = MagicMock()
        mock_answer_json = json.dumps({
            "in_scope": True,
            "answer": "Refraction is the bending of light as it passes from one transparent substance into another.",
            "used_chapters": ["Light – Reflection and Refraction"],
        })
        mock_llm_step1.invoke.return_value = MagicMock(content=mock_answer_json)
        graph_mod._llm = mock_llm_step1

        try:
            sid1 = sessions.create()
            result1 = run(sid1, "What is refraction?")
            assert result1["reply"], "First call should return an answer"
            assert result1["cache_hit"] is False
            assert mock_llm_step1.invoke.call_count >= 1

            # Step 2: Second call with same question must hit cache with 0 LLM calls
            mock_llm_step2 = MagicMock()
            mock_llm_step2.invoke.side_effect = AssertionError("LLM should NOT be called on cache hit")
            graph_mod._llm = mock_llm_step2

            sid2 = sessions.create()
            result2 = run(sid2, "What is refraction?")
            assert result2["cache_hit"] is True, f"Expected cache hit, got: {result2.get('cache_reason')}"
            assert result2["reply"] == result1["reply"]
            mock_llm_step2.invoke.assert_not_called()
        finally:
            graph_mod._llm = real_llm


class TestOutOfScope:
    """An out-of-scope question is declined without calling the LLM, returning empty citations."""

    @pytest.mark.parametrize("off_topic_q", [
        "Who won the 2024 Cricket World Cup?",
        "What is the stock price of Apple?",
        "Who directed the movie Interstellar?",
        "What is the capital of France?",
        "Who is the president of the United States?",
    ])
    def test_out_of_scope_declined_empty_citations(self, pipeline_env, off_topic_q):
        import app.graph as graph_mod
        run = pipeline_env["run"]
        sessions = pipeline_env["sessions"]

        sid = sessions.create()

        # Replace the LLM with a mock that fails if called
        real_llm = graph_mod._llm
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = AssertionError("LLM should NOT be called for out-of-scope question")
        graph_mod._llm = mock_llm

        try:
            result = run(sid, off_topic_q)
            assert result.get("in_scope") is False
            assert result.get("citations") == [], f"Expected empty citations, got: {result.get('citations')}"
            reply_lower = result["reply"].lower()
            assert "textbook" in reply_lower or "covered" in reply_lower
            mock_llm.invoke.assert_not_called()
        finally:
            graph_mod._llm = real_llm


class TestMultiTurnFollowUp:
    """A follow-up that uses pronouns gets rewritten using history."""

    def test_followup_uses_context(self, pipeline_env):
        import app.graph as graph_mod
        run = pipeline_env["run"]
        sessions = pipeline_env["sessions"]

        real_llm = graph_mod._llm
        mock_llm = MagicMock()

        # Handler for LLM calls based on prompt
        def mock_invoke_handler(messages):
            # Check if this is a rewrite prompt or answer generation prompt
            system_msg = messages[0]["content"] if messages else ""
            user_msg = messages[1]["content"] if len(messages) > 1 else ""

            if "rewrite" in system_msg.lower() or "standalone question" in system_msg.lower():
                # Rewriter should turn "What is its formula?" into a standalone question
                return MagicMock(content="What is the formula for Ohm's law?")
            elif "Ohm's law" in user_msg or "formula" in user_msg:
                # Answer generation for Ohm's law
                return MagicMock(content=json.dumps({
                    "in_scope": True,
                    "answer": "Ohm's law states that V = IR, where V is voltage, I is current, and R is resistance.",
                    "used_chapters": ["Electricity"],
                }))
            else:
                return MagicMock(content=json.dumps({
                    "in_scope": True,
                    "answer": "Default scientific answer.",
                    "used_chapters": ["Electricity"],
                }))

        mock_llm.invoke.side_effect = mock_invoke_handler
        graph_mod._llm = mock_llm

        try:
            sid = sessions.create()

            # Turn 1: standalone question
            result1 = run(sid, "What is Ohm's law?")
            assert result1["reply"], "First answer should exist"
            assert result1["msg_type"] == "STANDALONE"

            # Turn 2: follow-up with pronoun
            result2 = run(sid, "What is its formula?")
            assert result2["msg_type"] == "FOLLOW_UP"
            assert result2.get("standalone_question") == "What is the formula for Ohm's law?"

            # The answer should reference Ohm's law and V = IR
            reply_lower = result2["reply"].lower()
            ohm_keywords = ["ohm", "v = ir", "voltage", "resistance", "current"]
            found = any(k in reply_lower for k in ohm_keywords)
            assert found, f"Follow-up should relate to Ohm's law, got: {result2['reply'][:200]}"
            assert "Electricity" in result2["citations"]
        finally:
            graph_mod._llm = real_llm


class TestFollowUpClassification:
    """Test follow-up vs standalone classification rules."""

    @pytest.mark.parametrize("msg", [
        "Function of the right ventricle",
        "SI unit of resistivity",
        "veins and arteries difference",
        "What are the chemical properties of bases when they react with metals?",
        "Why does iron lose its shine in air?",
    ])
    def test_standalone_messages(self, pipeline_env, msg):
        from app.graph import classify_message, is_followup_message
        sessions = pipeline_env["sessions"]
        sid = sessions.create()
        sessions.set_last_standalone(sid, "What is refraction?")
        sessions.set_last_answer(sid, "Refraction is bending of light.", ["Light – Reflection and Refraction"])

        assert not is_followup_message(msg)
        state = {"session_id": sid, "message": msg}
        res = classify_message(state)
        assert res["msg_type"] == "STANDALONE"

    @pytest.mark.parametrize("msg", [
        "What about its laws?",
        "What is its unit?",
        "Why so?",
    ])
    def test_followup_messages(self, pipeline_env, msg):
        from app.graph import classify_message, is_followup_message
        sessions = pipeline_env["sessions"]
        sid = sessions.create()
        sessions.set_last_standalone(sid, "What is Ohm's law?")
        sessions.set_last_answer(sid, "Ohm's law states V=IR.", ["Electricity"])

        assert is_followup_message(msg)
        state = {"session_id": sid, "message": msg}
        res = classify_message(state)
        assert res["msg_type"] == "FOLLOW_UP"
