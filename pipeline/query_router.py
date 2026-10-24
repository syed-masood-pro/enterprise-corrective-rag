"""
Lightweight LLM router that classifies the incoming query and (for
non-RAG intents) generates a direct reply — avoiding the full retrieval
+ CRAG pipeline for greetings and small talk.

Intents:
  - greet     : "hi", "hello", "good morning", "hey there"
  - chitchat  : "how are you?", "who are you?", "thanks", "bye"
  - rag       : any real question that needs document retrieval

Design goals:
  * Single LLM call returns BOTH the intent AND (for non-RAG) the reply
    so we never pay two round-trips for a greeting.
  * Fail-safe: any parse error defaults to `rag` — we would rather run
    the full pipeline than hallucinate an answer.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

Intent = Literal["greet", "chitchat", "rag"]
_VALID: set[str] = {"greet", "chitchat", "rag"}

# ── Fast-path heuristic vocab (no LLM call) ───────────────────────────────────
_GREET_EXACT: set[str] = {
    "hi", "hello", "hey", "yo", "hii", "helo", "hiya", "heya",
    "good morning", "good afternoon", "good evening", "hey there",
    "hello there", "hi there",
}
_CHITCHAT_EXACT: set[str] = {
    "thanks", "thank you", "thanks a lot", "thank you so much", "thx", "ty",
    "bye", "goodbye", "cya", "see you", "see ya", "good night",
    "how are you", "how are you doing", "who are you", "what are you",
    "what can you do", "what do you do",
}
# Words that strongly signal a real information-seeking question → RAG.
_RAG_QUESTION_WORDS: tuple[str, ...] = (
    "what", "how", "why", "when", "where", "which", "who", "whom", "whose",
    "explain", "describe", "list", "show", "give", "find", "summarize",
    "summarise", "compare", "difference", "define", "tell me about",
)

_GREET_REPLY = (
    "Hi! I'm your document assistant — ask me anything about your uploaded files."
)
_THANKS_REPLY = "You're welcome! Let me know if you have another question."
_BYE_REPLY = "Goodbye! 👋"

_SYSTEM = (
    "You are the front-door router for a RAG assistant.\n"
    "Classify the user's message into ONE of these intents:\n"
    "  - greet    : simple greetings (hi, hello, good morning, hey).\n"
    "  - chitchat : small talk, thanks, bye, 'who are you', 'what can you do'.\n"
    "  - rag      : any question that needs document lookup / factual answer.\n\n"
    "Respond with a JSON object ONLY:\n"
    '  {\"intent\": \"<greet|chitchat|rag>\", \"reply\": \"<text>\"}\n\n'
    "Rules:\n"
    "  * If intent is 'rag', set reply to an empty string \"\".\n"
    "  * If intent is 'greet' or 'chitchat', write a short, warm one-line reply\n"
    "    introducing yourself as a document-search assistant when appropriate.\n"
    "  * Never invent facts. Never answer factual questions here — mark them 'rag'.\n"
    "  * No markdown, no code fences, JSON only."
)


def _normalize(text: str) -> str:
    """Lowercase, trim, and strip trailing punctuation for heuristic matching."""
    return re.sub(r"[!?.,\s]+$", "", text.strip().lower())


def _heuristic_decision(query: str) -> "RouterDecision | None":
    """
    Cheap, deterministic routing for obvious cases — returns a decision or
    None (meaning "unsure, fall back to the LLM router").

    Saves an LLM round-trip for the vast majority of traffic:
      * exact greetings / thanks / bye
      * clearly information-seeking questions (question word or length)
    """
    norm = _normalize(query)
    if not norm:
        return RouterDecision("rag", "")

    if norm in _GREET_EXACT:
        return RouterDecision("greet", _GREET_REPLY)
    if norm in _CHITCHAT_EXACT:
        reply = _BYE_REPLY if norm in {"bye", "goodbye", "cya", "see you", "see ya"} else _THANKS_REPLY
        return RouterDecision("chitchat", reply)

    words = norm.split()
    # A real question: starts with a question word, or contains one + is long,
    # or is simply long enough that it can't be small talk.
    if words[0] in _RAG_QUESTION_WORDS or norm.startswith("tell me about"):
        return RouterDecision("rag", "")
    if len(words) >= 5:
        return RouterDecision("rag", "")

    # Short, ambiguous (2–4 words, no question word) → let the LLM decide.
    return None


@dataclass
class RouterDecision:
    intent: Intent
    reply: str  # non-empty only when intent != 'rag'


async def route_query(query: str, llm: ChatGroq) -> RouterDecision:
    """
    Classify and (optionally) answer the query.

    First tries a zero-cost heuristic; only falls back to a single LLM call
    for genuinely ambiguous short inputs.

    Returns:
        RouterDecision(intent, reply).  reply is "" for rag intent.
    """
    # Fast-path heuristics (avoid an LLM call for obvious cases)
    decision = _heuristic_decision(query)
    if decision is not None:
        logger.debug("Router heuristic decision: intent=%s", decision.intent)
        return decision

    try:
        resp = await llm.ainvoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=query),
        ])
        raw = (resp.content or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        obj = json.loads(raw)
        intent = str(obj.get("intent", "")).strip().lower()
        reply = str(obj.get("reply", "")).strip()
        if intent not in _VALID:
            logger.warning("Router returned invalid intent %r — defaulting to rag", intent)
            return RouterDecision("rag", "")
        if intent == "rag":
            reply = ""  # never let the router answer factual questions
        logger.info("Router decision: intent=%s (reply_len=%d)", intent, len(reply))
        return RouterDecision(intent, reply)  # type: ignore[arg-type]
    except Exception as exc:
        logger.warning("Router LLM call failed (%s) — defaulting to rag", exc)
        return RouterDecision("rag", "")
