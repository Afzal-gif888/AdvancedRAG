"""Conversation handling: bounded, normalised history and follow-up query resolution.

History (Backend owns persistence; Agent Core only shapes what the LLM sees):
- only user/assistant turns with content, most recent last;
- at most MAX_HISTORY_MESSAGES turns, each truncated to MAX_HISTORY_MESSAGE_CHARS, and at most
  MAX_HISTORY_CHARS in total, so a long conversation never crowds out the current question;
- old `[n]` source markers are removed from assistant turns: they referred to an earlier source
  list and would otherwise be confused with this turn's numbered sources.

Follow-up resolution: the LLM rewrites the latest message into a standalone search query using
the recent conversation ("Can I do that from my apartment?" -> "remote work from home policy").
If the rewrite fails, retrieval falls back to the previous user question + the current message, so
the subject of the conversation is never silently lost. A rewrite failure that would equally break
the answer call (rate limited, gateway unreachable, bad credentials) fails the turn fast instead of
stacking a second call (FAIL_FAST_REASONS).
"""
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from agent_core import prompts
from agent_core.config import Settings
from agent_core.errors import LLM_RATE_LIMITED, AgentError
from agent_core.schemas import HistoryMessage

log = logging.getLogger("advancerag.agent.query")

# Rewrite failures that would equally break the answer call (gateway/service unreachable, bad
# credentials, not configured): fail the turn instead of falling back. Model-specific failures
# (timeout, provider 5xx, bad request for the fast model, invalid output) still fall back.
FAIL_FAST_REASONS = {"LLM_UNAVAILABLE", "llm_service_unreachable", "LLM_AUTH_FAILED", "LLM_NOT_CONFIGURED"}

_OLD_MARKERS = re.compile(r"\s?[\[【]\s*\d{1,3}(?:\s*[,;]\s*\d{1,3})*\s*[\]】]")
MAX_QUERY_CHARS = 1000


def normalize_history(history: List[HistoryMessage], cfg: Settings) -> List[Dict[str, str]]:
    kept: List[Dict[str, str]] = []
    used = 0
    for m in reversed(history):
        content = m.content.strip()
        if m.role == "assistant":
            content = _OLD_MARKERS.sub("", content).strip()
        if not content:
            continue
        if len(content) > cfg.MAX_HISTORY_MESSAGE_CHARS:
            content = content[: cfg.MAX_HISTORY_MESSAGE_CHARS] + " …"
        if len(kept) >= cfg.MAX_HISTORY_MESSAGES or used + len(content) > cfg.MAX_HISTORY_CHARS:
            break
        used += len(content)
        kept.append({"role": m.role, "content": content})
    return list(reversed(kept))


@dataclass(frozen=True)
class ResolvedQuery:
    text: str
    # "original" | "rewritten" | "fallback_with_subject" | "fallback_with_previous_question"
    method: str
    rejected_reason: Optional[str] = None  # why an LLM rewrite was not used (logs only)


# ---------------------------------------------------------------- subject (entity) tracking
#
# Deterministic guard around the LLM rewrite: names/model codes the conversation is about
# ("Aurora X2", "Halo Display", "SKU ZRy-SL18") must survive the rewrite, and the rewrite must not
# invent a different one ("Aurora X1"). Entities are capitalised or alphanumeric-code phrases with
# leading/trailing function words removed; matching is case-insensitive.

_CANDIDATE = re.compile(r"(?:[A-Z][\w-]*|[A-Za-z]*\d[\w-]*)(?:\s+(?:[A-Z][\w-]*|[A-Za-z]*\d[\w-]*))*")
_STOP = {
    "a", "an", "and", "also", "any", "are", "can", "could", "did", "do", "does", "for", "how", "i", "if", "in", "is",
    "it", "its", "may", "my", "no", "of", "ok", "okay", "on", "or", "our", "please", "should", "so", "tell", "that",
    "the", "their", "then", "these", "they", "this", "those", "to", "was", "we", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "would", "yes", "you", "your", "about", "and what", "q", "query",
}
# Text a model writes when it explains the task instead of answering it.
_META = re.compile(r"\b(rewrite|standalone|search (engine|query)|the (user|conversation|latest message)|we need to|"
                   r"resolv(e|ing) (the )?pronouns?|as an ai|here is|output:)\b", re.I)
MAX_REWRITE_CHARS = 300
MAX_REWRITE_WORDS = 40


def _sentence_start(text: str, pos: int) -> bool:
    before = text[:pos].rstrip()
    return not before or before[-1] in ".!?:\n"


def extract_entities(text: str) -> List[str]:
    text = text or ""
    found: List[str] = []
    for match in _CANDIDATE.finditer(text):
        tokens = match.group(0).split()
        stripped_lead = False
        while tokens and tokens[0].lower() in _STOP:
            tokens.pop(0)
            stripped_lead = True
        while tokens and tokens[-1].lower() in _STOP:
            tokens.pop()
        # A lone capitalised word that merely starts a sentence ("Up to three days…") is not a name.
        if (len(tokens) == 1 and not stripped_lead and _sentence_start(text, match.start())
                and tokens[0].isalpha() and tokens[0][1:].islower()):
            continue
        phrase = " ".join(tokens)
        if phrase and not phrase.replace(",", "").replace(".", "").isdigit() and phrase.lower() not in _STOP:
            if phrase.lower() not in (f.lower() for f in found):
                found.append(phrase)
    return found


# "it/its/this/that/they/them…": the message refers back to something named earlier.
_PRONOUN = re.compile(r"\b(it|its|it's|this|that|these|those|they|them|their|theirs|one|ones)\b", re.I)
# "And in France?", "What about the X1?": an elliptical follow-up that continues the previous turn.
_ELLIPSIS = re.compile(r"^\s*(and|also|plus|what about|how about|same for|as for)\b", re.I)


def has_pronoun(message: str) -> bool:
    return bool(_PRONOUN.search(message))


def is_follow_up_form(message: str) -> bool:
    return has_pronoun(message) or bool(_ELLIPSIS.match(message)) or len(message.split()) <= 3


def previous_subject(history: List[Dict[str, str]]) -> List[str]:
    """Names in the most recent user turn that names something, else the latest assistant turn."""
    for role in ("user", "assistant"):
        for m in reversed(history):
            if m["role"] == role:
                names = extract_entities(m["content"])
                if names:
                    return names
    return []


def subject_entities(message: str, history: List[Dict[str, str]]) -> List[str]:
    """What the current turn is about. A follow-up ("Does it…", "And in France?") is about the
    previous subject; a self-contained question is about what it names."""
    previous = previous_subject(history)
    if is_follow_up_form(message) and previous:
        return previous
    return extract_entities(message) or previous


def _shares_word(entity: str, others: List[str]) -> bool:
    words = {w.lower() for w in entity.split()}
    return any(words & {w.lower() for w in o.split()} for o in others)


def _contains(haystack: str, entity: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(entity.lower())}(?![\w-])", haystack.lower()) is not None


def rejection_reason(rewritten: str, message: str, history: List[Dict[str, str]], finish_reason: Optional[str] = None) -> Optional[str]:
    """Why an LLM rewrite must not be used as the search query (None = acceptable)."""
    if finish_reason == "length":
        return "truncated"
    if len(rewritten) > MAX_REWRITE_CHARS or len(rewritten.split()) > MAX_REWRITE_WORDS:
        return "too_long"
    if _META.search(rewritten):
        return "meta_text"
    conversation = " ".join([message, *(m["content"] for m in history)])
    if any(not _contains(conversation, e) for e in extract_entities(rewritten)):
        return "introduced_entity"  # e.g. switched from Aurora X2 to Aurora X1
    previous, own = previous_subject(history), extract_entities(message)
    if is_follow_up_form(message) and previous:
        # A pronoun must resolve to the previous subject. An elliptical follow-up may instead switch
        # to a sibling item it names ("And what about the Aurora X1?" after the Aurora X2); a named
        # attribute such as a country ("And in France?") does not replace the subject.
        siblings = [] if has_pronoun(message) else [e for e in own if _shares_word(e, previous)]
        required = previous + siblings
    else:
        required = own
    if required and not any(_contains(rewritten, e) for e in required):
        return "dropped_subject"
    return None


def _fallback(message: str, history: List[Dict[str, str]], rejected: Optional[str] = None) -> ResolvedQuery:
    """Deterministic resolution when the LLM rewrite is unavailable or unusable: attach the
    conversation's subject to a follow-up ("Aurora X2: And what about its operating temperature?").
    Only without any named subject does it fall back to the previous user question."""
    previous = previous_subject(history)
    if is_follow_up_form(message) and previous:
        return ResolvedQuery(f"{', '.join(previous)}: {message}"[:MAX_QUERY_CHARS], "fallback_with_subject", rejected)
    if extract_entities(message):
        return ResolvedQuery(message, "original", rejected)
    question = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
    if not question:
        return ResolvedQuery(message, "original", rejected)
    return ResolvedQuery(f"{question}\n{message}"[:MAX_QUERY_CHARS], "fallback_with_previous_question", rejected)


async def resolve_search_query(llm, cfg: Settings, message: str, history: List[Dict[str, str]]) -> ResolvedQuery:
    """`llm` is an LLMClient (anything with `generate(messages, model=, temperature=, max_tokens=)`)."""
    if not history:
        return ResolvedQuery(message, "original")
    if not cfg.QUERY_REWRITE_ENABLED:
        return _fallback(message, history)
    try:
        result = await llm.generate(prompts.rewrite_messages(message, history), model=cfg.AGENT_FAST_MODEL,
                                    temperature=0, max_tokens=120)
    except AgentError as exc:
        if exc.code == LLM_RATE_LIMITED or exc.reason in FAIL_FAST_REASONS:
            raise  # the answer call would fail the same way: do not stack a second call
        log.warning("query_rewrite_failed_using_fallback", extra={"reason": exc.reason})
        return _fallback(message, history, exc.reason)
    rewritten = clean_rewrite(result.content)
    reason = "empty" if rewritten is None else rejection_reason(rewritten, message, history, result.finish_reason)
    if reason or rewritten is None:
        log.warning("query_rewrite_rejected_using_fallback", extra={"reason": reason, "model": result.model})
        return _fallback(message, history, reason)
    return ResolvedQuery(rewritten, "rewritten")


def clean_rewrite(text: Optional[str]) -> Optional[str]:
    """Strip quotes/labels around a single search query (models sometimes add them)."""
    if not text:
        return None
    line = text.strip().splitlines()[0].strip().strip('"').strip("'").strip()
    line = re.sub(r"^(standalone )?(search )?query\s*:\s*", "", line, flags=re.I).strip().strip('"').strip("'").strip()
    return line if 0 < len(line) <= MAX_QUERY_CHARS else None
