"""Regression tests: follow-up resolution must keep the conversation's subject.

Reproduced on the real stack (2026-10-08): for
    Turn 1  "How long is the warranty on the Aurora X2?"
    Turn 2  "And what about its operating temperature?"
FreeLLMAPI routed the rewrite to a reasoning-style model that returned its reasoning, truncated at
max_tokens (finish_reason="length"). That paragraph was accepted as the search query
(rewritten=True), retrieval was skewed towards warranty text, and the answer was not cited.
"""
import pytest

from agent_core.config import Settings
from agent_core.errors import llm_error
from agent_core.query_resolution import extract_entities, rejection_reason, resolve_search_query, subject_entities
from agent_core.schemas import GenerateResult
from conftest import body

CFG = Settings(_env_file=None, ENVIRONMENT="test")

AURORA = [{"role": "user", "content": "How long is the warranty on the Aurora X2?"},
          {"role": "assistant", "content": "The Aurora X2 warranty covers manufacturing defects for 24 months from delivery."}]
FOLLOW_UP = "And what about its operating temperature?"
EXPECTED = "What is the operating temperature of the Aurora X2?"

# Verbatim output captured from cohere/north-mini-code:free (finish_reason="length").
REASONING_LEAK = (
    "We need to rewrite the user's latest message as a standalone search query for a document search engine, resolving "
    "pronouns and references using the conversation. The conversation: user asked \"How long is the warranty on the Aurora "
    "X2?\" assistant answered: \"The Aurora X2 warranty covers manufacturing defects for 24 months from delivery.\" Latest "
    "message: \"And what about its operating temperature?\" We need to resolve pronouns: \"its\" refers to Aurora X2. So "
    "query should be something like \"Aurora X2 operating temperature\". Keep names, numbers, key terms. So output: "
    "\"Aurora X2 operating temperature\"."
)


class LLM:
    def __init__(self, content=None, finish_reason="stop", error=None):
        self.content, self.finish_reason, self.error, self.calls = content, finish_reason, error, []

    async def generate(self, messages, **kw):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return GenerateResult(content=self.content, model="m", finish_reason=self.finish_reason)


# ---------------------------------------------------------------- the exact regression

async def test_exact_two_turn_regression_good_rewrite_is_sent_verbatim():
    llm = LLM(EXPECTED)
    out = await resolve_search_query(llm, CFG, FOLLOW_UP, AURORA)
    assert out.method == "rewritten" and out.text == EXPECTED
    sent = llm.calls[0]
    assert "Aurora X2" in sent[-1]["content"] and FOLLOW_UP in sent[-1]["content"]  # history reached the rewriter
    assert "do not switch to a different item" in sent[0]["content"]


async def test_exact_two_turn_regression_reasoning_leak_is_rejected_and_subject_kept():
    out = await resolve_search_query(LLM(REASONING_LEAK, finish_reason="length"), CFG, FOLLOW_UP, AURORA)
    assert out.method == "fallback_with_subject" and out.rejected_reason == "truncated"
    assert out.text == "Aurora X2: And what about its operating temperature?"
    assert "warranty" not in out.text  # the previous topic no longer skews retrieval


async def test_reasoning_leak_is_rejected_even_when_not_truncated():
    out = await resolve_search_query(LLM(REASONING_LEAK, finish_reason="stop"), CFG, FOLLOW_UP, AURORA)
    assert out.method == "fallback_with_subject" and out.rejected_reason in ("too_long", "meta_text")


@pytest.mark.parametrize("bad,reason", [
    ("What is the operating temperature of the Aurora X1?", "introduced_entity"),   # switched product
    ("What is the operating temperature?", "dropped_subject"),                      # lost the product
    ("operating temperature of the device", "dropped_subject"),
    ("Here is the standalone query: Aurora X2 temperature", "meta_text"),
])
async def test_rewrites_that_change_or_lose_the_entity_are_rejected(bad, reason):
    out = await resolve_search_query(LLM(bad), CFG, FOLLOW_UP, AURORA)
    assert out.rejected_reason == reason
    assert out.text == "Aurora X2: And what about its operating temperature?"


@pytest.mark.parametrize("good", [
    "What is the operating temperature of the Aurora X2?",
    "What is the operating temperature range for the Aurora X2?",   # real ling-3.0-flash output
    "aurora x2 operating temperature",
    '"Aurora X2 operating temperature"',
    "Query: Aurora X2 operating temperature",
])
async def test_valid_rewrites_are_accepted(good):
    out = await resolve_search_query(LLM(good), CFG, FOLLOW_UP, AURORA)
    assert out.method == "rewritten" and "aurora x2" in out.text.lower() and "temperature" in out.text.lower()
    assert not out.text.startswith(("Query", '"'))


# ---------------------------------------------------------------- similar pronoun / entity follow-ups

CASES = [
    # (history, follow-up, good rewrite, rewrite that loses/changes the subject, expected subject)
    ([{"role": "user", "content": "Tell me about the Halo Display."}], "Does it support Bluetooth?",
     "Does the Halo Display support Bluetooth?", "Does it support Bluetooth?", "Halo Display"),
    ([{"role": "user", "content": "What is the price of the Zephyr Relay in Germany?"}], "And in France?",
     "What is the price of the Zephyr Relay in France?", "What is the price of the Nimbus Gateway in France?", "Zephyr Relay"),
    ([{"role": "user", "content": "Which product has the SKU ZRy-SL18?"},
      {"role": "assistant", "content": "SKU ZRy-SL18 is the Zephyr Relay sold in Slovenia."}], "How much does that one cost?",
     "How much does SKU ZRy-SL18 cost?", "How much does it cost?", "SKU ZRy-SL18"),
    ([{"role": "user", "content": "Compare the Aurora X1 and the Aurora X2."}], "Which of them is lighter?",
     "Which is lighter, the Aurora X1 or the Aurora X2?", "Which sensor is lighter?", "Aurora X1"),
    ([{"role": "user", "content": "What does the Employee Handbook say about parental leave?"}], "Is this paid?",
     "Is parental leave in the Employee Handbook paid?", "Is the Code of Conduct paid?", "Employee Handbook"),
]


@pytest.mark.parametrize("history,message,good,bad,subject", CASES)
async def test_pronoun_follow_ups_keep_the_subject(history, message, good, bad, subject):
    assert subject in subject_entities(message, history)
    ok = await resolve_search_query(LLM(good), CFG, message, history)
    assert ok.method == "rewritten" and ok.text == good
    rejected = await resolve_search_query(LLM(bad), CFG, message, history)
    assert rejected.method == "fallback_with_subject" and subject in rejected.text and message in rejected.text


async def test_follow_up_naming_a_sibling_item_may_switch_subject():
    out = await resolve_search_query(LLM("What is the operating temperature of the Aurora X1?"), CFG,
                                     "And what about the Aurora X1?", AURORA)
    assert out.method == "rewritten" and "Aurora X1" in out.text


async def test_named_attribute_does_not_replace_the_subject():
    history = [{"role": "user", "content": "What is the price of the Zephyr Relay in Germany?"}]
    out = await resolve_search_query(LLM("What is the price in France?"), CFG, "And in France?", history)
    assert out.rejected_reason == "dropped_subject" and out.text == "Zephyr Relay, Germany: And in France?"


async def test_self_contained_new_question_is_not_forced_onto_the_old_subject():
    out = await resolve_search_query(LLM("What is the Halo Display refresh rate?"), CFG,
                                     "What is the refresh rate of the Halo Display?", AURORA)
    assert out.method == "rewritten" and out.text == "What is the Halo Display refresh rate?"


# ---------------------------------------------------------------- failed rewrite never changes the subject

@pytest.mark.parametrize("error", [llm_error("LLM_TIMEOUT"), llm_error("LLM_PROVIDER_UNAVAILABLE"), llm_error("LLM_INVALID_RESPONSE")])
async def test_failed_rewrite_keeps_the_subject(error):
    out = await resolve_search_query(LLM(error=error), CFG, FOLLOW_UP, AURORA)
    assert out.method == "fallback_with_subject" and out.text == "Aurora X2: And what about its operating temperature?"


async def test_empty_rewrite_keeps_the_subject():
    out = await resolve_search_query(LLM(""), CFG, FOLLOW_UP, AURORA)
    assert out.rejected_reason == "empty" and out.text.startswith("Aurora X2:")


async def test_without_any_named_subject_the_previous_question_is_used():
    history = [{"role": "user", "content": "what is the remote work policy?"}]
    out = await resolve_search_query(LLM(error=llm_error("LLM_TIMEOUT")), CFG, "can I do that from home?", history)
    assert out.method == "fallback_with_previous_question"
    assert out.text == "what is the remote work policy?\ncan I do that from home?"


def test_entity_extraction_ignores_sentence_start_words_and_numbers():
    assert extract_entities("Up to three days per week.") == []
    assert extract_entities("The Aurora X2 warranty covers defects for 24 months.") == ["Aurora X2"]
    assert extract_entities("And what about its operating temperature?") == []
    assert rejection_reason("Aurora X2 temperature", FOLLOW_UP, AURORA) is None


# ---------------------------------------------------------------- through the orchestrator (fakes)

async def test_orchestrator_sends_exactly_the_resolved_query_to_rag(client, fakes):
    fakes.rewrite = EXPECTED
    r = await client.post("/api/v1/agent/stream", json=body(message=FOLLOW_UP, history=AURORA))
    assert r.status_code == 200
    assert fakes.retrieve_calls[0]["query"] == EXPECTED
    # The answer still addresses the user's own words, with the conversation as context.
    msgs = fakes.stream_calls[0]["messages"]
    assert msgs[-1]["content"].endswith(f"Question: {FOLLOW_UP}") and msgs[1]["content"] == AURORA[0]["content"]


async def test_orchestrator_with_leaked_reasoning_retrieves_with_the_subject(client, fakes):
    fakes.rewrite = REASONING_LEAK
    r = await client.post("/api/v1/agent/stream", json=body(message=FOLLOW_UP, history=AURORA))
    assert fakes.retrieve_calls[0]["query"] == "Aurora X2: And what about its operating temperature?"
    done = [line for line in r.text.splitlines() if line.startswith("data:")][-1]
    assert '"query_rewritten": false' in done and '"query_rewrite_rejected": "too_long"' in done
