"""Strictly grounded, self-correcting RAG (unit level; RAG Core and the LLM service are fakes,
see conftest.py). Real-service behaviour: tests/contract and the live run in the delivery report."""
import asyncio
import json

import pytest

from conftest import DOC_A, DOC_B, body, chunk, parse

ABSTAIN = "I couldn't find sufficient evidence for this answer in your Knowledge Base."


def grade(verdict, relevant=(), qtype="document", conflict=False):
    return json.dumps({"verdict": verdict, "relevant": list(relevant), "conflict": conflict, "question_type": qtype})


SUPPORTED = json.dumps({"supported": True, "unsupported": [], "revised": ""})


async def stream(client, **kw):
    r = await client.post("/api/v1/agent/stream", json=body(**kw))
    assert r.status_code == 200
    return parse(r.text)


def final_text(events):
    text = "".join(d["delta"] for n, d in events if n == "content.delta")
    for n, d in events:
        if n == "content.replace":
            text = d["content"]
    return text


# ---------------------------------------------------------------- grading + bounded correction

async def test_weak_retrieval_is_corrected_once_then_abstains_without_generation(client, fakes):
    fakes.checks = {"EVIDENCE GRADER": grade("insufficient"), "SEARCH QUERY CORRECTION": "dental insurance coverage plan"}
    events = await stream(client, message="What does our dental insurance cover?")
    assert [c["query"] for c in fakes.retrieve_calls] == ["What does our dental insurance cover?", "dental insurance coverage plan"]
    assert final_text(events) == ABSTAIN and fakes.stream_calls == []  # nothing for the model to invent from
    done = events[-1][1]
    assert done["citations"] == [] and done["metadata"]["verification_status"] == "insufficient_evidence"
    assert done["metadata"]["correction_attempts"] == 1 and done["metadata"]["grades"] == ["insufficient", "insufficient"]


async def test_correction_finds_evidence_and_only_relevant_passages_are_used(client, fakes):
    fakes.rag_chunks = [chunk(1, text="Shipping is free over 50 EUR."), chunk(2, doc=DOC_B, filename="hr.pdf", text="Dental: two cleanings per year.")]
    fakes.checks = {"EVIDENCE GRADER": [grade("insufficient"), grade("sufficient", [2])],
                    "SEARCH QUERY CORRECTION": "dental benefits", "ANSWER VERIFIER": SUPPORTED}
    fakes.answer = "Two cleanings per year are covered [1]."
    events = await stream(client, message="What does our dental insurance cover?", document_ids=[DOC_A, DOC_B])
    user = fakes.stream_calls[0]["messages"][-1]["content"]
    # Only the relevant passage reaches the model, renumbered as source 1.
    assert '<source id="1" document="hr.pdf"' in user and "Shipping is free" not in user and 'id="2"' not in user
    done = events[-1][1]
    assert [(c["source_id"], c["filename"], c["chunk_id"]) for c in done["citations"]] == [(1, "hr.pdf", f"{DOC_B}:1")]
    assert done["metadata"]["verification_status"] == "verified" and done["metadata"]["evidence_grade"] == "sufficient"


async def test_correction_is_bounded(client, fakes, monkeypatch):
    from agent_core.config import settings

    monkeypatch.setattr(settings, "MAX_CORRECTION_ATTEMPTS", 2)
    fakes.checks = {"EVIDENCE GRADER": grade("insufficient"), "SEARCH QUERY CORRECTION": ["second query", "third query", "fourth"]}
    await stream(client, message="What does the contract say about penalties?")
    assert len(fakes.retrieve_calls) == 3  # first search + at most 2 corrections


async def test_general_question_with_irrelevant_knowledge_base_gets_a_general_answer(client, fakes):
    fakes.checks = {"EVIDENCE GRADER": grade("insufficient", qtype="general")}
    fakes.answer = "Python is a programming language."
    events = await stream(client, message="What is Python?")
    assert len(fakes.retrieve_calls) == 1  # no correction for general knowledge
    msgs = fakes.stream_calls[0]["messages"]
    assert "<evidence>" not in msgs[-1]["content"] and "Refunds" not in msgs[-1]["content"]
    assert final_text(events) == fakes.answer and events[-1][1]["citations"] == []


async def test_explicit_document_reference_without_evidence_abstains(client, fakes):
    fakes.rag_mode = "empty"  # grader unavailable (fake replies non-JSON): still no invented answer
    lib = [{"document_id": DOC_A, "filename": "policy.pdf"}]
    events = await stream(client, message="What does policy.pdf say about pets?", document_ids=[], library_documents=lib)
    assert final_text(events) == ABSTAIN and fakes.stream_calls == []


async def test_partial_and_conflicting_evidence_add_prompt_rules(client, fakes):
    fakes.rag_chunks = [chunk(1), chunk(2, doc=DOC_B, text="Refunds are processed within ten business days.")]
    fakes.checks = {"EVIDENCE GRADER": grade("partial", [1, 2], conflict=True), "ANSWER VERIFIER": SUPPORTED}
    await stream(client, message="How long do refunds take and who approves them?", document_ids=[DOC_A, DOC_B])
    system = fakes.stream_calls[0]["messages"][0]["content"]
    assert "cover only part of this question" in system and "give different answers" in system and "Never ask the user to choose a document" in system


async def test_grader_failure_falls_back_to_grounding_over_all_passages(client, fakes):
    fakes.answer = "Refunds take five business days [1]."
    events = await stream(client)  # grader + verifier reply non-JSON
    assert '<source id="1"' in fakes.stream_calls[0]["messages"][-1]["content"]
    assert events[-1][1]["metadata"]["verification_status"] == "unverified" and events[-1][1]["metadata"]["evidence_grade"] == "unknown"


# ---------------------------------------------------------------- answer verification

async def test_unsupported_claims_are_removed_and_the_revision_replaces_the_draft(client, fakes):
    fakes.answer = "Refunds take five business days [1] and are approved by the CFO [1]."
    revised = "Refunds take five business days [1]. The Knowledge Base does not say who approves refunds."
    fakes.checks = {"EVIDENCE GRADER": grade("sufficient", [1]),
                    "ANSWER VERIFIER": json.dumps({"supported": False, "unsupported": ["approved by the CFO"], "revised": revised})}
    events = await stream(client, message="How long do refunds take and who approves them?")
    assert ("content.replace", {"content": revised}) in events and final_text(events) == revised
    done = events[-1][1]
    assert done["metadata"]["verification_status"] == "revised" and len(done["citations"]) == 1


@pytest.mark.parametrize("verdict", [
    {"supported": False, "unsupported": ["everything"], "revised": ""},
    {"supported": False, "unsupported": ["x"], "revised": "Refunds are fast."},   # revision with no citation
])
async def test_answer_with_no_supported_claim_becomes_the_abstain_answer(client, fakes, verdict):
    fakes.checks = {"EVIDENCE GRADER": grade("sufficient", [1]), "ANSWER VERIFIER": json.dumps(verdict)}
    events = await stream(client)
    assert final_text(events) == ABSTAIN
    assert events[-1][1]["citations"] == [] and events[-1][1]["metadata"]["verification_status"] == "insufficient_evidence"


async def test_model_abstention_is_normalised_to_the_exact_text(client, fakes):
    fakes.checks = {"EVIDENCE GRADER": grade("sufficient", [1])}
    fakes.answer = "I couldn't find sufficient evidence for this answer in your Knowledge Base"
    events = await stream(client)
    assert final_text(events) == ABSTAIN and events[-1][1]["metadata"]["verification_status"] == "insufficient_evidence"


async def test_check_calls_are_bounded_by_a_timeout(monkeypatch):
    from agent_core import grading
    from agent_core.config import settings

    class Slow:
        async def generate(self, *a, **k):
            await asyncio.sleep(5)

    monkeypatch.setattr(settings, "CHECK_TIMEOUT_SECONDS", 0.05)
    g = await grading.grade(Slow(), settings, "q", "passages", 1)
    v = await grading.verify(Slow(), settings, "q", "a [1]", "passages")
    assert g.verdict == "unknown" and v.status == "unverified"


def test_grader_output_is_validated():
    from agent_core.grading import parse_json

    assert parse_json('```json\n{"verdict": "sufficient"}\n```') == {"verdict": "sufficient"}
    assert parse_json("no json") is None and parse_json("[1, 2]") is None


# ---------------------------------------------------------------- citation validation

def test_invalid_citation_markers_are_stripped():
    from agent_core.citations import strip_invalid_markers

    assert strip_invalid_markers("A [1]. B [7]. C [1, 9].", {1}) == "A [1]. B. C [1]."
    assert strip_invalid_markers("Use array[0] and [2].", {1, 2}) == "Use array[0] and [2]."  # indexing, not a citation
    assert strip_invalid_markers("All fine [1][2].", {1, 2}) == "All fine [1][2]."


# ---------------------------------------------------------------- exact table analysis

LIB = [{"document_id": DOC_A, "filename": "reviews.csv"}]
TABLE = {"document_id": DOC_A, "filename": "reviews.csv", "row_count": 300,
         "columns": [{"name": "sentiment", "type": "text", "distinct": 3, "missing": 0, "examples": ["positive", "negative", "neutral"]}]}
RESULTS = {"document_id": DOC_A, "filename": "reviews.csv", "results": [
    {"operation": "distribution", "rows_total": 300, "rows_matched": 300, "column": "sentiment",
     "result": {"distinct": 3, "values": [{"value": "positive", "count": 150, "percent": 50.0},
                                          {"value": "negative", "count": 90, "percent": 30.0},
                                          {"value": "neutral", "count": 60, "percent": 20.0}], "other_count": 0}}]}
PLAN = json.dumps({"document_id": DOC_A, "queries": [{"operation": "distribution", "column": "sentiment"}]})


async def test_quantitative_question_is_computed_over_the_whole_table(client, fakes):
    fakes.tables, fakes.table_results = [TABLE], RESULTS
    fakes.checks = {"TABLE QUERY PLANNER": PLAN}
    fakes.answer = "30% of the 300 reviews are negative (90 reviews) [1]."
    events = await stream(client, message="What percentage of entries are negative?", document_ids=[], library_documents=LIB)
    assert fakes.retrieve_calls == []  # no chunk retrieval: the figures come from the full table
    assert fakes.table_calls[1] == ("query", {"user_id": body()["user_id"], "document_id": DOC_A,
                                              "queries": [{"operation": "distribution", "column": "sentiment", "by": None, "filters": [], "limit": 20}]})
    user = fakes.stream_calls[0]["messages"][-1]["content"]
    assert "Computed exactly over the full table reviews.csv" in user and "negative: 90 (30.0%)" in user
    done = events[-1][1]
    assert final_text(events) == fakes.answer and done["metadata"]["verification_status"] == "computed"
    assert done["citations"][0]["chunk_id"] == f"{DOC_A}:table" and done["citations"][0]["page"] is None


async def test_invented_figures_are_replaced_by_the_computed_results(client, fakes):
    fakes.tables, fakes.table_results = [TABLE], RESULTS
    fakes.checks = {"TABLE QUERY PLANNER": PLAN, "ANSWER VERIFIER": json.dumps({"supported": False, "unsupported": ["35%"], "revised": ""})}
    fakes.answer = "About 35% of reviews are negative [1]."
    events = await stream(client, message="What percentage of entries are negative?", document_ids=[], library_documents=LIB)
    text = final_text(events)
    assert "35" not in text and "negative: 90 (30.0%)" in text and text.endswith("[1]")


async def test_rejected_plan_is_retried_once_then_falls_back_to_search(client, fakes):
    fakes.tables, fakes.table_results = [TABLE], None  # RAG Core rejects every spec
    fakes.checks = {"TABLE QUERY PLANNER": PLAN, "EVIDENCE GRADER": grade("sufficient", [1]), "ANSWER VERIFIER": SUPPORTED}
    fakes.answer = "Refunds take five business days [1]."
    events = await stream(client, message="What percentage of entries are negative?", document_ids=[], library_documents=LIB)
    assert [name for name, _ in fakes.table_calls] == ["describe", "query", "query"]
    assert len(fakes.retrieve_calls) == 1 and events[-1][1]["metadata"]["table_analysis"] == "no_plan"


async def test_planner_cannot_point_at_a_table_it_was_not_given(client, fakes):
    fakes.tables, fakes.table_results = [TABLE], RESULTS
    fakes.checks = {"TABLE QUERY PLANNER": json.dumps({"document_id": DOC_B, "queries": [{"operation": "count"}]})}
    await stream(client, message="How many entries are there?", document_ids=[], library_documents=LIB)
    assert [name for name, _ in fakes.table_calls] == ["describe"]  # never queried


def test_numbers_check():
    from agent_core.structured import numbers_supported, render

    text = render("reviews.csv", RESULTS["results"])
    assert numbers_supported("30% (90 of 300) are negative [1].", text)
    assert numbers_supported("30.0% are negative [1]", text)
    assert not numbers_supported("31% are negative [1].", text)


def test_filtered_results_state_which_rows_were_counted():
    """Live regression (2026-10-09): "500 of 2000 rows" without the filter made the model abstain on
    "what share of German reviews are negative?" although the exact answer had been computed."""
    from agent_core.structured import render

    res = [{"operation": "distribution", "rows_total": 2000, "rows_matched": 500, "column": "sentiment",
            "result": {"distinct": 2, "values": [{"value": "negative", "count": 143, "percent": 28.6},
                                                 {"value": "positive", "count": 357, "percent": 71.4}], "other_count": 0}}]
    text = render("reviews.csv", res, [{"operation": "distribution", "column": "sentiment",
                                        "filters": [{"column": "language", "op": "eq", "value": "de"}]}])
    assert 'rows where language = "de": 500 of 2000 rows' in text and "negative: 143 (28.6%)" in text


# ---------------------------------------------------------------- general questions when checks misbehave

async def test_malformed_grader_output_is_retried_once(client, fakes):
    fakes.checks = {"EVIDENCE GRADER": ["not json at all", grade("sufficient", [1])], "ANSWER VERIFIER": SUPPORTED}
    fakes.answer = "Refunds take five business days [1]."
    done = (await stream(client))[-1][1]
    assert done["metadata"]["evidence_grade"] == "sufficient" and done["metadata"]["verification_status"] == "verified"
    assert sum("EVIDENCE GRADER" in c["messages"][0]["content"] for c in fakes.generate_calls) == 2


async def test_general_question_is_answered_when_grading_is_unavailable(client, fakes):
    """Grader output unusable twice: a general question gets a general answer, not the abstain sentence."""
    fakes.answer = "A hash table maps keys to values using a hash function."
    events = await stream(client, message="In one sentence, what is a hash table?")
    system = fakes.stream_calls[0]["messages"][0]["content"]
    assert "It is not known whether these sources are relevant" in system
    done = events[-1][1]
    assert final_text(events) == fakes.answer and done["citations"] == []
    assert done["metadata"]["verification_status"] == "no_relevant_documents"


async def test_document_reference_with_grading_unavailable_stays_strict(client, fakes):
    fakes.answer = "The policy says something."
    lib = [{"document_id": DOC_A, "filename": "policy.pdf"}]
    await stream(client, message="What does policy.pdf say about pets?", document_ids=[], library_documents=lib)
    assert "It is not known whether" not in fakes.stream_calls[0]["messages"][0]["content"]
