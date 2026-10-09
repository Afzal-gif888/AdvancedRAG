"""Follow-ups on earlier answers of the same conversation (unit level; RAG Core and the LLM service
are fakes, see conftest.py). Real-stack behaviour: the live multi-turn run in the delivery report."""
import json

import pytest

from conftest import DOC_A, DOC_B, body, chunk, parse

ABSTAIN = "I couldn't find sufficient evidence for this answer in your Knowledge Base."
DIST = {"document_id": DOC_A, "filename": "AdvancedRAG_CSV_Exact_Stats_5000_Rows.csv", "queries": [{"operation": "distribution", "column": "sentiment", "filters": []}],
        "results": [{"operation": "distribution", "rows_total": 5000, "rows_matched": 5000, "column": "sentiment",
                     "result": {"distinct": 3, "other_count": 0, "values": [
                         {"value": "positive", "count": 2500, "percent": 50.0},
                         {"value": "negative", "count": 1500, "percent": 30.0},
                         {"value": "neutral", "count": 1000, "percent": 20.0}]}}]}
SOUTH = {**DIST, "queries": [{"operation": "distribution", "column": "sentiment", "filters": [{"column": "region", "op": "eq", "value": "South"}]}],
         "results": [{**DIST["results"][0], "rows_matched": 1250, "result": {"distinct": 3, "other_count": 0, "values": [
             {"value": "positive", "count": 625, "percent": 50.0}, {"value": "negative", "count": 400, "percent": 32.0},
             {"value": "neutral", "count": 225, "percent": 18.0}]}}]}
TABLE_CITE = {"source_id": 1, "document_id": DOC_A, "filename": DIST["filename"], "page": None, "chunk_id": f"{DOC_A}:table",
              "snippet": "Computed exactly over the full table"}
DIST_ANSWER = "Positive: 2500 (50.0%), Negative: 1500 (30.0%), Neutral: 1000 (20.0%). Total rows: 5000 [1]."
EXPECTED_TABLE = ("| Sentiment | Count | Percentage |\n|---|---:|---:|\n| Positive | 2500 | 50.0% |\n| Negative | 1500 | 30.0% |\n"
                  "| Neutral | 1000 | 20.0% |\n| Total | 5000 | 100.0% |")


def turn(q, a, cites=(), structured=None):
    return [{"role": "user", "content": q}, {"role": "assistant", "content": a, "citations": list(cites), "structured": structured}]


CSV_HISTORY = turn("What is the sentiment distribution in the CSV?", DIST_ANSWER, [TABLE_CITE], DIST)


async def stream(client, **kw):
    r = await client.post("/api/v1/agent/stream", json=body(**kw))
    assert r.status_code == 200, r.text
    return parse(r.text)


def final(events):
    text = "".join(d["delta"] for n, d in events if n == "content.delta")
    for n, d in events:
        if n == "content.replace":
            text = d["content"]
    return text


# ---------------------------------------------------------------- detection

@pytest.mark.parametrize("q", [
    "Give the above in table format.", "give above in table format", "Convert that into a table.",
    "Summarize your previous answer.", "Explain the previous result.", "Show those numbers as percentages.",
    "Use the results you just provided.", "Can you make that shorter?",
    "give the above information that you provided recently in table format",
])
def test_follow_up_requests_are_recognised(q):
    from agent_core.followup import is_followup

    assert is_followup(q)


@pytest.mark.parametrize("q", [
    "What is the negative sentiment percentage in the South region?", "Compare negative sentiment across all regions.",
    "What is the sentiment distribution in my Knowledge Base?", "How long is the warranty?",
    "Explain how the Orchid-7 descaling programme works and which parts of the machine it cleans.",
])
def test_new_questions_are_not_follow_ups(q):
    from agent_core.followup import is_followup

    assert not is_followup(q)


# ---------------------------------------------------------------- reformat a computed result

@pytest.mark.parametrize("q", ["give above in table format", "give the above information that you provided recently in table format"])
async def test_table_of_the_previous_csv_result_is_deterministic_exact_and_cited(client, fakes, q):
    events = await stream(client, message=q, history=CSV_HISTORY)
    text = final(events)
    assert EXPECTED_TABLE in text
    assert fakes.retrieve_calls == [] and fakes.table_calls == [] and fakes.generate_calls == [] and fakes.stream_calls == []
    assert "retrieval.started" not in [n for n, _ in events]
    done = events[-1][1]
    assert [c["chunk_id"] for c in done["citations"]] == [f"{DOC_A}:table"]  # the original computation, not a new retrieval
    meta = done["metadata"]
    assert meta["verification_status"] == "reformatted" and meta["followup"] == "deterministic_table"
    assert meta["structured_result"] == DIST  # kept, so later turns can refer to it again


async def test_shorter_summary_reuses_only_previous_figures(client, fakes):
    fakes.checks = {"YOUR EARLIER ANSWERS": "50% positive (2500), 30% negative (1500), 20% neutral (1000) [1].",
                    "ANSWER VERIFIER": json.dumps({"supported": True, "unsupported": [], "revised": ""})}
    events = await stream(client, message="Can you make that shorter?", history=CSV_HISTORY)
    assert final(events).startswith("50% positive") and fakes.retrieve_calls == [] and fakes.stream_calls == []
    assert [c["chunk_id"] for c in events[-1][1]["citations"]] == [f"{DOC_A}:table"]
    sent = next(c for c in fakes.generate_calls if "YOUR EARLIER ANSWERS" in c["messages"][0]["content"])
    assert "2500" in sent["messages"][-1]["content"] and "Request: Can you make that shorter?" in sent["messages"][-1]["content"]


async def test_different_table_layout_goes_through_the_model_with_the_numbers_check(client, fakes):
    fakes.checks = {"YOUR EARLIER ANSWERS": "| Sentiment | % |\n|---|---|\n| Positive | 50.0% |\n| Negative | 30.0% |\n| Neutral | 20.0% | [1]"}
    events = await stream(client, message="Show those numbers as percentages only, in a two-column table", history=CSV_HISTORY)
    assert "| Positive | 50.0% |" in final(events) and fakes.retrieve_calls == []


async def test_a_reformat_that_invents_a_figure_is_rejected_and_the_normal_path_runs(client, fakes):
    fakes.checks = {"YOUR EARLIER ANSWERS": "Positive 2500, negative 1500, neutral 1000, average rating 4.2 [1]."}
    events = await stream(client, message="Summarize your previous answer", history=CSV_HISTORY)
    assert "4.2" not in final(events)
    assert events[-1][1]["metadata"]["followup"] == "new_numbers_rejected" and fakes.retrieve_calls  # fell back to retrieval


async def test_a_request_needing_new_information_falls_back_to_analysis(client, fakes):
    fakes.checks = {"YOUR EARLIER ANSWERS": "NEEDS_NEW_ANALYSIS"}
    await stream(client, message="Use the results you just provided to find the busiest region", history=CSV_HISTORY)
    assert fakes.retrieve_calls or fakes.table_calls


async def test_a_result_several_messages_earlier_is_found(client, fakes):
    history = [*CSV_HISTORY,
               *turn("How long is the warranty?", "26 months [1].", [{"source_id": 1, "document_id": DOC_A, "filename": "w.pdf", "chunk_id": f"{DOC_A}:3", "snippet": "26 months"}]),
               *turn("And in the South region?", "South: 32.0% negative [1].", [TABLE_CITE], SOUTH)]
    events = await stream(client, message="Put the overall sentiment distribution from earlier in a table", history=history)
    assert EXPECTED_TABLE in final(events) and fakes.retrieve_calls == []


async def test_most_recent_filtered_result_is_tabled_with_its_filter(client, fakes):
    history = [*CSV_HISTORY, *turn("And in the South region?", "South: 32.0% negative [1].", [TABLE_CITE], SOUTH)]
    text = final(await stream(client, message="give above in table format", history=history))
    assert 'Rows where region = "South": 1250 of 5000' in text and "| Negative | 400 | 32.0% |" in text


# ---------------------------------------------------------------- document-grounded answers

DOC_HISTORY = turn("How long is the Orchid-7 warranty?", "The Orchid-7 has a 26-month warranty [1]; water damage is not covered [2].",
                   [{"source_id": 1, "document_id": DOC_A, "filename": "orchid.pdf", "page": 2, "chunk_id": f"{DOC_A}:1", "snippet": "26-month warranty"},
                    {"source_id": 2, "document_id": DOC_A, "filename": "orchid.pdf", "page": 3, "chunk_id": f"{DOC_A}:2", "snippet": "Water damage is not covered"}])


async def test_reformatting_a_document_answer_keeps_its_citations(client, fakes):
    fakes.checks = {"YOUR EARLIER ANSWERS": "- Warranty: 26 months [1]\n- Water damage: not covered [2]",
                    "ANSWER VERIFIER": json.dumps({"supported": True, "unsupported": [], "revised": ""})}
    events = await stream(client, message="Convert that into a bullet list", history=DOC_HISTORY)
    cites = events[-1][1]["citations"]
    assert [(c["chunk_id"], c["page"]) for c in cites] == [(f"{DOC_A}:1", 2), (f"{DOC_A}:2", 3)] and fakes.retrieve_calls == []


async def test_invented_citation_numbers_are_removed(client, fakes):
    fakes.checks = {"YOUR EARLIER ANSWERS": "Warranty: 26 months [1] [9]."}
    events = await stream(client, message="Summarize your previous answer", history=DOC_HISTORY)
    assert "[9]" not in final(events) and [c["chunk_id"] for c in events[-1][1]["citations"]] == [f"{DOC_A}:1"]


async def test_answer_citing_a_document_no_longer_in_the_knowledge_base_is_not_reused(client, fakes):
    fakes.checks = {"YOUR EARLIER ANSWERS": "26 months [1]."}
    fakes.rag_chunks = [chunk(1, doc=DOC_B, filename="other.pdf", text="Unrelated text.")]
    events = await stream(client, message="Summarize your previous answer", history=DOC_HISTORY, document_ids=[DOC_B])
    assert events[-1][1]["metadata"].get("followup") == "no_reusable_answer"
    assert not any("YOUR EARLIER ANSWERS" in c["messages"][0]["content"] for c in fakes.generate_calls)


async def test_an_earlier_abstention_is_never_reused_as_evidence(client, fakes):
    history = turn("What is the RTO?", ABSTAIN)
    events = await stream(client, message="Summarize your previous answer", history=history)
    assert events[-1][1]["metadata"].get("followup") == "no_reusable_answer"
    assert not any("YOUR EARLIER ANSWERS" in c["messages"][0]["content"] for c in fakes.generate_calls)


# ---------------------------------------------------------------- new analysis / isolation

async def test_a_new_csv_question_after_a_result_runs_a_fresh_calculation(client, fakes):
    fakes.tables = [{"document_id": DOC_A, "filename": DIST["filename"], "row_count": 5000,
                     "columns": [{"name": "sentiment", "type": "text"}, {"name": "region", "type": "text"}]}]
    fakes.table_results = {"document_id": DOC_A, "filename": DIST["filename"], "results": SOUTH["results"]}
    fakes.checks = {"TABLE QUERY PLANNER": json.dumps({"document_id": DOC_A, "queries": [SOUTH["queries"][0]]})}
    fakes.answer = "32.0% of South reviews are negative [1]."
    events = await stream(client, message="What is the negative sentiment percentage in the South region?", history=CSV_HISTORY,
                          library_documents=[{"document_id": DOC_A, "filename": DIST["filename"]}], document_ids=[])
    assert [name for name, _ in fakes.table_calls] == ["describe", "query"]
    assert events[-1][1]["metadata"]["verification_status"] == "computed"


@pytest.mark.parametrize("history", [[], turn("What is the RTO?", ABSTAIN)])
async def test_nothing_to_reformat_gets_a_targeted_question_not_a_search(client, fakes, history):
    """A new conversation (or one with only an abstention) has nothing above to reuse: no retrieval
    over raw data, no access to other conversations, just a question back."""
    events = await stream(client, message="give above in table format", history=history)
    assert final(events).startswith("There's no earlier answer in this conversation")
    assert fakes.retrieve_calls == [] and fakes.stream_calls == [] and events[-1][1]["citations"] == []


async def test_a_new_question_mentioning_previous_is_not_a_follow_up(client, fakes):
    fakes.checks = {"EVIDENCE GRADER": json.dumps({"verdict": "sufficient", "relevant": [1], "conflict": False, "question_type": "document"})}
    fakes.answer = "Refunds take five business days [1]."
    events = await stream(client, message="Who was the previous refunds manager according to my documents?", history=[],
                          library_documents=[{"document_id": DOC_A, "filename": "policy.pdf"}])
    assert fakes.retrieve_calls and events[-1][1]["metadata"].get("followup") is None


def test_deterministic_table_only_for_a_plain_table_request():
    from agent_core import followup
    from agent_core.schemas import HistoryMessage

    prior = followup.candidates([HistoryMessage.model_validate(m) for m in CSV_HISTORY], {__import__("uuid").UUID(DOC_A)})[-1]
    assert followup.deterministic_table_requested("give above in table format", prior)
    assert followup.deterministic_table_requested("Put the overall sentiment distribution from earlier in a table", prior)
    assert not followup.deterministic_table_requested("Show those numbers as percentages only, in a two-column table", prior)
    assert not followup.deterministic_table_requested("Make the above shorter", prior)
