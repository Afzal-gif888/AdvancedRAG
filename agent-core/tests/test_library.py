"""Knowledge Base scope: which of the user's ready documents a question searches, and the
resulting retrieval decision. Pure functions, no I/O."""
import uuid

import pytest

from agent_core.classification import RequestType, classify
from agent_core.library import Candidate, select
from agent_core.schemas import AgentRequest

REVIEWS, REVIEWS_OLD, SALES, AURORA, NOTES = (uuid.uuid4() for _ in range(5))
LIB = [Candidate(REVIEWS, "reviews.csv"), Candidate(REVIEWS_OLD, "reviews.csv"), Candidate(SALES, "sales-2025.csv"),
       Candidate(AURORA, "aurora-x2-datasheet.pdf"), Candidate(NOTES, "notes.txt")]


def req(message, *, attached=(), library=LIB, history=None, pending=0):
    return AgentRequest(
        user_id=uuid.uuid4(), conversation_id=uuid.uuid4(), message_id=uuid.uuid4(), message=message,
        document_ids=list(attached), history=history or [], context={"pending_document_count": pending},
        library_documents=[{"document_id": c.document_id, "filename": c.filename, "recent": c.recent} for c in library],
    )


@pytest.mark.parametrize("question,expected", [
    ("What does sales-2025.csv say about Q3 revenue?", [SALES]),          # named
    ("summarize the aurora x2 datasheet", [AURORA]),                       # named (stem with spaces)
    ("what does the pdf say about the warranty?", [AURORA]),               # typed, one candidate
    ("compare all my csvs", [REVIEWS, REVIEWS_OLD, SALES]),                # typed: every CSV
    ("what is the refund window in my documents?", [REVIEWS, REVIEWS_OLD, SALES, AURORA, NOTES]),  # generic: all
])
def test_references_select_only_the_relevant_library_documents(question, expected):
    assert list(select(question, LIB).document_ids) == expected


@pytest.mark.parametrize("question", [
    "What is the capital of France?", "What is a PDF?", "How do I clean the data in pandas?",
    "Write a haiku about autumn", "Explain recursion with an example",
])
def test_general_questions_select_nothing(question):
    s = select(question, LIB)
    assert s.kind == "none" and not s.document_ids and not s.referenced


def test_singular_type_reference_searches_every_file_of_that_type_without_asking():
    s = select("Analyze the CSV I uploaded", LIB)
    assert list(s.document_ids) == [REVIEWS, REVIEWS_OLD, SALES] and s.referenced


# Regression (2026-10-09): "my Knowledge Base" with two ready PDFs answered "I found more than one
# matching document ... Which one should I use?" instead of searching both.
INCIDENT = "What is the incident escalation code specified in my Knowledge Base?"
TWO_PDFS = [Candidate(uuid.uuid4(), "AdvancedRAG_Large_Knowledge_Test_100_Pages.pdf"),
            Candidate(uuid.uuid4(), "AdvancedRAG_Quick_Smoke_Test.pdf")]


@pytest.mark.parametrize("question", [
    INCIDENT, "What does my document say about escalation?", "According to the file I uploaded, who approves refunds?",
    "What is the escalation code in the PDF?", "Check my knowledge base for the on-call rota",
])
def test_references_to_the_users_documents_search_all_of_them(question):
    s = select(question, TWO_PDFS)
    assert list(s.document_ids) == [c.document_id for c in TWO_PDFS] and s.referenced
    c = classify(req(question, library=TWO_PDFS))
    assert c.retrieve and list(c.document_ids) == [d.document_id for d in TWO_PDFS] and c.request_type is RequestType.DOCUMENT_QUESTION


def test_only_an_explicitly_named_document_narrows_the_search():
    c = classify(req("What escalation code does AdvancedRAG_Quick_Smoke_Test.pdf specify?", library=TWO_PDFS))
    assert list(c.document_ids) == [TWO_PDFS[1].document_id] and c.scope == "named"


def test_singular_reference_with_one_candidate_needs_no_question():
    only = [Candidate(REVIEWS, "reviews.csv"), Candidate(AURORA, "aurora-x2-datasheet.pdf")]
    assert list(select("Please analyze the CSV I uploaded earlier.", only).document_ids) == [REVIEWS]


def test_reference_to_a_type_the_library_lacks():
    s = select("analyze the spreadsheet", [Candidate(AURORA, "aurora-x2-datasheet.pdf")])
    assert s.referenced and s.missing_type == "csv" and not s.document_ids


def test_short_or_generic_filenames_do_not_match_ordinary_words():
    lib = [Candidate(NOTES, "notes.txt"), Candidate(SALES, "data.csv")]
    assert select("What are good notes for a guitar beginner?", lib).kind == "none"


# ---------------------------------------------------------------- classification

ALL = [REVIEWS, REVIEWS_OLD, SALES, AURORA, NOTES]


def test_every_question_searches_the_whole_knowledge_base_without_attachments():
    c = classify(req("what is the warranty?"))
    assert c.retrieve and list(c.document_ids) == ALL and c.scope == "knowledge_base" and not c.referenced
    # A general-looking question is searched too; grading decides afterwards whether the
    # Knowledge Base is relevant (a general answer) or not (no invented document answer).
    c = classify(req("What is the capital of France?"))
    assert c.retrieve and c.reason == "knowledge_base_search" and list(c.document_ids) == ALL


def test_named_and_typed_references_narrow_the_search():
    c = classify(req("what does the pdf say about the warranty?"))
    assert c.retrieve and list(c.document_ids) == [AURORA] and c.scope == "named" and c.referenced
    attached = uuid.uuid4()  # older clients may still send ids; they stay in scope
    c = classify(req("compare this with sales-2025.csv", attached=[attached]))
    assert list(c.document_ids) == [attached, SALES] and c.scope == "named"


def test_several_documents_in_scope_never_produce_a_document_question():
    for q in ("Analyze the CSV I uploaded", INCIDENT, "what does the document say?"):
        c = classify(req(q))
        assert c.retrieve and len(c.document_ids) > 1 and RequestType.__members__.get("CLARIFY") is None


def test_follow_up_searches_the_knowledge_base_in_any_conversation():
    lib = [Candidate(AURORA, "aurora-x2-datasheet.pdf"), Candidate(SALES, "sales-2025.csv")]
    history = [{"role": "user", "content": "what does the pdf say about the warranty?"}, {"role": "assistant", "content": "Three years [1]."}]
    c = classify(req("And what about its operating temperature?", library=lib, history=history))
    assert c.request_type is RequestType.FOLLOW_UP and c.retrieve and list(c.document_ids) == [AURORA, SALES]
    # A brand-new conversation (no history) searches the same Knowledge Base.
    c = classify(req("What is the operating temperature of the Aurora X2?", library=lib))
    assert c.request_type is RequestType.DOCUMENT_QUESTION and AURORA in c.document_ids


def test_quantitative_question_over_a_csv_uses_table_analysis_with_text_fallback_scope():
    c = classify(req("How many rows does sales-2025.csv have?"))
    assert c.request_type is RequestType.STRUCTURED and list(c.table_ids) == [SALES] and list(c.document_ids) == [SALES]
    c = classify(req("What percentage of reviews are negative?"))  # "reviews" names reviews.csv
    assert c.request_type is RequestType.STRUCTURED and list(c.table_ids) == [REVIEWS, REVIEWS_OLD] and c.scope == "named"
    c = classify(req("What percentage of entries are negative?"))
    assert c.request_type is RequestType.STRUCTURED and list(c.table_ids) == [REVIEWS, REVIEWS_OLD, SALES]
    assert list(c.document_ids) == ALL
    pdf_only = [Candidate(AURORA, "aurora-x2-datasheet.pdf")]
    assert classify(req("How many cores does it have?", library=pdf_only)).request_type is RequestType.DOCUMENT_QUESTION


def test_referenced_but_missing_documents_are_reported_not_searched():
    c = classify(req("analyze the spreadsheet", library=[Candidate(AURORA, "aurora-x2-datasheet.pdf")]))
    assert not c.retrieve and c.reason == "referenced_documents_not_in_knowledge_base" and c.missing_type == "csv"
    c = classify(req("analyze the file I uploaded", library=[]))
    assert not c.retrieve and c.reason == "referenced_documents_not_in_knowledge_base"


def test_existing_no_library_behaviour_is_unchanged():
    assert classify(req("hello", library=[])).reason == "small_talk"
    assert classify(req("explain RAG", library=[])).reason == "no_documents_attached"
    assert classify(req("explain RAG", library=[], pending=2)).reason == "documents_still_processing"


def test_library_list_never_repeats_attached_documents():
    r = req("compare", attached=[SALES])
    assert SALES not in [d.document_id for d in r.library_documents]


def test_a_filename_sharing_a_word_with_the_question_is_not_selected_by_that_word():
    """Documents are found by their passages (retrieval + reranking), not by filename overlap."""
    warranty, pricing = uuid.uuid4(), uuid.uuid4()
    lib = [Candidate(warranty, "warranty-policy.txt"), Candidate(pricing, "pricing.csv")]
    s = select("According to the warranty document I uploaded, is water damage covered?", lib)
    assert list(s.document_ids) == [warranty, pricing]


def test_generic_reference_is_a_knowledge_base_search_not_a_narrowed_one():
    c = classify(req(INCIDENT, library=TWO_PDFS))
    assert c.scope == "knowledge_base" and c.reason == "knowledge_base_search" and c.referenced
