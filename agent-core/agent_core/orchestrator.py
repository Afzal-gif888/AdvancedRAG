"""One chat turn, end to end (strictly grounded, self-correcting RAG):

    classify
    -> [follow-up] the message refers to an earlier answer of this conversation ("give the above in
                 table format"): reuse that answer / its stored computed results, no retrieval
                 (followup.py); falls back to the normal path when new information is needed
    -> [table]   quantitative question over a CSV: plan -> compute over the whole table -> evidence
    -> [search]  resolve follow-up query -> retrieve over the user's Knowledge Base
                 -> grade evidence (relevant? sufficient? contradictory?)
                 -> bounded correction: a different query, merged results, re-graded
    -> decide    insufficient + document question  -> exact abstain answer, no generation
                 insufficient + general question   -> general answer, no citations
                 sufficient / partial              -> grounded answer over the relevant passages only
    -> stream    the answer, token by token
    -> verify    every claim against the cited passages; strip citation markers that name no
                 supplied source; a revised or rejected answer replaces the draft (content.replace)
    -> citations mapped to the passages actually supplied; safe metadata

Events (the backend's stable contract, in this order):
    agent.started
    retrieval.started            only when retrieval is required
    retrieval.completed          {retrieved_chunks}
    generation.started
    content.delta*               {delta}
    content.replace              {content}   at most once: the final answer replaces the streamed draft
    generation.completed         {citations, metadata}     or, on failure:   error {code, message}

Agent Core is stateless: it persists nothing and owns no conversation data.
"""
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from advancerag_common.observability import METRICS
from agent_core import citations as cit
from agent_core import followup, grading, prompts, structured
from agent_core.classification import Classification, RequestType, classify
from agent_core.clients import LLMClient, RAGClient
from agent_core.config import Settings
from agent_core.context import build_evidence
from agent_core.errors import GENERATION_FAILED, LLM_TIMEOUT, AgentError
from agent_core.query_resolution import normalize_history, resolve_search_query
from agent_core.schemas import AgentRequest, RetrievedChunk

log = logging.getLogger("advancerag.agent.orchestrator")

StreamItem = Tuple[str, Dict[str, Any]]

# How the answer is produced (decided after retrieval and grading)
GROUNDED, TABLE, NO_EVIDENCE, ABSTAIN, GENERAL = "grounded", "table", "no_evidence", "abstain", "general"


@dataclass
class Plan:
    mode: str
    sources: List[RetrievedChunk] = field(default_factory=list)   # exactly what the model sees, numbered 1..n
    evidence: str = ""
    notes: List[str] = field(default_factory=list)
    table_text: str = ""
    general_allowed: bool = False   # grading unavailable: a general question may be answered generally


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 2)


def _renumber(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    return [c.model_copy(update={"source_id": i}) for i, c in enumerate(chunks, start=1)]


def _merge(first: List[RetrievedChunk], second: List[RetrievedChunk]) -> List[RetrievedChunk]:
    seen, out = set(), []
    for c in [*first, *second]:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            out.append(c)
    return _renumber(out)


def is_abstention(text: str) -> bool:
    t = text.strip().strip('"*_ ').lower()
    target = prompts.ABSTAIN_MESSAGE.lower().rstrip(".")
    return t.startswith(target) and len(t) <= len(target) + 20


class Orchestrator:
    def __init__(self, cfg: Settings, rag: RAGClient, llm: LLMClient):
        self.cfg, self.rag, self.llm = cfg, rag, llm

    # ------------------------------------------------------------------ retrieval + self-correction
    async def _search(self, req: AgentRequest, cls: Classification, query: str, meta: Dict[str, Any]) -> List[RetrievedChunk]:
        started = time.perf_counter()
        result = await self.rag.retrieve(user_id=req.user_id, query=query, document_ids=list(cls.document_ids),
                                         top_k=self.cfg.RETRIEVAL_TOP_K)
        meta["rag_call_ms"] = round(meta.get("rag_call_ms", 0) + _ms(started), 2)
        rmeta = result.metadata
        strategy = rmeta.get("retrieval_strategy")
        meta["retrieval_strategy"] = strategy if isinstance(strategy, str) and len(strategy) <= 64 else "unknown"
        for key in ("retrieval_latency_ms", "rerank_latency_ms"):
            if isinstance(rmeta.get(key), (int, float)):
                meta[key] = rmeta[key]
        return result.chunks

    async def _grade(self, question: str, chunks: List[RetrievedChunk]) -> grading.Grade:
        if not self.cfg.EVIDENCE_GRADING_ENABLED:
            return grading.Grade()
        evidence = build_evidence(chunks, self.cfg.MAX_CONTEXT_CHARS)
        text = evidence.text or "(no passages were found)"
        return await grading.grade(self.llm, self.cfg, question, text, len(evidence.sources))

    async def _retrieve_and_grade(self, req: AgentRequest, cls: Classification, history: List[Dict[str, str]],
                                  meta: Dict[str, Any]) -> Plan:
        resolved = await resolve_search_query(self.llm, self.cfg, req.message, history)
        meta["query_rewritten"] = resolved.method == "rewritten"
        meta["query_resolution"] = resolved.method
        meta["query_rewrite_rejected"] = resolved.rejected_reason
        chunks = await self._search(req, cls, resolved.text, meta)
        started = time.perf_counter()
        grade = await self._grade(req.message, chunks)
        meta["grades"] = [grade.verdict]

        # Bounded correction: weak or empty retrieval for a question that is not general knowledge.
        attempts, query = 0, resolved.text
        while (self.cfg.EVIDENCE_GRADING_ENABLED and attempts < self.cfg.MAX_CORRECTION_ATTEMPTS
               and (not chunks or grade.verdict == "insufficient") and grade.question_type != "general"):
            attempts += 1
            better = await grading.corrected_query(self.llm, self.cfg, req.message, query)
            if not better:
                break
            query = better
            more = await self._search(req, cls, query, meta)
            if not more:
                continue
            # New results first (the previous ones were judged insufficient), then the earlier ones.
            chunks = _merge(more, chunks)
            grade = await self._grade(req.message, chunks)
            meta["grades"].append(grade.verdict)
        meta["correction_attempts"] = attempts
        meta["evidence_grade"] = grade.verdict
        meta["question_type"] = grade.question_type
        meta["evidence_conflict"] = grade.conflict
        meta["grading_ms"] = _ms(started)
        meta["candidate_chunks"] = len(chunks)

        if grade.verdict == "unknown":  # grading disabled or failed: all passages
            evidence = build_evidence(chunks, self.cfg.MAX_CONTEXT_CHARS)
            if not evidence.sources:
                return Plan(ABSTAIN if cls.referenced else NO_EVIDENCE)
            if cls.referenced:
                return Plan(GROUNDED, evidence.sources, evidence.text)
            # Not known whether this is about the user's documents: a general question must still
            # get a general answer (not the abstain sentence) when the passages are unrelated.
            return Plan(GROUNDED, evidence.sources, evidence.text, [prompts.OPEN_NOTE], general_allowed=True)
        if grade.verdict == "insufficient":
            if cls.referenced or grade.question_type == "document":
                return Plan(ABSTAIN)
            return Plan(GENERAL if grade.question_type == "general" else NO_EVIDENCE)
        shown = build_evidence(chunks, self.cfg.MAX_CONTEXT_CHARS).sources
        relevant = _renumber([shown[n - 1] for n in grade.relevant if n <= len(shown)])
        evidence = build_evidence(relevant, self.cfg.MAX_CONTEXT_CHARS)
        notes = ([prompts.PARTIAL_NOTE] if grade.verdict == "partial" else []) + ([prompts.CONFLICT_NOTE] if grade.conflict else [])
        return Plan(GROUNDED, evidence.sources, evidence.text, notes)

    # ------------------------------------------------------------------ exact table analysis
    async def _table(self, req: AgentRequest, cls: Classification, meta: Dict[str, Any]) -> Optional[Plan]:
        """Computed evidence over a whole stored CSV, or None (the caller falls back to search)."""
        started = time.perf_counter()
        try:
            tables = await self.rag.describe_tables(user_id=req.user_id, document_ids=list(cls.table_ids))
            if not tables:
                meta["table_analysis"] = "no_table"
                return None
            result, error, queries = None, None, []
            for _ in range(2):  # one planning retry with the rejection reason
                plan = await structured.plan(self.llm, self.cfg, req.message, tables, error)
                if plan is None:
                    error = "the plan was not valid JSON for one of the listed tables" if error is None else error
                    continue
                queries = [q.model_dump() for q in plan.queries]
                result = await self.rag.query_table(user_id=req.user_id, document_id=plan.document_id, queries=queries)
                if result is not None:
                    break
                error = "RAG Core rejected it (unknown column, wrong operation for the column type, or bad filter value)"
            if not result or not result.get("results"):
                meta["table_analysis"] = "no_plan"
                return None
        except AgentError as exc:
            meta["table_analysis"] = f"failed:{exc.code}"
            return None
        finally:
            meta["table_analysis_ms"] = _ms(started)
        doc_id = uuid.UUID(str(result["document_id"]))
        filename = str(result.get("filename") or "table.csv")[:255]
        text = structured.render(filename, result["results"], queries)
        source = RetrievedChunk(source_id=1, chunk_id=f"{doc_id}:table", document_id=doc_id, filename=filename,
                                page=None, chunk_index=0, text=text, score=1.0)
        meta["table_analysis"] = "computed"
        meta["table_operations"] = [r.get("operation") for r in result["results"]]
        meta["table_rows_total"] = result["results"][0].get("rows_total")
        # The exact results are stored with the answer (backend, server-side) so a later "give the
        # above as a table" reuses them instead of re-running anything.
        meta["structured_result"] = {"document_id": str(doc_id), "filename": filename, "results": result["results"],
                                     "queries": queries}
        meta["retrieval_strategy"] = "table_analysis"
        evidence = build_evidence([source], self.cfg.MAX_CONTEXT_CHARS)
        return Plan(TABLE, evidence.sources, evidence.text, [prompts.TABLE_NOTE], text)

    # ------------------------------------------------------------------ generation
    async def _generate(self, messages, deadline: float, parts: List[str], done: Dict[str, Any]) -> AsyncIterator[StreamItem]:
        async for ev in self.llm.stream(messages, model=self.cfg.AGENT_MODEL, temperature=self.cfg.AGENT_TEMPERATURE,
                                        max_tokens=self.cfg.AGENT_MAX_OUTPUT_TOKENS):
            if time.perf_counter() > deadline:
                raise AgentError(LLM_TIMEOUT, reason="agent_turn_deadline")
            if ev.event == "token":
                delta = ev.data["delta"]
                if delta:
                    parts.append(delta)
                    yield "content.delta", {"delta": delta}
            else:
                done.update(ev.data)

    async def _finalise(self, req: AgentRequest, plan: Plan, answer: str, meta: Dict[str, Any]) -> Tuple[str, str]:
        """(final answer, verification_status) for a generated answer."""
        if plan.mode in (NO_EVIDENCE, GENERAL):
            if is_abstention(answer):
                return prompts.ABSTAIN_MESSAGE, cit.INSUFFICIENT
            return answer, cit.NO_EVIDENCE if meta.get("retrieved") else cit.NOT_APPLICABLE
        if plan.mode not in (GROUNDED, TABLE):
            return answer, cit.NOT_APPLICABLE
        if is_abstention(answer):
            return prompts.ABSTAIN_MESSAGE, cit.INSUFFICIENT
        valid = {s.source_id for s in plan.sources}
        if plan.general_allowed and not cit.extract_citations(cit.strip_invalid_markers(answer, valid), plan.sources):
            meta["answer_check"] = "general_answer_without_document_claims"
            return cit.strip_invalid_markers(answer, valid), cit.NO_EVIDENCE
        cleaned = cit.strip_invalid_markers(answer, valid)
        meta["invalid_citations_removed"] = cleaned != answer

        started = time.perf_counter()
        try:
            if plan.mode == TABLE:
                if structured.numbers_supported(cleaned, plan.table_text):
                    meta["answer_check"] = "numbers_match_computed_results"
                    return cleaned, cit.COMPUTED
                if self.cfg.ANSWER_VERIFICATION_ENABLED:
                    v = await grading.verify(self.llm, self.cfg, req.message, cleaned, plan.evidence)
                    if v.status == "revised" and v.text and structured.numbers_supported(v.text, plan.table_text):
                        meta["answer_check"] = "revised"
                        return cit.strip_invalid_markers(v.text, valid), cit.REVISED
                meta["answer_check"] = "replaced_by_computed_results"
                return plan.table_text + " [1]", cit.COMPUTED

            if not self.cfg.ANSWER_VERIFICATION_ENABLED:
                return cleaned, cit.CITED if cit.extract_citations(cleaned, plan.sources) else cit.UNCITED
            v = await grading.verify(self.llm, self.cfg, req.message, cleaned, plan.evidence)
            meta["unsupported_claims"] = len(v.unsupported)
            if v.status == "supported":
                return cleaned, cit.VERIFIED if cit.extract_citations(cleaned, plan.sources) else cit.UNCITED
            if v.status == "unverified":
                return cleaned, cit.UNVERIFIED if cit.extract_citations(cleaned, plan.sources) else cit.UNCITED
            if v.status == "revised" and v.text:
                revised = cit.strip_invalid_markers(v.text, valid)
                if cit.extract_citations(revised, plan.sources) and not is_abstention(revised):
                    return revised, cit.REVISED
            return prompts.ABSTAIN_MESSAGE, cit.INSUFFICIENT  # nothing in the answer is supported
        finally:
            meta["verification_ms"] = _ms(started)

    # ------------------------------------------------------------------ follow-ups on earlier answers
    async def _followup(self, req: AgentRequest, meta: Dict[str, Any]) -> Optional[Tuple[str, List[RetrievedChunk], str, Optional[dict]]]:
        """(answer, sources, verification_status, structured result to keep) when the message is a
        follow-up the earlier answers can serve; None -> the normal path (retrieval / analysis)."""
        kb = {d.document_id for d in req.library_documents} | set(req.document_ids)
        priors = followup.candidates(req.history, kb)
        target = followup.pick(req.message, priors)
        if target is None:
            meta["followup"] = "no_reusable_answer"
            return None
        if followup.deterministic_table_requested(req.message, target):
            source = followup.table_source(target)
            if source is not None:
                meta["followup"] = "deterministic_table"
                rows = target.structured["results"][0].get("rows_total")
                text = (followup.render_tables(target.structured)
                        + f"\n\nExact counts computed earlier over all {rows} rows of {target.structured['filename']} [1].")
                return text, [source], cit.REFORMATTED, target.structured

        # The model rewrites the earlier answer(s); it may use nothing else.
        use = priors[-4:] if target in priors[-4:] else [target, *priors[-3:]]
        texts, sources = followup.as_sources(use)
        computed = [structured.render(p.structured["filename"], p.structured["results"], p.structured.get("queries"))
                    for p in use if p.structured]
        messages = followup.transform_messages(req.message, texts, sources, computed)
        try:
            result = await asyncio.wait_for(self.llm.generate(messages, model=self.cfg.AGENT_MODEL, temperature=0.1,
                                                              max_tokens=self.cfg.AGENT_MAX_OUTPUT_TOKENS),
                                            self.cfg.LLM_TIMEOUT_SECONDS)
        except (AgentError, asyncio.TimeoutError):
            meta["followup"] = "transform_failed"
            return None
        answer = (result.content or "").strip()
        if not answer or followup.NEEDS_NEW in answer or is_abstention(answer):
            meta["followup"] = "needs_new_analysis"
            return None
        valid = {s.source_id for s in sources}
        answer = cit.strip_invalid_markers(answer, valid)
        evidence = "\n\n".join([*texts, *computed, *(f"[{s.source_id}] {s.text}" for s in sources), "100.0%"])
        if not structured.numbers_supported(followup.without_list_numbers(answer), evidence):
            meta["followup"] = "new_numbers_rejected"  # never let a reformat introduce figures
            return None
        status = cit.REFORMATTED
        if self.cfg.ANSWER_VERIFICATION_ENABLED:
            v = await grading.verify(self.llm, self.cfg, req.message, answer, evidence)
            if v.status == "unsupported":
                meta["followup"] = "unsupported_rejected"
                return None
            if v.status == "revised" and v.text:
                revised = cit.strip_invalid_markers(v.text, valid)
                if not structured.numbers_supported(followup.without_list_numbers(revised), evidence):
                    return None
                answer, status = revised, cit.REVISED
        if sources and not cit.extract_citations(answer, sources):
            # Keep the earlier answer's sources attached to what was derived from it.
            cited = [s.source_id for s in sources if any(c.get("chunk_id") == s.chunk_id for c in target.citations)]
            if cited:
                answer += " " + "".join(f"[{n}]" for n in cited)
        meta["followup"] = "transformed"
        kept = target.structured if target.structured and any(p is target for p in use) else None
        return answer, sources, status, kept

    # ------------------------------------------------------------------ public API
    async def stream(self, req: AgentRequest) -> AsyncIterator[StreamItem]:
        started = time.perf_counter()
        deadline = started + self.cfg.AGENT_TURN_TIMEOUT_SECONDS
        cls = classify(req)
        meta: Dict[str, Any] = {"retrieval_strategy": "none", "retrieval_scope": cls.scope}
        outcome, reason = "error", None
        plan = Plan(GENERAL)
        citations: List[Dict[str, Any]] = []
        yield "agent.started", {}
        try:
            history = normalize_history(req.history, self.cfg)
            if followup.is_followup(req.message):
                if req.history:
                    done = await self._followup(req, meta)
                else:
                    done, meta["followup"] = None, "no_reusable_answer"
                if done is None and meta["followup"] == "no_reusable_answer" and followup.wants_transformation(req.message):
                    # "Give the above as a table" with nothing above to reuse: ask, don't search raw data.
                    meta.update({"answer_mode": "followup_clarify", "retrieval_scope": "conversation", "citation_count": 0,
                                 "verification_status": cit.NOT_APPLICABLE, "total_latency_ms": _ms(started)})
                    yield "generation.started", {}
                    yield "content.delta", {"delta": followup.NOTHING_TO_REUSE}
                    outcome = "completed"
                    yield "generation.completed", {"citations": [], "metadata": meta}
                    return
                if done is not None:
                    answer, sources, status, kept = done
                    plan = Plan(followup.MODE, sources)
                    citations = cit.extract_citations(answer, sources)
                    meta.update({"answer_mode": followup.MODE, "retrieval_scope": "conversation", "citation_count": len(citations),
                                 "verification_status": status, "total_latency_ms": _ms(started)})
                    if kept:
                        meta["structured_result"] = kept
                    yield "generation.started", {}
                    yield "content.delta", {"delta": answer}
                    outcome = "completed"
                    yield "generation.completed", {"citations": citations, "metadata": {k: v for k, v in meta.items() if v is not None}}
                    return
            meta["retrieved"] = cls.retrieve
            if cls.retrieve:
                yield "retrieval.started", {}
                table = None
                if cls.request_type is RequestType.STRUCTURED and self.cfg.TABLE_ANALYSIS_ENABLED:
                    table = await self._table(req, cls, meta)
                plan = table or await self._retrieve_and_grade(req, cls, history, meta)
                meta["retrieved_chunks"] = len(plan.sources)
                yield "retrieval.completed", {"retrieved_chunks": len(plan.sources)}
            meta["answer_mode"] = plan.mode

            yield "generation.started", {}
            if plan.mode == ABSTAIN:
                # The Knowledge Base does not support an answer to a document question: say so
                # exactly, without asking the model (nothing to ground it on, nothing to invent).
                yield "content.delta", {"delta": prompts.ABSTAIN_MESSAGE}
                meta.update({"citation_count": 0, "verification_status": cit.INSUFFICIENT, "total_latency_ms": _ms(started)})
                outcome = "completed"
                yield "generation.completed", {"citations": [], "metadata": {k: v for k, v in meta.items() if v is not None}}
                return

            missing = None
            if cls.reason == "referenced_documents_not_in_knowledge_base":
                missing = f"a {cls.missing_type.upper()} file" if cls.missing_type else "an uploaded document"
            grounded = plan.mode in (GROUNDED, TABLE)
            messages = prompts.build_messages(
                question=req.message, history=history, retrieved=cls.retrieve and plan.mode != GENERAL,
                evidence=plan.evidence if grounded else None, pending_documents=req.pending_document_count,
                missing_document=missing, notes=plan.notes)
            gen_started = time.perf_counter()
            parts: List[str] = []
            done: Dict[str, Any] = {}
            async for item in self._generate(messages, deadline, parts, done):
                yield item
            answer = "".join(parts)
            if not answer.strip():
                raise AgentError(GENERATION_FAILED, "The language model returned an empty answer.", reason="empty_answer")
            meta["generation_latency_ms"] = _ms(gen_started)

            final, status = await self._finalise(req, plan, answer, meta)
            if final != answer:
                yield "content.replace", {"content": final}
            citations = cit.extract_citations(final, plan.sources) if status != cit.INSUFFICIENT else []
            if citations and plan.mode == GROUNDED:
                check = cit.check_citation_numbers(final, plan.sources)
                meta["citation_number_check"] = {"checked": check["checked"], "mismatched": check["mismatched"]}
                if check["mismatched"]:
                    log.warning("citation_numbers_not_in_cited_source", extra={"mismatched": check["mismatched"], "examples": check["examples"]})
            meta.update({
                "model": done.get("model") if isinstance(done.get("model"), str) else None,
                "token_usage": done.get("usage") if isinstance(done.get("usage"), dict) else None,
                "total_latency_ms": _ms(started),
                "citation_count": len(citations),
                "verification_status": status,
                "answer_replaced": final != answer,
            })
            outcome = "completed"
            yield "generation.completed", {"citations": citations, "metadata": {k: v for k, v in meta.items() if v is not None}}
        except AgentError as exc:
            outcome, reason = exc.code, exc.reason
            yield "error", {"code": exc.code, "message": exc.message}
        except (asyncio.CancelledError, GeneratorExit):
            outcome = "cancelled"  # client went away: upstream streams are closed by unwinding
            raise
        except Exception:
            log.exception("turn_failed_unexpectedly")
            outcome, reason = GENERATION_FAILED, "unexpected_error"
            yield "error", {"code": GENERATION_FAILED, "message": "The response could not be generated."}
        finally:
            total = _ms(started)
            METRICS.observe_ms("agent_turn", total, outcome=outcome, request_type=cls.request_type.value)
            if outcome not in ("completed", "cancelled"):
                METRICS.inc("agent_errors", code=outcome, reason=str(reason)[:64])
            # Safe diagnostics only: ids, counts and decisions, never document text or the question.
            log.info("turn_finished", extra={
                "outcome": outcome, "reason": reason, "request_type": cls.request_type.value,
                "retrieval_decision": cls.retrieval.value, "decision_reason": cls.reason,
                "user_id": str(req.user_id), "conversation_id": str(req.conversation_id),
                "retrieval_scope": cls.scope, "answer_mode": plan.mode,
                "explicit_document_ids": [str(d) for d in req.document_ids][:50],
                "knowledge_base_documents": len(req.library_documents),
                "searched_document_ids": [str(d) for d in cls.document_ids][:50],
                "retrieved_document_ids": sorted({str(s.document_id) for s in plan.sources}),
                "cited_document_ids": sorted({str(c.get("document_id")) for c in citations}),
                "query_resolution": meta.get("query_resolution"), "query_rewrite_rejected": meta.get("query_rewrite_rejected"),
                "evidence_grades": meta.get("grades"), "correction_attempts": meta.get("correction_attempts"),
                "question_type": meta.get("question_type"), "table_analysis": meta.get("table_analysis"),
                "table_operations": meta.get("table_operations"), "answer_check": meta.get("answer_check"),
                "followup": meta.get("followup"),
                "retrieved_chunks": meta.get("retrieved_chunks"), "rag_call_ms": meta.get("rag_call_ms"),
                "grading_ms": meta.get("grading_ms"), "verification_ms": meta.get("verification_ms"),
                "generation_latency_ms": meta.get("generation_latency_ms"), "total_latency_ms": total,
                "citation_count": meta.get("citation_count"), "verification_status": meta.get("verification_status"),
                "answer_replaced": meta.get("answer_replaced"),
            })

    async def invoke(self, req: AgentRequest) -> Dict[str, Any]:
        content: List[str] = []
        citations: List[Dict[str, Any]] = []
        metadata: Dict[str, Any] = {}
        async for name, data in self.stream(req):
            if name == "content.delta":
                content.append(data["delta"])
            elif name == "content.replace":
                content = [data["content"]]
            elif name == "generation.completed":
                citations, metadata = data["citations"], data["metadata"]
            elif name == "error":
                raise AgentError(data["code"], data["message"])
        return {"content": "".join(content), "citations": citations, "metadata": metadata}
