"""AdvancedRAG Agent Core: orchestration. Decides whether to retrieve, rewrites follow-up
questions, assembles the grounded prompt, drives generation and builds citations.
Holds no provider keys: it talks to RAG Core and the LLM layer over internal HTTP."""
