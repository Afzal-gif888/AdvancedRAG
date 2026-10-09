"""Cross-cutting helpers for AdvancedRAG internal services.

Deliberately small: request-ID propagation, redacting structured logs, metrics,
internal service authentication and a uniform error envelope. No business logic.
"""
