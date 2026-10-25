from advancerag_common.service import ServiceError

# Non-retryable (4xx): the backend marks the document failed with a matching safe message.
UNSUPPORTED_CONTENT = "UNSUPPORTED_CONTENT"
NO_EXTRACTABLE_TEXT = "NO_EXTRACTABLE_TEXT"
PARSE_FAILED = "PARSE_FAILED"
DOCUMENT_TOO_LARGE = "DOCUMENT_TOO_LARGE"
EMBEDDING_REJECTED = "EMBEDDING_REJECTED"
EMBEDDING_MODEL_MISMATCH = "EMBEDDING_MODEL_MISMATCH"

# Retryable (5xx).
EMBEDDING_UNAVAILABLE = "EMBEDDING_UNAVAILABLE"
VECTOR_STORE_UNAVAILABLE = "VECTOR_STORE_UNAVAILABLE"
VECTOR_STORE_MISCONFIGURED = "VECTOR_STORE_MISCONFIGURED"


def unsupported(message: str) -> ServiceError:
    return ServiceError(UNSUPPORTED_CONTENT, message, 415)


def no_text() -> ServiceError:
    return ServiceError(NO_EXTRACTABLE_TEXT, "No extractable text was found in this document.", 422)


def parse_failed() -> ServiceError:
    return ServiceError(PARSE_FAILED, "The document could not be parsed.", 422)


def embedding_unavailable() -> ServiceError:
    return ServiceError(EMBEDDING_UNAVAILABLE, "The embedding service is unavailable.", 503, retryable=True)


def vector_store_unavailable() -> ServiceError:
    return ServiceError(VECTOR_STORE_UNAVAILABLE, "The vector store is unavailable.", 503, retryable=True)
