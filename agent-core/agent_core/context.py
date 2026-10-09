"""Grounded evidence builder: validated RAG chunks -> deterministic, delimited LLM evidence.

Each retrieved chunk becomes one numbered source block:

    <source id="1" document="employee_handbook.pdf" page="2">
    …chunk text, verbatim…
    </source>

- Order and numbering are RAG Core's rank order; the number is what the model cites as [n].
- Chunk text is untrusted document DATA. It is copied verbatim except that anything resembling
  the delimiters is neutralised, so a document cannot close its block and smuggle text outside.
- Metadata comes only from RAG Core (no invented pages); attributes are sanitised the same way.
- The total chunk text respects RAG Core's MAX_CONTEXT_CHARS contract (12,000): sources are kept
  in rank order until the next one would exceed it (the first is always kept). RAG Core already
  applies the same rule, so this only guards against contract drift; nothing is re-retrieved.
"""
import re
from dataclasses import dataclass
from typing import List

from agent_core.schemas import RetrievedChunk

_DELIMITER = re.compile(r"<\s*(/?)\s*(source|evidence)\b", re.I)


def _neutralise(text: str) -> str:
    return _DELIMITER.sub(lambda m: f"‹{m.group(1)}{m.group(2)}", text)


def _attr(value: str) -> str:
    return _neutralise(" ".join(value.split())).replace('"', "'").replace("<", "‹").replace(">", "›")


@dataclass(frozen=True)
class Evidence:
    text: str
    sources: List[RetrievedChunk]  # exactly the chunks shown to the model, in citation order


def build_evidence(chunks: List[RetrievedChunk], max_chars: int) -> Evidence:
    blocks, used, kept = [], 0, []
    for chunk in chunks:
        if kept and used + len(chunk.text) > max_chars:
            break
        page = f' page="{chunk.page}"' if chunk.page else ""
        blocks.append(f'<source id="{chunk.source_id}" document="{_attr(chunk.filename)}"{page}>\n'
                      f"{_neutralise(chunk.text)}\n</source>")
        used += len(chunk.text)
        kept.append(chunk)
    return Evidence("\n\n".join(blocks), kept)
