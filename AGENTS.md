# AGENTS

Notes on AI-assisted development of this repository.

## Tools used
- Claude Code (this repo's primary AI coding tool)
- Cursor / ChatGPT for spot research on LangGraph / Langfuse APIs

## Authorization posture
- Destructive commands (`git reset --hard`, `rm -rf`, `docker ...`) require explicit user approval per session.
- Network-facing calls (curl, real LLM/OCR providers) only executed on explicit request.

## Where AI was trusted
- Pydantic model scaffolding
- Alembic migration SQL
- Tool registry boilerplate
- README / spec documentation drafts

## Where AI was NOT trusted
- Prompt text for the categorization sub-agent (reviewed line by line)
- Error-band classification logic (hand-audited against the spec)
- Mock/real adapter contract parity (verified by Layer 4 e2e smoke test)
- Any decision that changed an externally-visible schema or SSE event name

## How to reproduce
- Claude Code session transcripts are stored in `transcripts/`.
- The design doc at `docs/superpowers/specs/2026-04-20-receipt-processing-agent-design.md` captures the Q&A that drove the design.
