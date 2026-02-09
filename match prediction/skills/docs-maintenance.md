---
name: docs-maintenance
description: Use this skill when the context window is getting full, at the end of a session, or when asked to "summarize session" or "update docs". It updates README.md with newly added app features evidenced by the work done, and writes context.md with exactly 10 lines summarizing what was done and next steps. Do not use for general writing tasks unrelated to repo docs.
---

## Goal
Keep repository documentation up to date when context is filling up:
- Update `README.md` with new app features implemented in this session (no speculation).
- Write `context.md` with **exactly 10 lines**: what was done + next steps.

## Inputs to rely on (evidence)
Use only what you can verify from:
- git diff / file changes you can inspect
- commits (if present)
- modifications you made in this session
- explicit decisions recorded in chat

If a feature is not clearly supported by evidence, DO NOT add it as a feature in README. Instead add a short note under a "To verify" / "Da verificare" subsection or omit it.

## Procedure
1) Inspect repository state:
   - Open `README.md` (create if missing).
   - Inspect changes from this session using available tools (prefer: `git diff`, list changed files).
2) Extract "new features" from evidence:
   - A "feature" is a user-facing capability (not refactors unless they change behavior).
   - Deduplicate vs existing bullets in README.
3) Patch README.md:
   - Find (or create) a section named "Funzionalità" (Italian) or "Features" (English); follow the existing README language.
   - Add concise bullet points for each new feature, keeping style consistent.
4) Write context.md with **exactly 10 lines**, compact:
   Line 1: date (YYYY-MM-DD) + session label
   Lines 2–6: what was done (max 1 item per line)
   Lines 7–10: next steps (actionable, max 1 per line)
   No extra blank lines. No headings.
5) Show the exact edits as a diff in your final output and ensure files are saved.

## Output constraints
- `context.md` MUST be exactly 10 non-empty lines.
- README updates must be minimal and evidence-based.
