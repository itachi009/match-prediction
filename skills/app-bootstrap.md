---
name: app-bootstrap
description: Use this skill at the start of a new Codex chat (or when asked to "continue the app", "riprendere il lavoro", "stato dell’arte", "next steps"). It reconstructs the current state of the app by reading README.md and context.md, optionally inspecting additional repo files only as needed, and proposes a concrete continuation plan (milestones, tasks, file-level actions, risks).
---

## Goal
Rebuild state-of-the-art quickly and propose a plan to continue development.

## Inputs / Evidence
Primary sources:
- README.md (project overview, features, how to run)
- context.md (10-line recap + next steps)

Secondary sources (read ONLY if needed):
- package manifests (package.json, pyproject.toml, requirements.txt, etc.)
- config files (docker-compose, .env.example, etc.)
- source directories (src/, app/, backend/, frontend/, packages/, etc.)
- tests/ and CI files

Avoid scanning the entire repo by default. Prefer targeted reads based on README/context and file discovery.

## Procedure
1) Read README.md and context.md.
   - If one is missing, note it and proceed with what exists.
2) Extract a "Current Snapshot":
   - What the app does (one paragraph)
   - Current features (bullets)
   - Architecture guess based on evidence (stack, folders, entrypoints)
   - Current status / progress (from context.md)
3) Identify "What’s unclear" (max 5 bullets).
4) Decide whether additional files are needed:
   - If README/context already define stack + structure, do minimal exploration.
   - Otherwise, list top-level folders and open only the most relevant manifests/entrypoints.
   - Examples:
     - If JS/TS: package.json, src/app entry, next.config, etc.
     - If Python: pyproject/requirements, main/app entry, etc.
5) Propose a continuation plan:
   - 3–5 milestones (ordered)
   - For each milestone: outcomes, tasks (checkbox list), files likely to touch, estimated complexity (S/M/L)
   - Include risks/dependencies and "definition of done"
6) Propose immediate next action:
   - A single “next commit” suggestion: what to implement first + where.

## Output format (required)
Produce the following sections in order:

### Snapshot
- (short paragraph)
- Features bullets
- Repo/stack notes (evidence-based)

### Gaps / Questions
- Up to 5 items (only if necessary)

### Plan
Milestone 1...
Milestone 2...
...

### Next Commit Suggestion
- Task list + file targets

## Constraints
- Do not invent features or architecture not supported by files.
- Prefer reading fewer files; expand exploration only when it unblocks planning.
- If the repo is large, avoid opening many files; use file listing to target reads.
- Always ask only one command at a time (otherwise Visual Studio Code will crash) and wait for it to respond before moving on to the next one.
- Always reply in Italian