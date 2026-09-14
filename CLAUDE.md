# Slotfill

Can a small LoRA-tuned model extract key fields from invoice PDFs into a strict
JSON schema at a fraction of a frontier model's cost? The deliverable is a
**recommendation backed by held-out eval numbers**, not a model. Shipping the
prompt is a valid and expected outcome.

Full plan: see `docs/plan.md`.

## Non-negotiables

These exist because the project's credibility depends on them:

- **Held-out sets are frozen.** The ~10 held-out FATURA layout IDs and the
  RVL-CDIP set never touch training or prompt iteration. Not once.
- **Split by layout ID, not by document.** Splitting by document measures
  template memorization, not generalization.
- **Training loss is never a headline result.** Report held-out metrics.
- **No LLM-generated training data.** Labels come from dataset annotations.
- **Every lossy field mapping gets written down** in the README mapping table.
- **Validate against the schema before returning** from the API. A schema
  violation is a structured error, never a silent pass.

## Agent usage

Agents are authorized to run proactively on this project:

| Agent | When |
|---|---|
| python-reviewer | After writing or changing any Python module |
| security-reviewer | Any change to `serve/` - it accepts PDF uploads from the network |
| planner | Before starting a new phase from the plan |
| code-reviewer | Before a commit that touches eval or scoring logic |

Not relevant here: e2e-runner, doc-updater, go-*, database-reviewer.

## Skills worth reaching for

- `eval-harness` before touching `eval/evaluate.py`
- `cost-aware-llm-pipeline` for the cost-per-1k-invoices measurement
- `fastapi-patterns` / `api-design` for `serve/`
- `python-patterns`, `python-testing` for general style

## Testing

TDD applies to the deterministic pieces: schema validation, the field-accuracy
scorer, the FATURA/RVL-CDIP mappers, the layout split. These have real correct
answers and silent bugs in them would corrupt every number in the ship gate.

TDD does not apply to the training loop. Asserting on a fine-tune is ceremony.

## Environment

- Local GPU is a 4GB RTX 3050: enough for data prep, PDF rendering and API-based
  eval. Not enough to train a 3B VLM. Training runs on Colab/Kaggle or rented GPU.
- `ruff format` + `ruff check --fix` run automatically on every Python edit via
  a PostToolUse hook in `.claude/settings.json`.
- Never commit dataset files. `data/download.py` fetches them.
- Secrets live in `.env`, never in source.
