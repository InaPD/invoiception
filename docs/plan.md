# Slotfill - Project Plan

**Question the project answers:** can a small LoRA-tuned model extract the key fields from invoice PDFs into a strict JSON schema at a fraction of a frontier model's cost - and if not, is the honest recommendation to ship the prompt?

Level: advanced · Effort: two weekends · Budget: ~$0 (free Colab/Kaggle for text-model work; a few dollars of rented GPU if the vision path needs it)

The deliverable is a **recommendation backed by held-out eval numbers**, not a model. Training loss is never a headline result.

---

## 0. Decisions (locked)

| Decision | Choice | Notes |
|---|---|---|
| Input | Invoice PDFs (born-digital and scanned) | Not pre-extracted text - the pipeline owns PDF handling |
| Primary architecture | **Path B: vision-native.** PDF pages -> images -> LoRA on a small vision-language model (Qwen2.5-VL-3B/7B class, Unsloth-trainable) -> JSON | No OCR stage to break; matches what the image datasets are annotated for; the more distinctive skill to show |
| Kept as pre-check | **Path A: text-mediated.** PyMuPDF text layer (born-digital) or OCR (scans) -> small text LLM -> JSON | Cheap. If a clean text layer + prompted small model already passes the gate, that's a finding worth reporting |
| Training data | **FATURA**, sampled ~1,500-2,000 docs across all 50 layouts | Test project - deliberately not training on all 10k; more data on 50 templates mostly buys memorization |
| Main eval | **FATURA held-out layouts**: split by LAYOUT ID, not document. Train on ~40 layouts, hold out ~10 layouts entirely (a few hundred docs) | The difference between measuring generalization and measuring template memorization |
| Reality check | **RVL-CDIP invoice subset** (~520 real invoices with field annotations), zero-shot, shared fields only | Real scans vs FATURA's clean synthetic; expect worse numbers and say so |
| Optional upgrade | **DocILE** (6.7k real annotated invoice-like docs, unseen-layout test split) | Gated: access form -> secret token -> download script; turnaround not guaranteed. Submit the form in the background; swap in as headline eval if/when the token arrives. Project does not depend on it |
| Baseline | Prompted frontier model with a STRONG prompt (schema + one worked example) on the same held-out sets | Beating a lazy prompt proves nothing |

Datasets deliberately not used: SROIE/CORD (receipts, not B2B invoices - only add if an extra OOD check is wanted); FATURA-as-test-only (too clean to be the sole eval).

---

## 1. Repo skeleton

```
slotfill/
  schema/
    invoice_schema.json      # strict target schema (~10-14 fields)
  data/
    download.py              # fetch FATURA (Zenodo) + RVL-CDIP invoices; NO dataset files committed
    map_fatura.py            # FATURA 24-class annotations -> schema; documents every lossy mapping
    map_rvlcdip.py           # RVL-CDIP fields -> shared-field subset of schema
    split.py                 # layout-ID split: ~40 train / ~10 held-out layouts
  train/
    train_vlm.py             # Unsloth vision LoRA (Path B), logs r/alpha/lr/epochs/seed
    train_text.py            # optional Path A LoRA
  eval/
    predict.py               # runs any condition via OpenAI-compatible endpoint; records latency + tokens
    evaluate.py              # schema validity, per-field accuracy, exact match
  serve/
    serve_vllm.sh            # vLLM --enable-lora, model field selects adapter
    api.py                   # FastAPI: accept PDF -> render pages -> model -> validate -> JSON or structured error
  README.md
```

License note: link datasets and ship `download.py`; do not redistribute images in the repo until each dataset's terms are checked.

---

## 2. Phase 1 - Schema + data mapping (weekend 1, first half)

1. Define `invoice_schema.json`: invoice_number, invoice_date (ISO), due_date, vendor/seller, buyer, currency, subtotal, tax, total_amount, line_items[{description, qty, unit_price, amount}], payment_terms. Nullable where genuinely optional; `additionalProperties: false`; validates or it doesn't - that is the correctness signal.
2. `map_fatura.py`: collapse FATURA's 24 annotated classes into the schema. Write down every lossy decision (which classes merged, which dropped, why) - this table goes in the README as the "documented labelling process".
3. `map_rvlcdip.py`: RVL-CDIP annotates ~6 semantic fields (receiver, supplier, invoice_info, total, ...) - map only the overlap; RVL-CDIP is scored on shared fields only.
4. `split.py`: sample ~1,500-2,000 FATURA docs stratified across all 50 layouts; hold out ~10 layout IDs completely. Freeze the held-out sets NOW - they never touch training or prompt iteration.

## 3. Phase 2 - Baseline before any training (weekend 1, second half)

Run the prompted frontier baseline (schema + worked example in the prompt) on both held-out sets via `eval/predict.py`. This sets the bar and decides what the fine-tune must win on: if the baseline is already ~99% schema-valid, the adapter competes on cost and latency, not accuracy - design the writeup accordingly.

Also run the Path A pre-check here: born-digital PDFs -> text layer -> prompted SMALL model. If that already passes the gate for text-layer PDFs, the honest architecture is "text path for born-digital, tuned VLM only for scans" - exactly the when-not-to-fine-tune judgment the project exists to demonstrate.

## 4. Phase 3 - LoRA training (weekend 2, first half)

- Unsloth vision fine-tune (Path B): page image + fixed short instruction -> JSON string. Start r=16, alpha=16, lr=2e-4, 1-2 epochs. Free T4 if the 3B fits; else a rented L4/A10 for a couple of hours.
- Log (r, alpha, lr, epochs, seed) per run to `run_config.json`.
- One real ablation, pick ONE: rank sweep (r = 4/16/64) or data mix (1,500 vs 4,000 FATURA docs). For this dataset the data-mix question is the more interesting one; report as held-out metrics, never loss.

## 5. Phase 4 - Eval + ship gate (weekend 2, second half)

One harness, every condition (frontier baseline, tuned adapter, optional Path A):

- **Schema validity rate** - parses as JSON and validates (markdown fences tolerated, nothing else)
- **Field-level accuracy** - two views: over all outputs (invalid = wrong) and over valid outputs only; money with small numeric tolerance
- **Exact match** on the full record
- **Cost / 1,000 invoices** - measured tokens x published prices (API path); GPU $/hr / measured throughput (self-hosted)
- **p95 latency** - same runs, same code path for both conditions
- Splits that matter for PDFs: FATURA seen-vs-unseen layouts; FATURA (synthetic) vs RVL-CDIP (real); born-digital vs scanned if using Path A anywhere

**Ship gate (README headline):** the comparison table + a clear recommendation, including the honest outcome where the prompt wins and the prompt ships. Expect RVL-CDIP numbers visibly below FATURA held-out - that synthetic-to-real gap is the finding, state it plainly.

## 6. Phase 5 - Serving

- `serve_vllm.sh`: vLLM with `--enable-lora`; the OpenAI `model` field selects base vs adapter, so both conditions are benchmarked through identical code. (Verify LoRA flag names against the installed vLLM version - they've changed across releases.)
- `api.py`: accept a PDF upload -> render pages (pdf2image/PyMuPDF) -> model -> **validate against the schema before returning** -> JSON on success, structured error on violation. That validate-before-return step is the fix for "schema violations break the accounting import" and deserves its own README paragraph.

## 6b. Phase 6 - App frontend

Make it a real app: upload a PDF, get validated JSON back.

- Single-page frontend (plain HTML+JS or one small React page) served as static files by the same FastAPI app - one deployment, no CORS.
- Upload via drag-and-drop or file picker; restrict to `application/pdf`, enforce a file-size limit; PDF preview pane and a loading state.
- Results rendered two ways: pretty-printed JSON with a copy button, and a key/value table of the extracted fields. On schema violation, show the validation errors, not a blank failure.
- **Compare mode (the fun part):** a model selector - *tuned adapter* vs *prompted baseline* - wired to the OpenAI `model` field the backend already routes on, plus a "run both" button that shows the two JSON outputs side by side with differing fields highlighted and per-run latency displayed. The app becomes a live demo of the project's whole thesis, and the side-by-side is the README GIF.
- Public demo option: HuggingFace Space. Gradio (`gr.File` in, `gr.JSON` out) would replace the frontend entirely at the cost of looking less like a product - keep the custom frontend for the repo, note Gradio as fallback.

## 7. README structure

1. The question + the answer (recommendation up front)
2. Ship-gate table: validity / field accuracy / exact match / cost per 1k / p95, baseline vs adapter, per eval set
3. FATURA seen vs unseen layouts + RVL-CDIP zero-shot - the generalization story
4. The ablation actually run (with config table)
5. Labelling process: the FATURA class-mapping table + layout split
6. Limitations: synthetic training data, small real test set, gap to production invoices; DocILE named as the upgrade path
7. The trap, addressed: no LLM-generated training data, layout-ID split prevents template memorization, headline = held-out metrics not loss

## 8. Milestones

| # | Deliverable | Done when |
|---|---|---|
| M1 | Schema + mappers + frozen splits | Both held-out sets written and untouched thereafter |
| M2 | Baseline numbers | Frontier (+ Path A pre-check) scored on both held-out sets |
| M3 | Tuned adapter | In-schema output on a smoke test; run config logged |
| M4 | Full eval | All conditions through one harness; cost + p95 measured |
| M5 | Serving + README | PDF-in -> validated-JSON-out endpoint works; ship-gate table filled; recommendation written |
| M6 | App frontend | Upload -> JSON in the browser; compare mode runs adapter vs baseline side by side with diff highlighting; GIF recorded for README |
| M7 (optional) | Ablation · DocILE swap-in if token arrives · SROIE extra OOD check | Charts/tables in README |

## 9. Open questions to verify while building

- FATURA licence terms (Zenodo record) before redistributing any images - default to download-script-only
- Exact RVL-CDIP invoice-subset annotation format and where its 6 fields overlap the schema
- Whether Qwen2.5-VL-3B vision LoRA fits free-T4 VRAM with Unsloth, or the run needs a rented L4/A10
- vLLM vision + LoRA serving flags for the installed version
- DocILE token turnaround (form submitted in background; nothing blocks on it)
