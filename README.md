# Slotfill

**Can a small LoRA-tuned model extract the key fields from invoice PDFs into a strict JSON
schema at a fraction of a frontier model's cost - and if not, is the honest recommendation
to ship the prompt?**

The deliverable is a recommendation backed by held-out eval numbers, not a model. Shipping
the prompt is a valid and expected outcome. Full plan: [`docs/plan.md`](docs/plan.md).

> **Status: phase 2 harness built, baseline model chosen, full held-out run not yet done.**
> Schema, dataset mappers, the frozen layout split and the eval harness (`eval/`) are in
> place and verified end to end with an oracle backend. A dev-set shakedown across several
> candidates picked **Gemini 3.8 Flash** as the prompted frontier baseline: 100% schema
> validity, field accuracy and exact match across 28 `dev_unseen` documents (two independent
> samples, zero failures), against 85% / 81% / 45% for the cheapest alternative tried
> (DeepSeek v4-flash-vision-exp), which also had a 15% catastrophic-failure rate (empty
> output, full token budget burned on hidden reasoning). The real held-out run (`test_seen`
> + `test_unseen` + `rvlcdip`, ~1,170 documents) has not happened yet, so there are no
> ship-gate numbers to report. The table below is empty on purpose.
>
> **Phase 3 done: the first adapter is trained.** Qwen2.5-VL-3B, LoRA r=16, one epoch over
> 1,750 FATURA pages, 81 minutes on a free Colab T4
> ([`runs/train/qwen25vl3b-r16-train/`](runs/train/qwen25vl3b-r16-train/)). On a 20-page
> smoke test over unseen layouts it scores **100% schema validity, 98% field accuracy,
> 80% exact match** - against 0% validity for the untuned base model, which reads the page
> correctly but answers in its own invented field names. Those are 20 pages of one layout,
> not a result; the held-out numbers come from phase 4
> ([Evaluating the adapter](#evaluating-the-adapter)), which is built and not yet run.

## The answer

_Pending phase 2. This section will lead with the recommendation._

## Ship-gate table

| Condition | Eval set | Schema validity | Field accuracy | Exact match | Cost / 1k | p95 latency |
|---|---|---|---|---|---|---|
| Prompted frontier baseline | FATURA unseen layouts | - | - | - | - | - |
| Prompted frontier baseline | RVL-CDIP (real scans) | - | - | - | - | - |
| Tuned adapter | FATURA unseen layouts | - | - | - | - | - |
| Tuned adapter | RVL-CDIP (real scans) | - | - | - | - | - |

---

## Getting the data

Dataset files are never committed. `data/download.py` fetches them and verifies each
archive against the md5 published in its Zenodo record, so a truncated transfer fails
loudly rather than quietly producing a short dataset.

```bash
python -m data.download            # both datasets
python -m data.download --verify   # re-check what is already on disk
python -m data.build_splits        # write data/splits/*.json
```

| Dataset | Role | Size | Licence |
|---|---|---|---|
| [FATURA 2](https://doi.org/10.5281/zenodo.10371464) | train + held-out layouts | 10,000 synthetic invoices, 50 layouts | CC-BY-4.0 |
| [RVL-CDIP layout ground truth](https://doi.org/10.5281/zenodo.3257319) | zero-shot reality check | 520 real scanned invoices | CC-BY-4.0 |

Both are CC-BY-4.0, so redistribution with attribution would be permitted; we still ship a
download script only, and cite both papers in [`data/sources.py`](data/sources.py).

## The schema

[`schema/invoice_schema.json`](schema/invoice_schema.json) is the correctness signal for
the whole project. 14 fields, `additionalProperties: false` at every level, and **every
property required** - nullability is expressed by the `null` type rather than by omitting a
key, so a missing key is always a real error and never an ambiguous one.

[`schema/validate.py`](schema/validate.py) turns a violation into a structured
`ValidationOutcome` carrying a dotted JSON path per error (`line_items.0.quantity`), never a
bare exception. It tolerates a markdown fence around model output and nothing else: prose
around the JSON is counted as a failure rather than salvaged, because "the model chatted at
us" is a failure mode the eval needs to be able to count.

One field carries two kinds of "nothing". `line_items` is `[]` when a line table was looked
for and not found, and `null` when the extractor does not produce line items at all. The
distinction exists because no dataset in this project annotates line items (see below), so
the tuned adapter cannot learn them and is trained to say `null` - an abstention - rather
than `[]`, which would claim on every FATURA page that a table it plainly has is absent.
The prompted baseline reads the table and fills the array; the adapter declines. Both are
schema-valid, and neither is scored on it.

`null` was added to `line_items` after the frontier baseline had run on the held-out sets.
Because the system prompt embeds the schema text, this moved the baseline's prompt digest,
so those runs are closed: they cannot be resumed under their recorded digest, only rerun
under a new one. It did not move a number. Re-validating every stored baseline output under
the relaxed schema gives the same verdict on all 1,083 documents (350/350, 298/300 and
431/433 valid), no valid output had `line_items: null`, and every invalid one is a JSON
parse failure or an empty reply. Run configs now record a `schema_digest` alongside the
prompt digest so a future schema edit is visible as such.

## The split

Splitting by document would put images from the same template in both train and test, and
the resulting accuracy would measure template memorisation. Every split here is keyed on
the **layout id**.

| Split | Layouts | Documents | Frozen | Purpose |
|---|---|---|---|---|
| `train` | 35 | 1,750 | no | fine-tuning |
| `dev_unseen` | 5 | 200 | no | prompt iteration on unseen layouts |
| `test_seen` | 35 | 350 | **yes** | same layouts as train, disjoint documents |
| `test_unseen` | 10 | 300 | **yes** | the 10 held-out layouts |
| RVL-CDIP | n/a | 520 | **yes** | real scans, zero-shot, shared fields only |
| `train_4k` | 35 | 4,025 | no | data-mix ablation: `train` plus 65 more documents per layout, same layouts, disjoint from every eval set |

FATURA ships `Strat2_Split.txt`, a 40/10 inter-template split - but it sets
`test_inds = dev_inds`, so its dev set *is* its test set. Iterating a prompt against that
would burn the held-out set immediately. We keep its 10 held-out layouts untouched and
carve a 5-layout dev set out of its 40 training layouts instead.

The manifests in [`data/splits/`](data/splits/) are committed and each carries a SHA-256
digest of its own contents, so the freeze is auditable in git history.
`write_manifests` refuses to change a frozen split once written, and `load_manifest`
refuses to load one that has been edited by hand.

## Labelling process

### FATURA class mapping

The released data has **35 annotation keys, not the 24 the paper tabulates**. The
eleven-class gap: the paper shows the seven `GST(<rate>%)` classes as a single "GST" row
(-6), omits the three `GSTIN*` classes (-3), and does not tabulate `INVOICE_INFO` or
`OTHER` at all (-2) even though both appear in every one of the 10,000 released files.
Every annotated value also arrives with its printed label attached
(`"TOTAL : 972.30 EUR"`, not `"972.30"`), so label stripping and normalisation are the bulk
of [`data/map_fatura.py`](data/map_fatura.py).

This table is generated by `python -m data.mapping_report`, so it cannot drift from the
code.

| FATURA class | Schema field | Lossy? | Note |
|---|---|---|---|
| `AMOUNT_DUE` | `amount_due` | no | printed label stripped |
| `BILL_TO` | `buyer` | yes | precedence BUYER > BILL_TO > SEND_TO; SHIP_TO is not the billed party |
| `BUYER` | `buyer` | yes | precedence BUYER > BILL_TO > SEND_TO; SHIP_TO is not the billed party |
| `CONDITIONS` | `payment_terms` | no | printed label stripped |
| `DATE` | `invoice_date` | no | printed label stripped |
| `DISCOUNT` | `discount` | no | printed as '(-) 51.2'; stored as a positive amount deducted |
| `DUE_DATE` | `due_date` | no | printed label stripped |
| `GST(1%)` | `tax` | yes | rate-parameterised; VAT wins when both appear on one document |
| `GST(12%)` | `tax` | yes | rate-parameterised; VAT wins when both appear on one document |
| `GST(18%)` | `tax` | yes | rate-parameterised; VAT wins when both appear on one document |
| `GST(20%)` | `tax` | yes | rate-parameterised; VAT wins when both appear on one document |
| `GST(5%)` | `tax` | yes | rate-parameterised; VAT wins when both appear on one document |
| `GST(7%)` | `tax` | yes | rate-parameterised; VAT wins when both appear on one document |
| `GST(9%)` | `tax` | yes | rate-parameterised; VAT wins when both appear on one document |
| `NUMBER` | `invoice_number` | no | printed label stripped |
| `PO_NUMBER` | `purchase_order_number` | no | printed label stripped |
| `SELLER_ADDRESS` | `vendor.address` | no | printed label stripped |
| `SELLER_EMAIL` | `vendor.email` | no | printed label stripped |
| `SELLER_NAME` | `vendor.name` | no | printed label stripped |
| `SELLER_SITE` | `vendor.website` | no | printed label stripped |
| `SEND_TO` | `buyer` | yes | precedence BUYER > BILL_TO > SEND_TO; SHIP_TO is not the billed party |
| `SUB_TOTAL` | `subtotal` | no | printed label stripped |
| `TAX` | `tax` | no | printed label stripped |
| `TOTAL` | `total_amount` | no | printed label stripped |
| `GSTIN` | _dropped_ | yes | a tax registration number, not a tax amount - no schema field for it |
| `GSTIN_BUYER` | _dropped_ | yes | a tax registration number, not a tax amount - no schema field for it |
| `GSTIN_SELLER` | _dropped_ | yes | a tax registration number, not a tax amount - no schema field for it |
| `INVOICE_INFO` | _dropped_ | yes | present as an empty list in all 10,000 files; carries no content |
| `LOGO` | _dropped_ | yes | graphical element, not an extractable field |
| `NOTE` | _dropped_ | yes | free-text remarks and footers with no schema counterpart |
| `OTHER` | _dropped_ | yes | a text pass over the whole page; supervising on it would leak the answer, and it names a different vendor than the image on every document (it is the Path A text layer, see eval/datasets.py) |
| `PAYMENT_DETAILS` | _dropped_ | yes | bank account details; deliberately out of scope for this schema |
| `TABLE` | _dropped_ | yes | bounding box only - no cell text is released, so line items cannot be labelled |
| `TITLE` | _dropped_ | yes | document heading ('INVOICE', 'COMMERCIAL INVOICE'), not invoice data |
| `TOTAL_WORDS` | _dropped_ | yes | the total spelled out in words - redundant with total_amount |

### What FATURA can and cannot supervise

Coverage over the 1,750 training documents:

| Schema field | Supervised by FATURA | Coverage |
|---|---|---|
| `invoice_number` | yes | 1550 / 1750 (89%) |
| `purchase_order_number` | yes | 250 / 1750 (14%) |
| `invoice_date` | yes | 1700 / 1750 (97%) |
| `due_date` | yes | 899 / 1750 (51%) |
| `vendor.name` | yes | 1050 / 1750 (60%) |
| `vendor.address` | yes | 1350 / 1750 (77%) |
| `vendor.email` | yes | 800 / 1750 (46%) |
| `vendor.website` | yes | 300 / 1750 (17%) |
| `buyer.name` | yes | 1550 / 1750 (89%) |
| `buyer.address` | yes | 1550 / 1750 (89%) |
| `currency` | yes | 1600 / 1750 (91%) |
| `subtotal` | yes | 1150 / 1750 (66%) |
| `discount` | yes | 300 / 1750 (17%) |
| `tax` | yes | 850 / 1750 (49%) |
| `total_amount` | yes | 1350 / 1750 (77%) |
| `amount_due` | yes | 200 / 1750 (11%) |
| `payment_terms` | yes | 250 / 1750 (14%) |
| `line_items` | **no** - TABLE is a bounding box with no cell text; no line item is annotated | 0 |

**`line_items` is not annotated anywhere in FATURA.** `TABLE` is a bare bounding box - a
single `table` token in the LayoutLM view - with no cell text in any of the three released
annotation formats. The mapper therefore *omits* the key rather than emitting `[]`, which
would be a label nobody wrote. `line_items` stays in the schema because the API contract
needs it, but nothing in this project trains or scores on it until a dataset with real line
item annotations (DocILE) is available.

Three conflicts are counted rather than hidden:

| Conflict | Documents | Resolution |
|---|---|---|
| `tax:multiple_gst_rates` | 30 / 300 in `test_unseen` (Template25 only) | the page prints a **GST rate card**: five lines at 1/5/12/18/20% of the subtotal. `TOTAL - SUB_TOTAL` matches none of them, so no printed figure is the tax charged. `tax` is left **unsupervised** for these documents rather than guessed. |
| `tax:vat_and_gst` | 50 / 1,750 in `train` | a VAT line and a single GST line on one invoice. The schema has one `tax` field; we take the VAT line, which is a real printed charge. Summing would invent a number printed nowhere on the page. |
| `buyer:multiple_sources` | 150 / 1,750 in `train` | precedence `BUYER > BILL_TO > SEND_TO`. `SEND_TO` is a ship-to party, which is not strictly the billed party, and is only used when nothing better exists. |

The rate-card case is worth dwelling on: resolving it by tuple order would have put an
arbitrary figure into the `tax` ground truth of 10% of the frozen held-out set, where no
conflict counter would have shown it. Coverage numbers are the place such a bug surfaces,
which is why the table above is generated rather than written.

One derived field: **`currency` is not annotated**. It is read off the amount strings
(`"1309.36  EUR"`, `"379.82 $"`) and the most common code across a document's amounts wins.
FATURA's `$` is unqualified throughout and is read as `USD`.

### RVL-CDIP: regions, not field values

This is the mapping that most needs stating plainly. RVL-CDIP's ground truth is **region
boxes over noisy OCR, not field values**. A `*_gt.xml` file contains six labelled regions
(`supplier`, `receiver`, `invoice_info`, `positions`, `total`, `other`) and no text at all;
the text comes from a separate ABBYY OCR pass over 1970s-90s litigation scans. Pooling all
65,370 OCR words across the 520 documents, the median per-word confidence is **0.51** and
the mean is **0.52** (`python -m data.ocr_stats`).

There is no per-field ground truth value anywhere in the dataset. A `total` region routinely
holds a whole column of figures; an `invoice_info` region holds the invoice number, the
date and their printed labels as one run of text.

So **RVL-CDIP scores grounding, not accuracy**: for eight fields we can ask "does the
predicted value appear inside the correct annotated region?" and nothing stronger. Every
other schema field is declared unscoreable rather than handed an invented label.

| Schema field | Region | Scoring | Note |
|---|---|---|---|
| `vendor.name`, `vendor.address` | `supplier` | containment | the block mixes name, address and phone; not separable |
| `buyer.name`, `buyer.address` | `receiver` | containment | the block mixes name, address and attn line |
| `invoice_number`, `invoice_date`, `due_date` | `invoice_info` | containment | one run of text holding all three plus their labels |
| `total_amount` | `total` | containment | the region often holds a whole column of figures |
| `subtotal`, `tax`, `discount`, `amount_due`, `currency`, `payment_terms`, `purchase_order_number`, `line_items` | - | **not scoreable** | no annotation exists; scoring these would invent labels |

Region text recovered across the 520 documents: `supplier` 463, `receiver` 473,
`invoice_info` 480, `positions` 450, `total` 359, `other` 500.

## Running the baseline

Phase 2 sets the bar before any training: the prompted frontier model on both held-out
sets, plus the Path A pre-check (text layer -> small model). Everything runs through one
harness, so a later adapter is compared on the model and nothing else.

```bash
cp .env.example .env            # OPENAI_API_KEY=<Google AI Studio key>
set -a; source .env; set +a

# 1. Iterate the prompt on the dev set only. It is unseen layouts, but not frozen.
#    Cheap dev iteration goes through OpenRouter: no daily quota, ~2x the direct-Google
#    cost for this model, negligible at this sample size.
python -m eval.predict --set dev_unseen --input image \
    --backend openai --base-url https://openrouter.ai/api/v1 \
    --model google/gemini-3.8-flash --condition vision-frontier --limit 20
python -m eval.evaluate runs/vision-frontier/dev_unseen

# 2. The frontier baseline on the held-out sets. --held-out is required on purpose.
#    Direct to Google AI Studio: needs paid billing enabled on the project first - the
#    free tier caps at ~20 requests/day pooled across models, nowhere near enough for
#    ~1,170 documents. Direct pricing also measured cheaper than the OpenRouter path for
#    this model (lower output-token counts observed), which is why the real run goes here.
python -m eval.predict --set test_seen   --input image \
    --backend openai --base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
    --model gemini-3.8-flash --condition vision-frontier --held-out
python -m eval.predict --set test_unseen --input image \
    --backend openai --base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
    --model gemini-3.8-flash --condition vision-frontier --held-out
python -m eval.predict --set rvlcdip     --input image \
    --backend openai --base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
    --model gemini-3.8-flash --condition vision-frontier --held-out

# 3. Path A pre-check: shipped text layer -> small model.
python -m eval.predict --set test_unseen --input text --model claude-haiku-4-5 \
    --condition text-small --held-out
python -m eval.predict --set rvlcdip     --input text --model claude-haiku-4-5 \
    --condition text-small --held-out

python -m eval.evaluate runs/*/test_seen runs/*/test_unseen runs/*/rvlcdip
```

Each run writes `runs/<condition>/<set>/run_config.json` (model, effort, every knob, and a
SHA-256 of the prompt) and `predictions.jsonl` (raw output, token counts and wall-clock
latency per document). Runs resume: answered documents are skipped, errored ones are
retried, and a run whose prompt or model changed is refused rather than silently mixed.
The prompt digest is how a held-out run can be shown to have used the prompt that was
frozen before it, not one tweaked after.

### Why Gemini 3.8 Flash, and why two different endpoints for it

The frontier baseline was chosen by shakedown-testing several candidates on 20-28
`dev_unseen` documents each rather than committing API spend to an unproven model. Claude
Opus 5 was the original default (see the plan); Gemini 3.8 Flash was tried instead because
the plan only requires the baseline be "genuinely strong," not a specific vendor, and this
kept the shakedown itself nearly free. Findings:

- **DeepSeek v4-flash-vision-exp** (OpenRouter): 85% schema validity, 45% exact match, and
  a 15% rate of returning nothing at all - the model spent its full 16K-token output budget
  on invisible reasoning and produced zero visible text. Cheap per token, expensive per
  usable invoice.
- **Gemini 3.8 Flash**: 100% schema validity, field accuracy and exact match across every
  document tried, on both the direct-Google and OpenRouter paths (28 documents combined,
  zero failures).

Gemini 3.8 Flash's own free tier (direct via Google) was also tried and ruled out for
anything beyond a handful of documents: the observed daily quota is pooled across model
versions at roughly 20 requests/day total, not 20 per model as the API's own error messages
imply - confirmed empirically by switching model names mid-quota and getting blocked
immediately rather than a fresh allowance. Two paid endpoints remain for the same model:
OpenRouter (already funded, no setup, but this model measured ~2x the output tokens per
document through it) and direct Google (needs billing enabled, but cheaper per document) -
hence dev iteration on the former and the real run on the latter.

### Trying other models through OpenRouter

`--backend openai` targets any OpenAI-compatible endpoint, which is how the vLLM adapter
gets served in phase 5, but the same flag also reaches OpenRouter today - one balance,
many providers, useful for comparing candidate models cheaply before committing real API
spend to a held-out run. This is exactly how the shakedown above was run:

```bash
# .env: OPENROUTER_API_KEY=... (eval/predict.py reads it automatically - no --api-key needed)
python -m eval.predict --set dev_unseen --input image \
    --backend openai --base-url https://openrouter.ai/api/v1 \
    --model deepseek/deepseek-v4-flash-vision-exp \
    --condition deepseek-test --limit 20
python -m eval.evaluate runs/deepseek-test/dev_unseen
```

Two things do not come for free through this path:

- **Cost.** [`eval/pricing.py`](eval/pricing.py) only prices models it has an explicit,
  verified entry for - Anthropic's own models, plus Gemini 3.8 Flash now that it is the
  chosen baseline. `Cost / 1k invoices` reads `-` for anything else (DeepSeek included) -
  the metric is honestly absent, not silently wrong. Add an entry to `PRICES` (with a
  source for the rate) before trusting that column for a new model.
- **Credibility as the published baseline.** The plan asks for a genuinely strong
  frontier model here, specifically so beating it means something. An unfamiliar or
  low-quality model is a fine, cheap way to smoke-test the harness end to end - it is not
  a substitute for the shakedown that picks the actual baseline.

**The prompt** ([`eval/prompt.py`](eval/prompt.py)) is the strong baseline the plan asks
for: the full schema, explicit conventions (ISO dates, ISO 4217 codes, numbers as numbers,
null when not printed) and one worked example, `Template48_Instance97` from the *train*
split, as a page image plus its hand-written record. The example is the cacheable prefix.

**Metrics** ([`eval/evaluate.py`](eval/evaluate.py)), identical for every condition:

| Metric | Definition |
|---|---|
| Schema validity | parses as JSON and validates; a markdown fence is tolerated, prose is not |
| Field accuracy (all) | correct field instances / scoreable ones, an invalid output wrong on every field. **The headline view.** |
| Field accuracy (valid) | the same over schema-valid outputs only |
| Exact match | documents with every scoreable field right (FATURA only; RVL-CDIP has no values) |
| Cost / 1k invoices | measured tokens x published prices; self-hosted cost is measured at the serving layer |
| p95 latency | nearest-rank over the run's successful requests, same code path for every condition |

Comparison rules live in [`eval/scoring.py`](eval/scoring.py): money agrees when rounded
half-up to the cent, dates and currency codes match exactly, other strings match after
case folding and collapsing whitespace and commas (so a wrapped address matches the joined
label). Only fields the dataset annotated for that document are scored; a prediction for
an unannotated field has no label to check against. `line_items` is never scored.

On RVL-CDIP "correct" means grounded in the right annotated region, and a null prediction
is an **abstention** - excluded from the denominator and reported separately - because a
region box cannot say whether a field was printed. Expect the OCR noise to cap grounding
below what the model actually reads: `I5UO.OO` will not match `1500.00`, and that ceiling
is counted, not corrected.

### Path A and the text layers

Neither dataset ships PDFs, so the text path consumes the text layer each dataset does
ship: FATURA's `OTHER` class and RVL-CDIP's ABBYY OCR words joined in reading order.
FATURA's is a stale text pass: on every document it names a different vendor than both the
image and the `SELLER_NAME` label (0 of 360 agree; every other field agrees on 83-100%).
`vendor.name` and `vendor.website` are therefore **unscoreable on the text path** and the
evaluator prints a per-field *text-layer ceiling* - the share of reference values present
in the input text at all - next to the text-path accuracy. The text path cannot beat it,
and reading Path A numbers without it would blame the model for the input.

## Training the adapter

Phase 3. The local GPU (4GB) cannot fine-tune a 3B vision model, so the split between
machines is explicit: everything deterministic happens here and is tested; the training
loop runs on Colab/Kaggle from a bundle this repo produces.

```bash
python -m train.export        # data/interim/slotfill-bundle/ + slotfill-bundle.tar.gz (~135 MB)
```

The bundle holds every page image once (4,025 of them: `train` is a subset of `train_4k`),
one `<split>.jsonl` per split, and a `bundle.json` with a SHA-256 digest per split.
`dev_unseen` travels too, marked `trainable: false`, for the post-training smoke test; the
GPU side refuses to train on it, and refuses to export a frozen set at all.

### What the model learns to say

The target for a page is the **full schema record as one compact JSON string**
([`train/targets.py`](train/targets.py)). Two rules bridge the mapper's partial record to
the schema's required-everything contract, and both are checked rather than assumed:

- **An unannotated field is `null`.** FATURA annotates every field its template prints: 34
  of the 35 training layouts have exactly one supervised field set across all 50 of their
  documents, and the 35th (layout 40) differs on a single document that lacks a `DUE_DATE`
  annotation (`python -m train.target_report`; the same holds over `train_4k`). "Not
  annotated" is therefore "not printed", and `null` is the schema's word for exactly that.
  This is a label the dataset wrote, not one we filled in.
- **`line_items` is `null`**, never `[]` - see [The schema](#the-schema).

Every target is validated against the schema at export time; a mapper bug cannot reach the
model. Keys are emitted in schema order and the JSON is compact, because output tokens are
what the adapter's cost per invoice is made of and formatting is invisible to the scorer.

The prompt is the page image followed by one fixed sentence,
`Extract the fields from this invoice page. Reply with the JSON object only.` - **no schema,
no worked example**. The schema goes into the weights; that is the cost argument in one
line. The adapter is evaluated through the same harness on exactly this prompt
(`eval/predict.py --no-example --no-schema`), and the run config records a distinct prompt
digest for it.

### The run

[`train/train_vlm.py`](train/train_vlm.py), via [`train/colab_train.ipynb`](train/colab_train.ipynb)
(one notebook, `PLATFORM = 'colab' | 'kaggle'`; the two runs are deliberately split across the
two free tiers so both are covered):
Unsloth vision LoRA on `Qwen2.5-VL-3B-Instruct` loaded in 4-bit, starting from the plan's
knobs (`r=16, alpha=16, lr=2e-4`, 1 epoch, seed 3407), loss on the assistant turn only.
Vision layers are frozen by default: vLLM's LoRA support for multimodal models covers the
language model, and an adapter that cannot be served is not a result. Each run writes:

| File | Contents |
|---|---|
| `adapter/` | LoRA weights and `adapter_config.json`, loadable by vLLM `--enable-lora` |
| `run_config.json` | every knob, the base model, the **dataset digest** from the bundle, library versions, GPU, wall-clock training time |
| `smoke.json` | schema validity and field accuracy on 20 `dev_unseen` pages - enough to tell a working adapter from a broken run before spending a held-out evaluation on it |
| `train_log.json` | the trainer's loss curve. It lives here and nowhere else: training loss is never a headline result |
| `trainer/checkpoint-N/` | a checkpoint every 50 optimizer steps; rerunning with the same `--out` resumes from the latest. Free Colab reclaims sessions and Kaggle wipes the disk on a crash, so `--out` lives on Drive or in Kaggle's saved output. Delete once the adapter is saved |

The digest chain is deliberate: `bundle.json` names the exact examples, `run_config.json`
copies that digest, and the held-out run's own `run_config.json` names the model. A
number in the ship-gate table can be traced back to the documents it was trained on.

**The ablation is data mix**, the one the plan calls more interesting for this dataset:
the same knobs on `train` (1,750 documents) and `train_4k` (4,025 documents, the same 35
layouts). More documents on the same templates either buys generalisation to the 10 unseen
layouts or it buys memorisation that `test_seen` will reward and `test_unseen` will not;
the two eval sets are what tell those apart. Reported as held-out metrics, never as loss.

Adapter weights are not committed. Their `run_config.json`, `smoke.json` and
`train_log.json` are, under `runs/train/<adapter>/`, as evidence.

## Evaluating the adapter

Phase 4. The adapter is scored on the same three frozen sets the frontier baseline was
scored on, through the same harness, so the two differ on the model and the prompt and
nothing else.

The serving layer is where "same harness" becomes literal:
[`serve/serve_vllm.sh`](serve/serve_vllm.sh) starts vLLM with `--enable-lora` and both
conditions behind one endpoint, and the OpenAI `model` field selects which - base model or
adapter. `eval/predict.py --backend openai --base-url http://localhost:8000/v1` is the
same code path that reached Gemini.

```bash
serve/serve_vllm.sh <adapter dir>          # base + adapter on :8000

python -m eval.predict --set test_unseen --input image \
    --backend openai --base-url http://localhost:8000/v1 --model slotfill-lora \
    --condition vision-adapter --no-example --no-schema --max-tokens 1024 --held-out

python -m eval.evaluate --gpu-usd-per-hour 0.35 runs/vision-adapter/test_unseen
```

`--no-example --no-schema` is not a weaker prompt by accident: it is the prompt the
adapter was trained on (page image + one sentence), and the run config's prompt digest
records that it differs from the baseline's. The baseline keeps its schema and worked
example. That asymmetry is the experiment - the ~1,450 tokens of schema the baseline
re-sends on every invoice are what the adapter moved into its weights.

**Cost, for a model we host.** There is no published per-token price for it, and borrowing
one from an API would be a different measurement wearing this one's label. So each run
records its own wall clock in `sessions.jsonl`, `eval/evaluate.py --gpu-usd-per-hour`
turns that into `cost / 1k = rate / measured throughput`, and `metrics.json` reports the
throughput next to it. Without a rate the column reads `-`.

[`eval/colab_eval.ipynb`](eval/colab_eval.ipynb) runs all of this on a free Colab or
Kaggle GPU. One caveat, stated because it is the likeliest thing to go wrong: a T4 is a
2018 card with no bfloat16, and vLLM's newer kernels increasingly assume newer hardware.
The notebook therefore sends a single request before the 1,083-document run, and names a
paid L4 (a dollar or two) as the fallback rather than pretending the free path is certain.

## Limitations

- Training data is **synthetic**. FATURA's content is generated, its layouts are clean, and
  its dates are internally inconsistent (due dates frequently precede invoice dates). Expect
  RVL-CDIP numbers visibly below FATURA held-out numbers; that gap is a finding, not a bug.
- The real-document test set is **520 scans of one narrow domain** (tobacco litigation
  archives), with poor OCR, and it can only score grounding.
- **No line item supervision exists** in either dataset.
- DocILE (6.7k real annotated invoice-like documents with an unseen-layout test split) is
  the upgrade path for both limitations. Access is gated behind a form; nothing here depends
  on it.

## The trap, addressed

- **No LLM-generated training data.** Every label comes from dataset annotations, and where
  an annotation does not exist the field is declared unsupervised rather than filled in.
- **Split by layout id, not by document**, so the headline number cannot be template
  memorisation.
- **Held-out sets are frozen** with committed digests, and the code refuses to move them.
- **Training loss is never a headline result.** The README reports held-out metrics.

## Layout

```
schema/    invoice_schema.json, validate.py   the correctness signal
data/      download.py, sources.py            fetch + provenance
           map_fatura.py, map_rvlcdip.py      annotations -> schema
           split.py, build_splits.py          layout-id split
           splits/*.json                      frozen manifests (committed)
           mapping_report.py                  generates the table above
train/     targets.py                         mapped record -> full schema record; leak guard
           target_report.py                   the "unannotated = not printed" measurement
           export.py                          images + targets -> self-describing bundle
           train_vlm.py                       Unsloth vision LoRA; run_config.json, smoke.json
           smoke.py                           post-training check on dev_unseen pages
           colab_train.ipynb                  thin notebook around train_vlm.py
eval/      predict.py                         run one condition on one set; resumable
           colab_eval.ipynb                   serve the adapter and score the held-out sets
           evaluate.py                        a run -> metrics.json + ship-gate row
           prompt.py                          schema prompt + the worked example
           scoring.py                         per-document field comparison rules
           backends.py                        Anthropic SDK / OpenAI-compatible (vLLM)
           datasets.py, pages.py, pricing.py  eval items, image encoding, published prices
serve/     serve_vllm.sh                      vLLM + --enable-lora; model field picks condition
runs/      <condition>/<set>/                 run_config.json, predictions.jsonl,
                                              sessions.jsonl, metrics.json
           train/<adapter>/                   run_config.json, smoke.json, train_log.json
adapters/  <adapter>/adapter/                 LoRA weights (gitignored)
tests/                                        schema, mappers, split, scorer, harness, targets
```

## Citations

- Limam et al., *FATURA: A Multi-Layout Invoice Image Dataset for Document Analysis and
  Understanding*, arXiv:2311.11856, 2023.
- Riba, Dutta, Goldmann, Fornes, Ramos, Llados, *Table Detection in Invoice Documents by
  Graph Neural Networks*, ICDAR 2019.
