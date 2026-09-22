# Slotfill

**Can a small LoRA-tuned model extract the key fields from invoice PDFs into a strict JSON
schema at a fraction of a frontier model's cost - and if not, is the honest recommendation
to ship the prompt?**

The deliverable is a recommendation backed by held-out eval numbers, not a model. Shipping
the prompt is a valid and expected outcome. Full plan: [`docs/plan.md`](docs/plan.md).

> **Status: phases 1-4 complete.** Schema, mappers, frozen layout splits, eval harness,
> the prompted frontier baseline, one trained LoRA adapter and both sides of the ship-gate
> table, measured on ~1,083 held-out documents. The ablation (data mix) and the serving app
> are the remaining work.

## The answer

**Ship the prompt. Keep the adapter warm for one specific case.**

A LoRA-tuned Qwen2.5-VL-3B does get within **2.6 points of field accuracy** of a prompted
frontier model on invoice layouts neither has seen, at **1/30th the cost** - and on real
scanned invoices, where the frontier model burns ten times its usual output budget,
**1/90th**. That part of the hypothesis holds up.

It fails on the two things an accounting import actually needs:

- **Whole-record correctness.** 96.2% of fields right sounds close to 98.9%, but a record
  has ~12 fields, and per-document perfection compounds: **65% exact match versus 95%** on
  unseen layouts. One in three extracted invoices needs a human to touch it, against one in
  twenty. The misses are character-level misreads of long, unpredictable strings
  (`3934 Gonzales Loop` read as `9394 Gonzales Loop`, `Scottland` as `Scotland`) - the
  reading resolution of a 3B model, not a fixable labelling problem.
- **Schema compliance on real documents.** On the RVL-CDIP scans the adapter returns
  **80.6% valid JSON against the baseline's 99.5%**. Nearly one in five real invoices comes
  back as a structured error. That is precisely the failure this project set out to prevent.
  Note what this is *not*: among the outputs that do validate, the adapter grounds slightly
  better than the baseline (56.4% against 54.8%). The problem is the envelope, not the
  reading.

The honest reading of the cost column is that it is a **floor**, not a forecast: $0.17 per
1,000 assumes the GPU is busy every second it is rented. Break-even against the API is
around **70 invoices per hour sustained**; below that volume, an idle GPU makes the prompt
cheaper as well as better.

**Where the adapter would win anyway:** high-volume batch extraction over clean,
template-like documents close to its training distribution, where its 97.8% field accuracy
on seen layouts and 30-90x cost advantage are decisive and the 1-in-6 whole-record miss
rate is acceptable because the work is queued, not interactive.

**The experiment that could change this verdict** is constrained decoding: the RVL-CDIP
failures are structural drift (the model invents `buyer.email`, `buyer.city`, or emits
`"DM"` where the schema wants ISO 4217), not garbled output, and a grammar-constrained
decoder at the serving layer eliminates that class of error by construction. If validity
goes to ~100% there, the recommendation for real scans deserves to be rerun. It is not
claimed here because it has not been measured.

One thing the fine-tune is unambiguously good at, and it is not accuracy: it **learned the
dataset's labelling conventions**. On `amount_due` it scores 100% against the baseline's
52.5%, because "which printed figure counts as the amount due" is a convention you can
teach but not easily prompt.

## Ship-gate table

Every number below is measured on a frozen held-out set that neither condition was
iterated against. Field accuracy is the **all-outputs** view: an invalid output is wrong on
every field.

| Condition | Eval set | n | Schema validity | Field accuracy | Exact match | Cost / 1k | Throughput |
|---|---|---|---|---|---|---|---|
| Prompted frontier (Gemini 3.8 Flash) | FATURA seen layouts | 350 | 100.0% | 99.0% | 90.9% | $4.86 | - |
| Prompted frontier (Gemini 3.8 Flash) | FATURA **unseen** layouts | 300 | 99.3% | 98.9% | **95.0%** | $4.95 | - |
| Prompted frontier (Gemini 3.8 Flash) | RVL-CDIP (real scans) | 433 | **99.5%** | 54.4% | n/a | $18.65 | - |
| Tuned adapter (Qwen2.5-VL-3B, r=16) | FATURA seen layouts | 350 | 99.4% | 97.8% | 83.7% | $0.16 | 2,211/h |
| Tuned adapter (Qwen2.5-VL-3B, r=16) | FATURA **unseen** layouts | 300 | 100.0% | 96.2% | 65.0% | **$0.17** | 2,118/h |
| Tuned adapter (Qwen2.5-VL-3B, r=16) | RVL-CDIP (real scans) | 433 | **80.6%** | 43.4% | n/a | $0.21 | 1,683/h |

RVL-CDIP has no field values, only region boxes, so it scores **grounding** (is the
predicted value inside the right annotated region?) and has no exact-match column; a null
prediction there is an abstention, excluded from the denominator. Among *valid* outputs the
two conditions ground equally well (**56.4% adapter, 54.8% baseline**) - the adapter's
deficit on that set is entirely schema compliance, not reading. And both of those numbers
are floors; see [why the RVL-CDIP column is a floor](#why-the-rvl-cdip-column-is-a-floor).

**Reading the cost column.** The baseline's is measured tokens x published prices. The
adapter's is a rented T4 at $0.35/hour divided by measured throughput, which assumes 100%
utilisation - a floor. Break-even against the baseline is ~3.3% utilisation, about **70
invoices/hour sustained**. On owned hardware (a ~$300 consumer card, amortised, plus power)
the floor is roughly $0.02 per 1,000, but the machine still has to be up, reachable and
maintained, which the API does not.

Note where the cost gap comes from: the baseline sends **4,444 input tokens per invoice**
(the schema and the worked example, re-sent every time) and the adapter sends **666**, with
the schema in its weights instead. On RVL-CDIP the baseline also spent 4,084 output tokens
per document against the adapter's 153, which is where the 90x gap on real scans comes
from.

**Latency is not compared here on purpose.** The baseline ran at the harness default of 4
concurrent requests against a remote API; the adapter ran at 16 against one local GPU,
where per-request latency rises with batch load by design (p50 26s, p95 34s on
`test_unseen`). Those measure different things. Throughput is the meaningful self-hosted
number, and a concurrency-matched single-request latency measurement is still outstanding.

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
| RVL-CDIP | n/a | 433 | **yes** | real scans, zero-shot, shared fields only. 433 of the 520 available, frozen in `data/splits/rvlcdip_sample.json` when a budget constraint cut the first run short, so every later condition scores the same documents |
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

### What the held-out run showed

**Seen versus unseen layouts.** The adapter loses 1.6 points of field accuracy and 18.7
points of exact match moving from layouts it trained on to layouts it has never seen
(97.8% -> 96.2%, 83.7% -> 65.0%). The baseline, which trained on nothing, moves the other
way (90.9% -> 95.0% exact match; the unseen layouts happen to annotate fewer hard fields).
That asymmetry is the layout-ID split doing its job: a document-level split would have hidden
it completely.

**Where the adapter's errors are.** Concentrated in long, unpredictable strings. On unseen
layouts: `vendor.email` 77.3% (against 99.3%), `buyer.address` 92.0%, `vendor.address`
91.5%, while `invoice_date`, `currency` and `subtotal` are all at 100%. Short, structured,
guessable fields are solved; character-perfect transcription of invented street names and
addresses is not. Sampling the mismatches shows the field located correctly and one or two
characters misread, which is a vision-resolution limit of a 3B model in 4-bit.

**Where the adapter beats the baseline**, and why it is not an accuracy story: `amount_due`
100% against 52.5%, `tax` 99.4% against 94.1% on seen layouts. Both are convention
questions - which printed figure the dataset counts as the amount due, whether a VAT line
or a GST line is the tax - and conventions are exactly what supervised labels transmit and
a prompt has to guess.

**The synthetic-to-real gap is a schema gap.** The plan predicted RVL-CDIP numbers below
FATURA's, and both conditions deliver that (98.9% -> 50.4% for the baseline). The
unpredicted part is *how* the adapter degrades. Its 84 invalid outputs are not garbled
text; they are structural drift, categorised from the raw outputs:

| Failure | Documents | What happened |
|---|---|---|
| extra keys in `buyer` | 42 | invented `email`, `website`, `city`, `state`, `postcode`, `currency` - fields FATURA's `vendor` has and its `buyer` never did |
| `currency` not ISO 4217 | 9 | `"DM"` read off 1990s German invoices. The model read the page correctly and the schema rejected the answer |
| malformed JSON | 7 | a missing quote or brace, on the noisiest scans |
| `buyer.address` missing | 5 | a required key omitted rather than nulled |

Every one of those is a decoding-time constraint away from being impossible, which is why
constrained decoding is the named next experiment rather than more training data.

### Why the RVL-CDIP column is a floor

Roughly 55% grounding looks alarming next to 99% on FATURA, and most of the gap is the
ground truth rather than the models. RVL-CDIP has no field values: a prediction is scored
by whether it appears inside the right annotated region of an **ABBYY OCR pass over
1970s-90s microfilm, median per-word confidence 0.51**. When the OCR is wrong, a correct
answer scores wrong.

That is easy to assert and harder to measure without inventing labels, so
[`eval/agreement.py`](eval/agreement.py) bounds it using the two conditions as independent
witnesses: field instances where **both produced the identical value and both were marked
wrong**. Two different models agreeing character for character is unlikely to be a shared
hallucination.

```
1,313 field instances where both conditions returned the same value
      869 scored correct, 444 scored wrong

Of the ones scored wrong:
  194 (43.7%)  no resemblance: region lacks the value
  160 (36.0%)  value present, OCR garbled a few characters
   90 (20.3%)  value present, heavier OCR damage
```

So **at least 250 of those 444** have the agreed value sitting in the region in damaged
form - the models read the page and the ground truth could not confirm it. Examples, taken
verbatim from the run:

| Both models answered | Region OCR says |
|---|---|
| `PHILIP MORRIS INCORP.` | `To PHII IP MORRIS INCORP. 120 PARK AVENUE` |
| `2001-10-19` | `InvoJco Number: 59276-1 Dafo: October 19. 9001` |
| `6800.0` | `90 Days 120 Days BALANCE DUE Ar800.00` |
| `1991-06-13` | `PURCHASE ORDER NO. DATE 1470 06 13 j 91 TERMS` |

The remaining 44% ("no resemblance") is genuinely ambiguous: the OCR pass routinely drops
letterheads and logos, which is exactly where `vendor.name` is printed, so some of it is
the same problem - and some of it is both models being wrong. The analysis deliberately
stops there rather than crediting itself a corrected score.

**One real scorer bug came out of this.** 5% of the agreed-and-wrong cases differed from
the region text only in punctuation or spacing (`Philip Morris-USA` vs `Philip Morris -USA`).
Region containment now compares alphanumeric skeletons, which moved both conditions up by
the same ~3-4 points (baseline 50.4% -> 54.4%, adapter 40.0% -> 43.4%). The leniency is
confined to RVL-CDIP containment; FATURA, which has real field values, is still compared
strictly, and its numbers did not move. The runs were not repeated - the raw outputs are
committed, so every condition was simply re-scored with the fixed rule.

## Limitations

- The adapter's headline weakness is **character-level transcription**, and no amount of
  FATURA would fix it: the misread strings are randomly generated place names that no
  model can infer from context. A larger base model or a higher-resolution vision path is
  the lever, not more examples.
- **Constrained decoding is untested.** The RVL-CDIP validity gap (80.6% vs 99.5%) is the
  single biggest input to the recommendation, and it is plausibly an artefact of
  unconstrained generation rather than of the model. Until that is measured, the
  recommendation for real scans is provisional.
- **Latency is not comparable across conditions** as measured (different concurrency,
  remote API vs local GPU); only throughput is.
- Training data is **synthetic**. FATURA's content is generated, its layouts are clean, and
  its dates are internally inconsistent (due dates frequently precede invoice dates). Expect
  RVL-CDIP numbers visibly below FATURA held-out numbers; that gap is a finding, not a bug.
- The RVL-CDIP grounding numbers are **floors, not accuracy estimates**, for the reasons
  measured above. They are comparable *between* conditions, which is what the ship-gate
  table uses them for, and should not be read as "the model got half the fields right".
- The real-document test set is **433 scans of one narrow domain** (tobacco litigation
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
           agreement.py                       bounds the RVL-CDIP OCR ceiling
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
