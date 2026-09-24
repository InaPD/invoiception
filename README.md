# Invoiception

Invoiception extracts the fields of an invoice page into a strict JSON schema, and measures
two ways of doing it against each other: a prompted frontier model, and a LoRA-tuned
Qwen2.5-VL-3B served locally. Every number here comes from frozen held-out sets that
neither condition was iterated against.

## Recommendation

**Ship the prompt. Keep the adapter for high-volume batch work on clean documents.**

The tuned adapter lands within **2.6 points of field accuracy** of the prompted frontier
model on invoice layouts neither has seen, at **1/30th the cost**, and at **1/90th** on real
scans where the frontier model spends ten times its usual output budget. It falls short on
the two properties an accounting import depends on:

- **Whole-record correctness.** 96.2% of fields correct against 98.9% is close; per-document
  perfection over ~12 fields is not. Exact match is **65% against 95%** on unseen layouts,
  so one extracted invoice in three needs a human, against one in twenty. The misses are
  character-level misreads of long unpredictable strings (`3934 Gonzales Loop` as
  `9394 Gonzales Loop`, `Scottland` as `Scotland`), which is reading resolution rather than
  a labelling defect.
- **Schema compliance on real documents.** Zero-shot on real scans the adapter returns
  **80.6% valid JSON against 99.5%**, so nearly one real invoice in five comes back as a
  structured error. Restricted to outputs that do validate, it reads those scans as well as
  the baseline, so the deficit is the envelope rather than the reading.

The cost figure is a floor rather than a forecast: $0.17 per 1,000 assumes the rented GPU is
busy every second. Break-even against the API sits near **70 invoices per hour sustained**,
and below that an idle GPU makes the prompt cheaper as well as more accurate.

The adapter wins on high-volume batch extraction over clean, template-like documents near
its training distribution, where 97.8% field accuracy on seen layouts and a 30-90x cost
advantage decide it and a 1-in-6 whole-record miss rate is tolerable because the work is
queued rather than interactive.

One thing the fine-tune does unambiguously well is not accuracy: it absorbs the dataset's
labelling conventions. On `amount_due` it scores 100% against the baseline's 52.5%, because
which printed figure counts as the amount due is teachable but hard to prompt.

**Untested lever: constrained decoding.** The RVL-CDIP failures are structural drift (the
model invents `buyer.email` or `buyer.city`, or emits `"DM"` where the schema wants ISO
4217), not garbled text, and a grammar-constrained decoder at the serving layer rules out
that class by construction. No measurement of it exists here, so it is not claimed.

## Results

Measured on the frozen FATURA held-out sets. Field accuracy is the **all-outputs** view: an
invalid output counts wrong on every field.

| Condition | Eval set | n | Schema validity | Field accuracy | Exact match | Cost / 1k | Throughput |
|---|---|---|---|---|---|---|---|
| Prompted frontier (Gemini 3.8 Flash) | seen layouts | 350 | 100.0% | 99.0% | 90.9% | $4.86 | - |
| Prompted frontier (Gemini 3.8 Flash) | **unseen** layouts | 300 | 99.3% | 98.9% | **95.0%** | $4.95 | - |
| Tuned adapter (Qwen2.5-VL-3B, r=16) | seen layouts | 350 | 99.4% | 97.8% | 83.7% | $0.16 | 2,211/h |
| Tuned adapter (Qwen2.5-VL-3B, r=16) | **unseen** layouts | 300 | 100.0% | 96.2% | 65.0% | **$0.17** | 2,118/h |

**Seen against unseen layouts.** The adapter loses 1.6 points of field accuracy and 18.7
points of exact match moving from trained layouts to unseen ones (97.8% -> 96.2%, 83.7% ->
65.0%). The baseline moves the other way on exact match (90.9% -> 95.0%; the unseen layouts
annotate fewer hard fields). A document-level split would hide that asymmetry entirely.

**Where the adapter's errors sit.** In long, unpredictable strings. On unseen layouts:
`vendor.email` 77.3% (against 99.3%), `buyer.address` 92.0%, `vendor.address` 91.5%, while
`invoice_date`, `currency` and `subtotal` are at 100%. Sampled mismatches show the field
located correctly with one or two characters misread.

**Where the adapter beats the baseline** is convention, not accuracy: `amount_due` 100%
against 52.5%, `tax` 99.4% against 94.1% on seen layouts. Which printed figure counts as the
amount due, and whether a VAT or GST line is the tax, are what supervised labels transmit
and a prompt has to guess.

**The cost column.** The baseline's is measured tokens x published prices. The adapter's is
a rented T4 at $0.35/hour over measured throughput, at an assumed 100% utilisation.
Break-even against the baseline is ~3.3% utilisation. On owned hardware (a ~$300 consumer
card, amortised, plus power) the floor is roughly $0.02 per 1,000, with the machine still to
be kept up, reachable and maintained.

The gap comes from prompt size: the baseline sends **4,444 input tokens per invoice** (the
schema and the worked example, re-sent every call) against the adapter's **666**, which
carries the schema in its weights.

**Latency is not comparable across conditions.** The baseline ran at 4 concurrent requests
against a remote API, the adapter at 16 against one local GPU where per-request latency
rises with batch load by design (p50 26s, p95 34s on `test_unseen`). Throughput is the
meaningful self-hosted number.

### Real scans: the schema-compliance check

FATURA is synthetic, so a zero-shot pass over 433 real scanned invoices (RVL-CDIP) is the
only evidence here about documents outside that distribution. It answers one question
cleanly and one only partially.

**Cleanly: does the output still satisfy the schema?** This is read off the model's own
output and needs no ground truth.

| Condition | Schema validity | Invalid outputs | Cost / 1k |
|---|---|---|---|
| Prompted frontier | **99.5%** | 2 / 433 | $18.65 |
| Tuned adapter | **80.6%** | 79 / 433 | $0.21 |

That gap is the single biggest input to the recommendation. The adapter's invalid outputs
are structural drift rather than garbled text:

| Failure | Documents | What happened |
|---|---|---|
| extra keys in `buyer` | 42 | invented `email`, `website`, `city`, `state`, `postcode`, `currency` - fields FATURA's `vendor` has and its `buyer` never did |
| `currency` not ISO 4217 | 9 | `"DM"` read off 1990s German invoices; the page was read correctly and the schema rejected the answer |
| malformed JSON | 7 | a missing quote or brace, on the noisiest scans |
| `buyer.address` missing | 5 | a required key omitted rather than nulled |

Each of those is a decoding-time constraint away from being impossible, which is why
constrained decoding is the named next experiment rather than more training data.

The baseline's output-token cost is the other real-scan finding: 4,084 output tokens per
document against the adapter's 153, which is where the 90x cost gap comes from.

**Partially: how much did each model actually read?** RVL-CDIP has no field values, only
region boxes over an ABBYY OCR pass of 1970s-90s microfilm at **median per-word confidence
0.51**, so a correct answer scores wrong wherever the OCR is. The metric is containment
inside the right region, it is a **floor rather than an accuracy estimate**, and it is worth
reading only between conditions, which face identical ground truth:

| Condition | Grounding, valid outputs only | Grounding, all outputs |
|---|---|---|
| Prompted frontier | 54.8% | 54.4% |
| Tuned adapter | **56.4%** | 43.4% |

Restricted to outputs that validate, the adapter grounds marginally better than the
baseline. Its deficit on real scans is the envelope, not the reading. Neither number should
be read as "the model got half the fields right".

[`eval/agreement.py`](eval/agreement.py) bounds the OCR damage using the two conditions as
independent witnesses: field instances where both produced an identical value and both were
marked wrong. Two models agreeing character for character is unlikely to be a shared
hallucination.

```
1,313 field instances where both conditions returned the same value
      869 scored correct, 444 scored wrong

Of the ones scored wrong:
  194 (43.7%)  no resemblance: region lacks the value
  160 (36.0%)  value present, OCR garbled a few characters
   90 (20.3%)  value present, heavier OCR damage
```

So at least **250 of those 444** have the agreed value in the region in damaged form
(`PHILIP MORRIS INCORP.` against region text `To PHII IP MORRIS INCORP. 120 PARK AVENUE`;
`1991-06-13` against `PURCHASE ORDER NO. DATE 1470 06 13 j 91 TERMS`). The remaining 44% is
ambiguous: the OCR pass routinely drops letterheads and logos, which is where `vendor.name`
is printed, and some of it is both models being wrong. The analysis stops there rather than
crediting itself a corrected score. Region containment compares alphanumeric skeletons, so
`Philip Morris-USA` matches `Philip Morris -USA`; that leniency is confined to RVL-CDIP, and
FATURA, which has real field values, is compared strictly.

---
## Running it

```bash
pip install -e ".[dev]"
cp .env.example .env            # OPENAI_API_KEY=<Google AI Studio key>
set -a; source .env; set +a
```

### Data

Dataset files are never committed. `data/download.py` fetches them and verifies each archive
against the md5 published in its Zenodo record, so a truncated transfer fails loudly instead
of producing a short dataset.

```bash
python -m data.download            # both datasets
python -m data.download --verify   # re-check what is already on disk
python -m data.build_splits        # write data/splits/*.json
```

| Dataset | Role | Size | Licence |
|---|---|---|---|
| [FATURA 2](https://doi.org/10.5281/zenodo.10371464) | train + held-out layouts | 10,000 synthetic invoices, 50 layouts | CC-BY-4.0 |
| [RVL-CDIP layout ground truth](https://doi.org/10.5281/zenodo.3257319) | zero-shot reality check | 520 real scanned invoices | CC-BY-4.0 |

Both are CC-BY-4.0 and redistribution with attribution would be permitted; this repo ships a
download script only and cites both papers in [`data/sources.py`](data/sources.py).

### The frontier baseline

`--held-out` is required for any run against a frozen set.

```bash
# Prompt iteration runs on dev_unseen only: unseen layouts, not frozen.
# OpenRouter has no daily quota and costs ~2x direct Google for this model.
python -m eval.predict --set dev_unseen --input image \
    --backend openai --base-url https://openrouter.ai/api/v1 \
    --model google/gemini-3.8-flash --condition vision-frontier --limit 20
python -m eval.evaluate runs/vision-frontier/dev_unseen

# Held-out runs go direct to Google AI Studio, which needs paid billing enabled.
# The free tier caps at ~20 requests/day pooled across model versions, against
# ~1,170 documents here, and measures more output tokens per document than the
# direct path.
for SET in test_seen test_unseen rvlcdip; do
  python -m eval.predict --set "$SET" --input image \
      --backend openai --base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
      --model gemini-3.8-flash --condition vision-frontier --held-out
done

# Text-layer path: shipped text layer -> small model.
python -m eval.predict --set test_unseen --input text --model claude-haiku-4-5 \
    --condition text-small --held-out

python -m eval.evaluate runs/*/test_seen runs/*/test_unseen runs/*/rvlcdip
```

Each run writes `runs/<condition>/<set>/run_config.json` (model, every knob, a SHA-256 of
the prompt and of the schema) and `predictions.jsonl` (raw output, token counts, wall-clock
latency per document). Runs resume: answered documents are skipped, errored ones retried,
and a run whose prompt or model changed is refused rather than silently mixed. The prompt
digest is what shows a held-out run used the prompt frozen before it.

`--backend openai` targets any OpenAI-compatible endpoint, which covers both OpenRouter and
the local vLLM server. [`eval/pricing.py`](eval/pricing.py) prices only models with an
explicit verified entry, so `Cost / 1k` reads `-` for anything else rather than reporting a
borrowed rate.

### Training the adapter

The local 4GB GPU cannot fine-tune a 3B vision model, so everything deterministic runs
locally and is tested, while the training loop runs on Colab or Kaggle from a bundle this
repo produces.

```bash
python -m train.export        # data/interim/slotfill-bundle/ + .tar.gz (~135 MB)
```

The bundle holds every page image once (4,025: `train` is a subset of `train_4k`), one
`<split>.jsonl` per split, and a `bundle.json` carrying a SHA-256 digest per split.
`dev_unseen` travels marked `trainable: false` for the post-training smoke test; the GPU side
refuses to train on it and refuses to export a frozen set at all.

[`train/train_vlm.py`](train/train_vlm.py), driven by
[`train/colab_train.ipynb`](train/colab_train.ipynb) (`PLATFORM = 'colab' | 'kaggle'`), runs
an Unsloth vision LoRA on `Qwen2.5-VL-3B-Instruct` in 4-bit at `r=16, alpha=16, lr=2e-4`, 1
epoch, seed 3407, loss on the assistant turn only. Vision layers stay frozen: vLLM's LoRA
support for multimodal models covers the language model, and an adapter that cannot be
served is not a result.

| File | Contents |
|---|---|
| `adapter/` | LoRA weights and `adapter_config.json`, loadable by vLLM `--enable-lora` |
| `run_config.json` | every knob, base model, the **dataset digest** from the bundle, library versions, GPU, wall-clock training time |
| `smoke.json` | schema validity and field accuracy on 20 `dev_unseen` pages |
| `train_log.json` | the trainer's loss curve, which lives here and nowhere else |
| `trainer/checkpoint-N/` | a checkpoint every 50 optimizer steps; rerunning with the same `--out` resumes from the latest |

Adapter weights are not committed. `run_config.json`, `smoke.json` and `train_log.json` are,
under `runs/train/<adapter>/`. The digest chain runs `bundle.json` -> `run_config.json` ->
the held-out run's own config, so a number in the results table traces back to the documents
it was trained on.

### Evaluating the adapter

[`serve/serve_vllm.sh`](serve/serve_vllm.sh) starts vLLM with `--enable-lora` and puts both
conditions behind one endpoint, where the OpenAI `model` field selects base model or
adapter. `eval/predict.py --backend openai --base-url http://localhost:8000/v1` is the same
code path that reaches Gemini.

```bash
serve/serve_vllm.sh <adapter dir>          # base + adapter on :8000

python -m eval.predict --set test_unseen --input image \
    --backend openai --base-url http://localhost:8000/v1 --model slotfill-lora \
    --condition vision-adapter --no-example --no-schema --max-tokens 1024 --held-out

python -m eval.evaluate --gpu-usd-per-hour 0.35 runs/vision-adapter/test_unseen
```

`--no-example --no-schema` is the prompt the adapter was trained on (page image plus one
sentence), and the run config records a prompt digest distinct from the baseline's. The
baseline keeps its schema and worked example; that asymmetry is the experiment, since the
~1,450 tokens of schema the baseline re-sends per invoice are what the adapter holds in its
weights.

**Constrained decoding.** `--guided-json` sends `schema/invoice_schema.json` to vLLM as a
decoding constraint, so tokens that would break the schema are masked before sampling and an
invalid reply is unreachable. It needs `--backend openai`, and the run config records it as
an identity key, so a constrained run cannot resume or be mixed with an unconstrained one.
Schema validity under this setting describes the decoder rather than the model, and
`eval/evaluate.py` prints that caveat beside the row.

Nothing in a response confirms a constraint was applied, and a hosted OpenAI-compatible
endpoint accepts `guided_json` and ignores it, so the flag records a request rather than a
fact. The evidence is the output: under a working constraint no invalid output can be
emitted, so `eval/evaluate.py` reports any invalid output that was neither truncated nor a
request error as the server having ignored the constraint, and says the run is not
schema-constrained. `GUIDED_DECODING_BACKEND` on the serve script pins which implementation
vLLM uses, which matters because `currency` is constrained by a regex (`^[A-Z]{3}$`) rather
than an enum and regex support inside JSON schema differs between the backends.

For a self-hosted model there is no published per-token price, so each run records its own
wall clock in `sessions.jsonl`, `--gpu-usd-per-hour` turns that into
`cost / 1k = rate / measured throughput`, and `metrics.json` reports throughput beside it.
Without a rate the column reads `-`.

[`eval/colab_eval.ipynb`](eval/colab_eval.ipynb) runs this on a free Colab or Kaggle GPU. A
T4 is a 2018 card with no bfloat16 and vLLM's newer kernels increasingly assume newer
hardware, so the notebook sends a single request before the 1,083-document run and names a
paid L4 as the fallback.

---

## Design

### Schema

[`schema/invoice_schema.json`](schema/invoice_schema.json) is the correctness signal for the
whole project: 14 fields, `additionalProperties: false` at every level, and **every property
required**. Nullability is the `null` type rather than an absent key, so a missing key is
always a real error.

[`schema/validate.py`](schema/validate.py) turns a violation into a structured
`ValidationOutcome` carrying a dotted JSON path per error (`line_items.0.quantity`) rather
than a bare exception. It tolerates a markdown fence around model output and nothing else;
prose around the JSON counts as a failure, because a model chatting at the harness is a
failure mode worth counting.

`line_items` carries two kinds of nothing. It is `[]` when a line table was looked for and
not found, and `null` when the extractor does not produce line items at all. No dataset here
annotates line items, so the adapter is trained to say `null`, an abstention, rather than
`[]`, which would claim that a table a FATURA page plainly has is absent. The baseline reads
the table and fills the array. Both are schema-valid and neither is scored on it. Run
configs record a `schema_digest` beside the prompt digest, so a schema edit is visible as
one.

### Splits

Every split is keyed on the **layout id**. A document-level split would put images from one
template in both train and test, and the resulting accuracy would measure template
memorisation.

| Split | Layouts | Documents | Frozen | Purpose |
|---|---|---|---|---|
| `train` | 35 | 1,750 | no | fine-tuning |
| `dev_unseen` | 5 | 200 | no | prompt iteration on unseen layouts |
| `test_seen` | 35 | 350 | **yes** | same layouts as train, disjoint documents |
| `test_unseen` | 10 | 300 | **yes** | the 10 held-out layouts |
| RVL-CDIP | n/a | 433 | **yes** | real scans, zero-shot, shared fields only; 433 of 520, frozen in `data/splits/rvlcdip_sample.json` so every condition scores the same documents |
| `train_4k` | 35 | 4,025 | no | data-mix ablation: `train` plus 65 more documents per layout, same layouts, disjoint from every eval set |

FATURA ships `Strat2_Split.txt`, a 40/10 inter-template split that sets
`test_inds = dev_inds`, so its dev set is its test set. This repo leaves its 10 held-out
layouts untouched and carves a 5-layout dev set out of its 40 training layouts.

The manifests in [`data/splits/`](data/splits/) are committed and each carries a SHA-256
digest of its own contents. `write_manifests` refuses to change a frozen split once written,
and `load_manifest` refuses to load one edited by hand.

### FATURA field mapping

The released data has **35 annotation keys, not the 24 the paper tabulates**: the paper shows
seven `GST(<rate>%)` classes as a single row (-6), omits three `GSTIN*` classes (-3), and does
not tabulate `INVOICE_INFO` or `OTHER` (-2) though both appear in all 10,000 files. Every
annotated value arrives with its printed label attached (`"TOTAL : 972.30 EUR"`), so label
stripping and normalisation are the bulk of [`data/map_fatura.py`](data/map_fatura.py).

Fourteen classes map straight across, losing only the printed label: `AMOUNT_DUE`,
`CONDITIONS` -> `payment_terms`, `DATE` -> `invoice_date`, `DUE_DATE`, `NUMBER` ->
`invoice_number`, `PO_NUMBER` -> `purchase_order_number`, `SELLER_*` -> `vendor.*`,
`SUB_TOTAL` -> `subtotal`, `TAX` -> `tax`, `TOTAL` -> `total_amount`, and `DISCOUNT`, printed
as `(-) 51.2` and stored as a positive amount deducted.

The lossy and dropped mappings are below. Both tables are generated by
`python -m data.mapping_report`, which prints the full 35-class version, so neither can drift
from the code.

| FATURA class | Schema field | Note |
|---|---|---|
| `BILL_TO`, `BUYER`, `SEND_TO` | `buyer` | precedence BUYER > BILL_TO > SEND_TO; SHIP_TO is not the billed party |
| `GST(1%)`, `GST(5%)`, `GST(7%)`, `GST(9%)`, `GST(12%)`, `GST(18%)`, `GST(20%)` | `tax` | rate-parameterised; VAT wins when both appear on one document |
| `GSTIN`, `GSTIN_BUYER`, `GSTIN_SELLER` | _dropped_ | a tax registration number, not a tax amount - no schema field for it |
| `INVOICE_INFO` | _dropped_ | present as an empty list in all 10,000 files; carries no content |
| `LOGO` | _dropped_ | graphical element, not an extractable field |
| `NOTE` | _dropped_ | free-text remarks and footers with no schema counterpart |
| `OTHER` | _dropped_ | a text pass over the whole page; supervising on it would leak the answer, and it names a different vendor than the image on every document (it is the text-layer input, see `eval/datasets.py`) |
| `PAYMENT_DETAILS` | _dropped_ | bank account details; out of scope for this schema |
| `TABLE` | _dropped_ | bounding box only - no cell text is released, so line items cannot be labelled |
| `TITLE` | _dropped_ | document heading ('INVOICE', 'COMMERCIAL INVOICE'), not invoice data |
| `TOTAL_WORDS` | _dropped_ | the total spelled out in words - redundant with `total_amount` |

**`currency` is derived, not annotated.** It is read off the amount strings
(`"1309.36  EUR"`, `"379.82 $"`) and the most common code across a document's amounts wins.
FATURA's `$` is unqualified throughout and reads as `USD`.

**Coverage over the 1,750 training documents** runs from `invoice_date` at 97% down to
`amount_due` at 11%, with `purchase_order_number`, `payment_terms`, `discount` and
`vendor.website` all at 14-17%. The thin fields are worth holding in mind when reading a
per-field result. `python -m data.mapping_report` prints the per-field counts.

`line_items` is supervised **nowhere**: `TABLE` is a bare bounding box with no cell text in
any released annotation format. The mapper omits the key rather than emitting `[]`, which
would be a label nobody wrote. It stays in the schema because the API contract needs it, and
nothing trains or scores on it until a dataset with real line item annotations is available.

Three conflicts are counted rather than hidden:

| Conflict | Documents | Resolution |
|---|---|---|
| `tax:multiple_gst_rates` | 30 / 300 in `test_unseen` (Template25 only) | the page prints a **GST rate card**: five lines at 1/5/12/18/20% of the subtotal, and `TOTAL - SUB_TOTAL` matches none of them, so no printed figure is the tax charged. `tax` is left **unsupervised** for these documents |
| `tax:vat_and_gst` | 50 / 1,750 in `train` | a VAT line and a single GST line on one invoice; the schema has one `tax` field and takes the VAT line, which is a real printed charge. Summing would invent a number printed nowhere |
| `buyer:multiple_sources` | 150 / 1,750 in `train` | precedence `BUYER > BILL_TO > SEND_TO`; `SEND_TO` is a ship-to party, used only when nothing better exists |

### RVL-CDIP mapping

RVL-CDIP's ground truth is **region boxes over noisy OCR, not field values**. A `*_gt.xml`
file holds six labelled regions (`supplier`, `receiver`, `invoice_info`, `positions`,
`total`, `other`) and no text; the text comes from a separate ABBYY OCR pass over 1970s-90s
litigation scans. Across all 65,370 OCR words in the 520 documents the median per-word
confidence is **0.51** and the mean **0.52** (`python -m data.ocr_stats`).

A `total` region routinely holds a whole column of figures, and an `invoice_info` region
holds the invoice number, the date and their printed labels as one run of text. So RVL-CDIP
scores grounding and nothing stronger, and every other field is declared unscoreable rather
than handed an invented label.

| Schema field | Region | Scoring | Note |
|---|---|---|---|
| `vendor.name`, `vendor.address` | `supplier` | containment | the block mixes name, address and phone; not separable |
| `buyer.name`, `buyer.address` | `receiver` | containment | the block mixes name, address and attn line |
| `invoice_number`, `invoice_date`, `due_date` | `invoice_info` | containment | one run of text holding all three plus their labels |
| `total_amount` | `total` | containment | the region often holds a whole column of figures |
| `subtotal`, `tax`, `discount`, `amount_due`, `currency`, `payment_terms`, `purchase_order_number`, `line_items` | - | **not scoreable** | no annotation exists |

Region text recovered across the 520 documents: `supplier` 463, `receiver` 473,
`invoice_info` 480, `positions` 450, `total` 359, `other` 500.

### Training targets

The target for a page is the full schema record as one compact JSON string
([`train/targets.py`](train/targets.py)). Two rules bridge the mapper's partial record to the
schema's required-everything contract, both checked rather than assumed:

- **An unannotated field is `null`.** FATURA annotates every field its template prints: 34 of
  the 35 training layouts have exactly one supervised field set across all 50 of their
  documents, and layout 40 differs on a single document lacking a `DUE_DATE` annotation
  (`python -m train.target_report`, and the same holds over `train_4k`). Not annotated is
  therefore not printed, which is what `null` means here. This is a label the dataset wrote.
- **`line_items` is `null`**, never `[]` - see [Schema](#schema).

Every target is validated against the schema at export time, so a mapper bug cannot reach the
model. Keys are emitted in schema order and the JSON is compact, because output tokens are
what the adapter's cost per invoice is made of and formatting is invisible to the scorer.

The adapter's prompt is the page image followed by one fixed sentence,
`Extract the fields from this invoice page. Reply with the JSON object only.` - no schema and
no worked example. The schema lives in the weights, which is the cost argument in one line.

### Metrics

[`eval/evaluate.py`](eval/evaluate.py), identical for every condition:

| Metric | Definition |
|---|---|
| Schema validity | parses as JSON and validates; a markdown fence is tolerated, prose is not |
| Field accuracy (all) | correct field instances / scoreable ones, an invalid output wrong on every field. **The headline view** |
| Field accuracy (valid) | the same over schema-valid outputs only |
| Exact match | documents with every scoreable field right (FATURA only) |
| Cost / 1k invoices | measured tokens x published prices; self-hosted cost measured at the serving layer |
| p95 latency | nearest-rank over the run's successful requests |

Comparison rules live in [`eval/scoring.py`](eval/scoring.py): money agrees when rounded
half-up to the cent, dates and currency codes match exactly, and other strings match after
case folding and collapsing whitespace and commas, so a wrapped address matches the joined
label. Only fields the dataset annotated for that document are scored. `line_items` is never
scored. On RVL-CDIP, correct means grounded in the right annotated region and a null
prediction is an abstention, excluded from the denominator and reported separately, because a
region box cannot say whether a field was printed.

### Text-layer path

Neither dataset ships PDFs, so the text path consumes the text layer each dataset does ship:
FATURA's `OTHER` class and RVL-CDIP's ABBYY OCR words joined in reading order. FATURA's is a
stale text pass that names a different vendor than both the image and the `SELLER_NAME` label
on every document (0 of 360 agree; every other field agrees on 83-100%). `vendor.name` and
`vendor.website` are therefore unscoreable on the text path, and the evaluator prints a
per-field **text-layer ceiling** - the share of reference values present in the input text at
all - next to the text-path accuracy.

### Repository layout

```
schema/    invoice_schema.json, validate.py   the correctness signal
data/      download.py, sources.py            fetch + provenance
           map_fatura.py, map_rvlcdip.py      annotations -> schema
           split.py, build_splits.py          layout-id split
           splits/*.json                      frozen manifests (committed)
           mapping_report.py                  generates the mapping table above
train/     targets.py                         mapped record -> full schema record; leak guard
           target_report.py                   the "unannotated = not printed" measurement
           export.py                          images + targets -> self-describing bundle
           train_vlm.py                       Unsloth vision LoRA; run_config.json, smoke.json
           smoke.py                           post-training check on dev_unseen pages
           colab_train.ipynb                  thin notebook around train_vlm.py
eval/      predict.py                         run one condition on one set; resumable
           agreement.py                       bounds the RVL-CDIP OCR ceiling
           colab_eval.ipynb                   serve the adapter and score the held-out sets
           evaluate.py                        a run -> metrics.json + results row
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

## Limitations

- The adapter's headline weakness is **character-level transcription**, and more FATURA would
  not fix it: the misread strings are randomly generated place names no model can infer from
  context. A larger base model or a higher-resolution vision path is the lever.
- **Constrained decoding is unmeasured.** The RVL-CDIP validity gap (80.6% against 99.5%) is
  the single biggest input to the recommendation and is plausibly an artefact of
  unconstrained generation rather than of the model, so the recommendation for real scans is
  provisional until that is measured.
- **Latency is not comparable across conditions** as measured; only throughput is.
- Training data is **synthetic**. FATURA's content is generated, its layouts are clean, and
  its dates are internally inconsistent (due dates frequently precede invoice dates).
- The RVL-CDIP grounding numbers are **floors, not accuracy estimates**. They are comparable
  between conditions, which is what the results table uses them for, and should not be read
  as "the model got half the fields right".
- The real-document test set is **433 scans of one narrow domain** (tobacco litigation
  archives), with poor OCR, scoring grounding only.
- **No line item supervision exists** in either dataset.
- DocILE (6.7k real annotated invoice-like documents with an unseen-layout test split) is the
  upgrade path for both gaps. Access is gated behind a form and nothing here depends on it.

## Citations

- Limam et al., *FATURA: A Multi-Layout Invoice Image Dataset for Document Analysis and
  Understanding*, arXiv:2311.11856, 2023.
- Riba, Dutta, Goldmann, Fornes, Ramos, Llados, *Table Detection in Invoice Documents by
  Graph Neural Networks*, ICDAR 2019.
