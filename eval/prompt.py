"""The prompt every condition sees, and the one worked example it comes with.

This is the STRONG baseline the plan asks for: the full schema, explicit conventions, and
a worked example on a real training-split page. Beating a lazy prompt proves nothing.

The worked example is `Template48_Instance97` from the **train** split, chosen as the
training document with the widest field coverage. Its record below was written by hand
from the page image (line items included: FATURA does not annotate them, but a prompt
example is not training data). The text-path example is a faithful transcription of the
same page rather than FATURA's shipped text layer, which names the wrong vendor.

`prompt_digest()` fingerprints all of this. `predict.py` writes it into every run config,
so a held-out run can be shown to have used the prompt frozen before the run - and not
one tweaked afterwards.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Literal

from eval.backends import DEFAULT_MAX_TOKENS, Message, Part, Request, TextPart
from eval.pages import ImagePart
from schema.validate import load_schema

InputKind = Literal["image", "text"]

WORKED_EXAMPLE_DOC_ID = "Template48_Instance97"

INSTRUCTION = "Extract the fields from this invoice page. Reply with the JSON object only."

_RULES = """\
You extract structured data from one invoice page. Reply with exactly one JSON object that \
validates against the schema below - no prose, no markdown fence, no comments.

Conventions:
- Every key in the schema is required. Use null for anything the page does not state. Never guess.
- Dates are ISO 8601 calendar dates (YYYY-MM-DD), whatever format the page prints.
- Money fields are numbers, not strings: no currency symbols, no thousands separators. \
`discount` is the positive amount deducted.
- `currency` is the ISO 4217 code for the symbol or code printed next to the amounts \
("$" is USD). Null if none is printed.
- `invoice_number` and `purchase_order_number` are copied as printed, prefixes and punctuation \
included, without their label.
- `vendor` is the party issuing the invoice; `buyer` is the party billed. Names without their \
label; addresses on one line, comma separated, without phone, email or website lines.
- `total_amount` is the grand total. `amount_due` is only set when a separate balance-due figure \
is printed.
- `tax` is the tax amount charged; sales tax, VAT and GST all belong there.
- `payment_terms` is the printed terms text, or null.
- `line_items` has one entry per billed row of the line table, in document order. \
Use [] when the page has no line table.

Schema:
"""

#: Hand-written from the page image. See the module docstring.
EXAMPLE_RECORD = {
    "invoice_number": "INV/79-83/438",
    "purchase_order_number": None,
    "invoice_date": "2013-08-08",
    "due_date": "2020-05-09",
    "vendor": {
        "name": "Brooks LLC",
        "address": "9950 Santos Squares, Garzafurt, MH 31937 US",
        "email": "ospencer@example.com",
        "website": "www.BrooksLLC.com",
    },
    "buyer": {
        "name": "Hannah Kim",
        "address": "7094 Joseph Ports Suite 501, Amandaville, AR 61257 US",
    },
    "currency": "EUR",
    "subtotal": 381.59,
    "discount": None,
    "tax": 34.34,
    "total_amount": 392.91,
    "amount_due": None,
    "payment_terms": None,
    "line_items": [
        {"description": "Son.", "quantity": 6.0, "unit_price": 3.69, "amount": 22.14},
        {"description": "Over field lot.", "quantity": 5.0, "unit_price": 71.89, "amount": 359.45},
    ],
}

#: Faithful transcription of the example page, in reading order.
EXAMPLE_TEXT = """\
INVOICE # INV/79-83/438
Invoice Date: 08-Aug-2013
Due Date : 09-May-2020

Brooks LLC
Address:9950 Santos Squares
Garzafurt, MH 31937 US
Email:ospencer@example.com
www.BrooksLLC.com

Bill to:Hannah Kim
7094 Joseph Ports Suite 501
Amandaville, AR 61257 US
Tel:+(030)649-0275
Email:tracisimmons@example.com
Site:http://www.glover-sanchez.com/

Qty ID Description Unit price Amount
6.00 532128 Son. 3.69 22.14
5.00 018895 Over field lot. 71.89 359.45

Bank Name Central Bank of USA
Branch Name Raf CAMP
Bank Account Number 11958841
Bank Swift Code SBININBB250

SUB_TOTAL : 381.59 EUR
GST(9%) : 34.34
BALANCE DUE : 392.91 EUR

Total in words: three hundred and ninety-two point nine one

Note: All payments to be made in cash.
Contact us for queries on these quotations.
"""


@lru_cache(maxsize=1)
def system_prompt() -> str:
    return _RULES + json.dumps(load_schema(), indent=1, sort_keys=True)


def text_part(page_text: str) -> TextPart:
    """Wrap a page's text layer so the model can tell it from the instruction."""
    return TextPart(f"<page_text>\n{page_text.rstrip()}\n</page_text>")


def worked_example(
    input_kind: InputKind, *, image: ImagePart | None = None
) -> tuple[Message, Message]:
    """The example exchange. For the image path the caller supplies the encoded page."""
    if input_kind == "image":
        if image is None:
            raise ValueError("the image-path worked example needs its page image")
        example_input: Part = image
    else:
        example_input = text_part(EXAMPLE_TEXT)
    answer = json.dumps(EXAMPLE_RECORD, indent=1, ensure_ascii=False)
    return (
        Message("user", (example_input, TextPart(INSTRUCTION))),
        Message("assistant", (TextPart(answer),), cache_breakpoint=True),
    )


def build_request(
    target: Part,
    *,
    input_kind: InputKind,
    example_image: ImagePart | None = None,
    with_example: bool = True,
    effort: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Request:
    messages: list[Message] = []
    if with_example:
        messages.extend(worked_example(input_kind, image=example_image))
    messages.append(Message("user", (target, TextPart(INSTRUCTION))))
    return Request(
        system=system_prompt(), messages=tuple(messages), max_tokens=max_tokens, effort=effort
    )


def prompt_digest(input_kind: InputKind, *, with_example: bool = True) -> str:
    """SHA-256 of everything that shapes the prompt, for the run config."""
    payload = {
        "system": system_prompt(),
        "instruction": INSTRUCTION,
        "input_kind": input_kind,
        "example": (
            {
                "doc_id": WORKED_EXAMPLE_DOC_ID,
                "record": EXAMPLE_RECORD,
                "text": EXAMPLE_TEXT if input_kind == "text" else None,
            }
            if with_example
            else None
        ),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
