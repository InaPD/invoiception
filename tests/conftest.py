"""Shared fixtures. Keeps every test honest about what a *complete* record looks like."""

import copy

import pytest


@pytest.fixture
def valid_record():
    """A minimal record that satisfies every `required` key in the schema."""
    return {
        "invoice_number": "INV-0042",
        "purchase_order_number": None,
        "invoice_date": "2021-03-04",
        "due_date": "2021-04-03",
        "vendor": {
            "name": "Acme Supplies Ltd",
            "address": "12 Mill Road, Leeds LS1 1AA",
            "email": "billing@acme.example",
            "website": None,
        },
        "buyer": {"name": "Globex Inc", "address": "8 Harbour Way, Bristol BS1 5TY"},
        "currency": "USD",
        "subtotal": 1200.0,
        "discount": None,
        "tax": 240.0,
        "total_amount": 1440.0,
        "amount_due": None,
        "payment_terms": "Net 30",
        "line_items": [
            {"description": "Widget, large", "quantity": 4.0, "unit_price": 300.0, "amount": 1200.0}
        ],
    }


@pytest.fixture
def mutate(valid_record):
    """Return a copy of the valid record with one key changed, without touching the original."""

    def _mutate(**overrides):
        record = copy.deepcopy(valid_record)
        record.update(overrides)
        return record

    return _mutate
