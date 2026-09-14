"""Validation of extracted invoice records against the strict target schema.

A schema violation is a structured error, never a silent pass and never a bare
exception: callers get an immutable `ValidationOutcome` they can count, log or
return to an API client.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.validators import validator_for

SCHEMA_PATH = Path(__file__).with_name("invoice_schema.json")

# Model output is allowed to arrive wrapped in a markdown fence and nothing else.
_FENCE = re.compile(r"\A\s*```(?:json)?\s*\n(?P<body>.*?)\n?\s*```\s*\Z", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ValidationError:
    """One violation, addressed by a dotted JSON path. `$` means the whole document."""

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    """Result of validating one record. Immutable so it cannot be massaged after the fact."""

    ok: bool
    errors: tuple[ValidationError, ...] | list[ValidationError] = field(default_factory=tuple)
    record: dict[str, Any] | None = None


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    """Load the target schema. Cached: it is read once per process and never mutated."""
    with SCHEMA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    schema = load_schema()
    cls = validator_for(schema)
    cls.check_schema(schema)
    # format_checker is what makes `format: date` reject 2021-02-30 rather than shrug at it.
    return cls(schema, format_checker=cls.FORMAT_CHECKER)


def _json_path(error) -> str:
    """Render a jsonschema error path as `line_items.0.quantity`, or `$` for the root."""
    parts = [str(p) for p in error.absolute_path]
    return ".".join(parts) if parts else "$"


def validate_record(record: Any) -> ValidationOutcome:
    """Validate an already-parsed record. Reports every violation, not just the first."""
    if not isinstance(record, dict):
        return ValidationOutcome(
            ok=False,
            errors=[ValidationError("$", f"expected a JSON object, got {type(record).__name__}")],
        )

    errors = [
        ValidationError(_json_path(e), e.message)
        for e in sorted(_validator().iter_errors(record), key=lambda e: list(e.absolute_path))
    ]
    if errors:
        return ValidationOutcome(ok=False, errors=errors)
    return ValidationOutcome(ok=True, errors=[], record=record)


def strip_fence(text: str) -> str:
    """Remove a surrounding markdown fence. Anything else is left alone, and so will fail."""
    match = _FENCE.match(text)
    return match.group("body") if match else text


def parse_and_validate(text: str) -> ValidationOutcome:
    """Parse raw model output and validate it.

    Tolerates a markdown fence around the JSON. Prose before or after the object is a
    failure, deliberately: chatty output is a failure mode the eval needs to be able to count.
    """
    body = strip_fence(text).strip()
    if not body:
        return ValidationOutcome(ok=False, errors=[ValidationError("$", "empty output, no JSON")])

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        return ValidationOutcome(
            ok=False,
            errors=[
                ValidationError("$", f"output is not valid JSON: {exc.msg} (line {exc.lineno})")
            ],
        )

    return validate_record(parsed)
