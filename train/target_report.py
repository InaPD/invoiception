"""Is "not annotated" really "not printed"? Measure it rather than assert it.

    python -m train.target_report [--split train]

For every training layout, count the distinct supervised field sets across its documents.
A layout with exactly one set prints a fixed set of fields, so an absent annotation on any
of its documents means the field is not on the page - and `null` in the training target
is the dataset's own label. The README quotes this output.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from eval.datasets import load_fatura


def report(split: str) -> str:
    per_layout: dict[int, Counter[frozenset[str]]] = defaultdict(Counter)
    for item in load_fatura(split):
        per_layout[item.reference.layout_id][item.reference.supervised_fields] += 1

    constant = [layout for layout, sets in per_layout.items() if len(sets) == 1]
    lines = [
        f"{split}: {len(per_layout)} layouts, {len(constant)} with one supervised field set "
        "across every document"
    ]
    for layout, sets in sorted(per_layout.items()):
        if len(sets) == 1:
            continue
        common = frozenset.intersection(*sets)
        variants = ", ".join(
            f"{n} docs {'+' + '/'.join(sorted(s - common)) if s - common else 'baseline'}"
            for s, n in sets.most_common()
        )
        lines.append(f"  layout {layout}: {variants}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    args = parser.parse_args(argv)
    print(report(args.split))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
