"""Where the datasets come from, and under what terms.

Kept separate from the download logic so the provenance of every byte in `data/raw/`
is one short, readable file. Nothing here is committed to the repo except this record -
see the licence note in the README before redistributing any image.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetSource:
    key: str
    name: str
    url: str
    filename: str
    md5: str
    size_bytes: int
    doi: str
    licence: str
    citation: str


FATURA = DatasetSource(
    key="fatura",
    name="FATURA 2 - synthetic multi-layout invoice images",
    url="https://zenodo.org/api/records/10371464/files/FATURA2.zip/content",
    filename="FATURA2.zip",
    md5="4c9404462f22c5241eb1a290a02eb2a2",
    size_bytes=690_717_658,
    doi="10.5281/zenodo.10371464",
    licence="CC-BY-4.0",
    citation=(
        "Limam et al., 'FATURA: A Multi-Layout Invoice Image Dataset for Document "
        "Analysis and Understanding', arXiv:2311.11856, 2023."
    ),
)

RVLCDIP_INVOICE = DatasetSource(
    key="rvlcdip",
    name="Layout Analysis Groundtruth for the RVL-CDIP Dataset (520 real scanned invoices)",
    url="https://zenodo.org/api/records/3257319/files/dataset.zip/content",
    filename="rvlcdip_invoices.zip",
    md5="94bfcad49ed27f5a2fe9f0e4286106a3",
    size_bytes=35_640_391,
    doi="10.5281/zenodo.3257319",
    licence="CC-BY-4.0",
    citation=(
        "Riba, Dutta, Goldmann, Fornes, Ramos, Llados, 'Table Detection in Invoice "
        "Documents by Graph Neural Networks', ICDAR 2019."
    ),
)

SOURCES = {source.key: source for source in (FATURA, RVLCDIP_INVOICE)}
