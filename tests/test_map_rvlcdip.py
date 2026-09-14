"""RVL-CDIP mapper.

The dataset annotates *regions* (supplier / receiver / invoice_info / positions / total /
other) over noisy OCR - it does not annotate field values. These tests pin down the honest
consequence: the mapper returns region text spans plus an explicit statement of which
schema fields are scoreable and how. Anything that pretends to be a field value here
would be a fabricated label.
"""

import textwrap

import pytest

from data.map_rvlcdip import (
    RVLCDIP_REGIONS,
    ScoringMode,
    field_scoring,
    parse_gt,
    parse_ocr,
    region_text,
    to_reference,
    words_in_region,
)

GT_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="utf-8"?>
    <PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
      <Page imageFilename="doc1.tif" imageHeight="1000" imageWidth="800">
        <TextRegion id="r1">
          <Property key="entity" value="supplier"/>
          <Coords points="0,0 200,0 200,100 0,100"/>
        </TextRegion>
        <TextRegion id="r2">
          <Property key="entity" value="total"/>
          <Coords points="400,500 700,500 700,560 400,560"/>
        </TextRegion>
        <TextRegion id="r3">
          <Property key="entity" value="other"/>
          <Coords points="0,900 700,900 700,980 0,980"/>
        </TextRegion>
      </Page>
    </PcGts>
""")

OCR_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="utf-8"?>
    <PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
      <Page imageFilename="doc1.tif" imageHeight="1000" imageWidth="800">
        <TextRegion id="b1">
          <TextLine id="l1">
            <Word id="w1"><Coords points="10,10 60,10 60,24 10,24"/>
              <TextEquiv conf="0.9"><Unicode>Acme</Unicode></TextEquiv></Word>
            <Word id="w2"><Coords points="70,10 140,10 140,24 70,24"/>
              <TextEquiv conf="0.9"><Unicode>Supplies</Unicode></TextEquiv></Word>
          </TextLine>
          <TextLine id="l2">
            <Word id="w3"><Coords points="10,40 120,40 120,54 10,54"/>
              <TextEquiv conf="0.8"><Unicode>Woking</Unicode></TextEquiv></Word>
          </TextLine>
          <TextLine id="l3">
            <Word id="w4"><Coords points="410,510 500,510 500,530 410,530"/>
              <TextEquiv conf="0.7"><Unicode>AMOUNT</Unicode></TextEquiv></Word>
            <Word id="w5"><Coords points="520,510 640,510 640,530 520,530"/>
              <TextEquiv conf="0.7"><Unicode>$11,780.63</Unicode></TextEquiv></Word>
          </TextLine>
          <TextLine id="l4">
            <Word id="w6"><Coords points="10,920 90,920 90,940 10,940"/>
              <TextEquiv conf="0.6"><Unicode>Footer</Unicode></TextEquiv></Word>
          </TextLine>
        </TextRegion>
      </Page>
    </PcGts>
""")


@pytest.fixture
def doc(tmp_path):
    (tmp_path / "doc1_gt.xml").write_text(GT_XML, encoding="utf-8")
    (tmp_path / "doc1_ocr.xml").write_text(OCR_XML, encoding="utf-8")
    (tmp_path / "doc1.tif").write_bytes(b"not-a-real-tif")
    return tmp_path


class TestParsing:
    def test_parses_the_six_annotated_entities_only(self):
        assert RVLCDIP_REGIONS == (
            "supplier",
            "receiver",
            "invoice_info",
            "positions",
            "total",
            "other",
        )

    def test_parses_regions_with_entity_and_bbox(self, doc):
        regions = parse_gt(doc / "doc1_gt.xml")
        assert [r.entity for r in regions] == ["supplier", "total", "other"]
        assert regions[0].bbox == (0.0, 0.0, 200.0, 100.0)

    def test_parses_words_with_text_and_confidence(self, doc):
        words = parse_ocr(doc / "doc1_ocr.xml")
        assert [w.text for w in words] == [
            "Acme",
            "Supplies",
            "Woking",
            "AMOUNT",
            "$11,780.63",
            "Footer",
        ]
        assert words[0].confidence == pytest.approx(0.9)

    def test_skips_words_with_no_unicode_content(self, tmp_path):
        xml = OCR_XML.replace("<Unicode>Woking</Unicode>", "<Unicode></Unicode>")
        path = tmp_path / "e_ocr.xml"
        path.write_text(xml, encoding="utf-8")
        assert "Woking" not in [w.text for w in parse_ocr(path)]


class TestMalformedInput:
    def test_empty_coords_raise_a_clear_error(self, tmp_path):
        from data.map_rvlcdip import parse_gt

        xml = GT_XML.replace('points="0,0 200,0 200,100 0,100"', 'points=""')
        path = tmp_path / "bad_gt.xml"
        path.write_text(xml, encoding="utf-8")
        with pytest.raises(ValueError, match="Coords"):
            parse_gt(path)


class TestRegionAssignment:
    def test_assigns_words_by_centre_containment(self, doc):
        regions = parse_gt(doc / "doc1_gt.xml")
        words = parse_ocr(doc / "doc1_ocr.xml")
        supplier = next(r for r in regions if r.entity == "supplier")
        assert [w.text for w in words_in_region(words, supplier)] == ["Acme", "Supplies", "Woking"]

    def test_a_word_outside_every_region_is_dropped(self, doc):
        regions = [r for r in parse_gt(doc / "doc1_gt.xml") if r.entity == "supplier"]
        words = parse_ocr(doc / "doc1_ocr.xml")
        assigned = {w.text for r in regions for w in words_in_region(words, r)}
        assert "$11,780.63" not in assigned

    def test_region_text_is_in_reading_order(self, doc):
        regions = parse_gt(doc / "doc1_gt.xml")
        words = parse_ocr(doc / "doc1_ocr.xml")
        supplier = next(r for r in regions if r.entity == "supplier")
        assert region_text(words_in_region(words, supplier)) == "Acme Supplies Woking"

    def test_reading_order_is_top_to_bottom_then_left_to_right(self):
        from data.map_rvlcdip import OcrWord

        words = [
            OcrWord("second", (0.0, 50.0, 40.0, 64.0), 1.0),
            OcrWord("right", (200.0, 10.0, 260.0, 24.0), 1.0),
            OcrWord("left", (10.0, 12.0, 60.0, 26.0), 1.0),
        ]
        assert region_text(words) == "left right second"


class TestReference:
    def test_builds_one_reference_per_document(self, doc):
        ref = to_reference(doc, "doc1")
        assert ref.doc_id == "doc1"
        assert ref.regions["supplier"] == "Acme Supplies Woking"
        assert "$11,780.63" in ref.regions["total"]

    def test_absent_regions_are_none_not_empty_string(self, doc):
        ref = to_reference(doc, "doc1")
        assert ref.regions["receiver"] is None
        assert ref.regions["invoice_info"] is None

    def test_other_region_is_carried_but_never_scoreable(self, doc):
        ref = to_reference(doc, "doc1")
        assert ref.regions["other"] == "Footer"
        assert field_scoring("other") is None

    def test_reference_is_immutable(self, doc):
        ref = to_reference(doc, "doc1")
        with pytest.raises((AttributeError, TypeError)):
            ref.doc_id = "tampered"

    def test_records_mean_ocr_confidence(self, doc):
        ref = to_reference(doc, "doc1")
        assert 0.0 < ref.mean_ocr_confidence < 1.0


class TestScoringDeclaration:
    """The mapper must be explicit that no schema field gets exact-match credit here."""

    @pytest.mark.parametrize(
        "field,region",
        [
            ("vendor.name", "supplier"),
            ("vendor.address", "supplier"),
            ("buyer.name", "receiver"),
            ("buyer.address", "receiver"),
            ("invoice_number", "invoice_info"),
            ("invoice_date", "invoice_info"),
            ("due_date", "invoice_info"),
            ("total_amount", "total"),
        ],
    )
    def test_scoreable_fields_map_to_their_region(self, field, region):
        scoring = field_scoring(field)
        assert scoring is not None
        assert scoring.region == region
        assert scoring.mode is ScoringMode.CONTAINMENT

    @pytest.mark.parametrize(
        "field",
        [
            "subtotal",
            "tax",
            "discount",
            "amount_due",
            "currency",
            "payment_terms",
            "purchase_order_number",
            "line_items",
        ],
    )
    def test_unscoreable_fields_are_declared_unscoreable(self, field):
        assert field_scoring(field) is None, (
            "RVL-CDIP has no annotation for this field; scoring it would invent a label"
        )

    def test_no_field_is_exact_match_scoreable(self):
        from data.map_rvlcdip import FIELD_SCORING

        assert all(s.mode is ScoringMode.CONTAINMENT for s in FIELD_SCORING.values())
