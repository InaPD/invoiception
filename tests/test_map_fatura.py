"""FATURA mapper: 35 annotated classes -> the slotfill schema.

Every annotated value in FATURA arrives with its printed label attached
("TOTAL : 972.30 EUR", "Bill to:Robert Wheeler\\n..."), so the mapper's real job is
label stripping and normalisation. These tests use the exact strings found in the
released annotations, not invented ones.
"""

import pytest

from data.map_fatura import (
    DROPPED_CLASSES,
    FATURA_CLASSES,
    MappingReport,
    layout_id,
    map_annotation,
    parse_amount,
    parse_currency,
    parse_date,
    parse_party,
    strip_label,
)


class TestClassVocabulary:
    def test_vocabulary_matches_the_released_dataset(self):
        assert len(FATURA_CLASSES) == 35, "the released data has 35 keys, not the paper's 24"
        for key in ("GST(18%)", "GSTIN_SELLER", "INVOICE_INFO", "OTHER", "TABLE"):
            assert key in FATURA_CLASSES

    def test_every_class_is_either_mapped_or_explicitly_dropped(self):
        from data.map_fatura import CLASS_TO_FIELD

        assert set(CLASS_TO_FIELD) | set(DROPPED_CLASSES) == set(FATURA_CLASSES)
        assert not set(CLASS_TO_FIELD) & set(DROPPED_CLASSES)

    def test_every_dropped_class_has_a_written_reason(self):
        assert all(reason.strip() for reason in DROPPED_CLASSES.values())

    def test_tax_identifiers_are_dropped_not_folded_into_tax(self):
        """GSTIN is a registration number, not an amount. Folding it in would be a bug."""
        for key in ("GSTIN", "GSTIN_SELLER", "GSTIN_BUYER"):
            assert key in DROPPED_CLASSES


class TestLayoutId:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("Template17_Instance112.json", 17),
            ("Template1_Instance0.jpg", 1),
            ("Template50_Instance199", 50),
        ],
    )
    def test_extracts_template_number(self, name, expected):
        assert layout_id(name) == expected

    def test_rejects_a_name_with_no_template(self):
        with pytest.raises(ValueError):
            layout_id("invoice_42.json")


class TestStripLabel:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("INVOICE ID INV/08-29/201", "INV/08-29/201"),
            ("INVOICE # 6608-592", "6608-592"),
            ("Invoice number INV/90-12/743", "INV/90-12/743"),
            ("INVOICE ID INV61200588", "INV61200588"),
        ],
    )
    def test_strips_invoice_number_labels(self, raw, expected):
        assert strip_label(raw, "NUMBER") == expected

    def test_strips_po_number_label(self):
        assert strip_label("PO Number :02", "PO_NUMBER") == "02"

    def test_strips_address_label_and_joins_lines(self):
        raw = "Address:19021 Christopher Estate\nSouth Williammouth, NH 22397 US"
        assert strip_label(raw, "SELLER_ADDRESS") == (
            "19021 Christopher Estate, South Williammouth, NH 22397 US"
        )

    def test_strips_email_label(self):
        assert strip_label("Email:lewisjames@example.com", "SELLER_EMAIL") == (
            "lewisjames@example.com"
        )

    def test_leaves_an_unlabelled_class_alone(self):
        assert strip_label("Buchanan and Sons", "SELLER_NAME") == "Buchanan and Sons"


class TestParseAmount:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("TOTAL : 972.30 EUR", 972.30),
            ("BALANCE DUE : 774.49 EUR", 774.49),
            ("SUB_TOTAL : 1309.36  EUR", 1309.36),
            ("AMOUNT_DUE : 606.69 EUR", 606.69),
            ("DUE_AMOUNT : 1299.34 USD", 1299.34),
            ("TOTAL : 379.82 $", 379.82),
        ],
    )
    def test_parses_plain_amounts(self, raw, expected):
        assert parse_amount(raw) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("TAX:VAT (4.96%):  65.01 EUR", 65.01),
            ("TAX:VAT (6.38%):  29.48 $", 29.48),
            ("GST(18%) : 116.35", 116.35),
            ("GST(1%) : 3.81", 3.81),
        ],
    )
    def test_ignores_the_percentage_and_takes_the_amount(self, raw, expected):
        """The rate in brackets is the classic off-by-one-field bug here."""
        assert parse_amount(raw) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "raw,expected",
        [("DISCOUNT(3.91%): (-)  51.2", 51.2), ("DISCOUNT(2.28%): (-)  23.94", 23.94)],
    )
    def test_discount_is_returned_as_a_positive_amount(self, raw, expected):
        assert parse_amount(raw) == pytest.approx(expected)

    def test_handles_thousands_separators(self):
        assert parse_amount("TOTAL : 1,309.36 USD") == pytest.approx(1309.36)

    def test_preserves_a_negative_sign(self):
        """Credit notes exist. Forcing every amount positive would silently flip them."""
        assert parse_amount("TOTAL : -51.20 USD") == pytest.approx(-51.20)

    def test_returns_none_when_there_is_no_amount(self):
        assert parse_amount("Thank you for your business!") is None


class TestParseCurrency:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("TOTAL : 972.30 EUR", "EUR"),
            ("SUB_TOTAL : 1309.36  EUR", "EUR"),
            ("TOTAL : 1338.24 USD", "USD"),
            ("TAX:VAT (6.38%):  29.48 $", "USD"),
        ],
    )
    def test_reads_code_or_symbol(self, raw, expected):
        assert parse_currency(raw) == expected

    def test_returns_none_when_unmarked(self):
        assert parse_currency("GST(18%) : 116.35") is None


class TestParseDate:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Invoice Date: 22-Jun-1998", "1998-06-22"),
            ("Invoice Date: 01-Oct-2007", "2007-10-01"),
            ("Due Date : 03-May-2018", "2018-05-03"),
            ("Due Date : 01-Feb-1998", "1998-02-01"),
        ],
    )
    def test_normalises_to_iso(self, raw, expected):
        assert parse_date(raw) == expected

    def test_returns_none_on_an_unparseable_date(self):
        assert parse_date("Invoice Date: sometime last spring") is None


class TestParseParty:
    BUYER = (
        "Bill to:Robert Wheeler\n08328 Joshua Hill Suite 079\nPetersenburgh, NC 11673 US\n"
        "Tel:+(483)711-0481\nEmail:nowens@example.com\nSite:https://fitzgerald.com/"
    )

    def test_splits_name_from_address(self):
        party = parse_party(self.BUYER, "BUYER")
        assert party.name == "Robert Wheeler"
        assert party.address == "08328 Joshua Hill Suite 079, Petersenburgh, NC 11673 US"

    def test_drops_contact_lines_from_the_address(self):
        party = parse_party(self.BUYER, "BUYER")
        assert "Tel:" not in (party.address or "")
        assert "Email:" not in (party.address or "")
        assert "Site:" not in (party.address or "")

    def test_handles_the_bill_to_block_layout(self):
        raw = (
            "BILL_TO: \nSarah Rich \n15628 Patel Fords Apt. 449 \nWest Carl, MI 46650 US \n"
            "Tel:+(893)270-8264 \nEmail:dawn38@example.net"
        )
        party = parse_party(raw, "BILL_TO")
        assert party.name == "Sarah Rich"
        assert party.address == "15628 Patel Fords Apt. 449, West Carl, MI 46650 US"

    def test_handles_the_buyer_colon_variant(self):
        raw = "Buyer :Theresa Moreno\n745 Shaw Field Apt. 378\nEast Michael, HI 43104 US"
        assert parse_party(raw, "BUYER").name == "Theresa Moreno"


class TestMapAnnotation:
    ANNOTATION = {
        "TABLE": [[{"bbox": [[7.0, 416.0], [557.0, 326.0]]}]],
        "INVOICE_INFO": [],
        "NUMBER": {"text": "INVOICE # INV/78-64/532"},
        "DATE": {"text": "Invoice Date: 14-Apr-2022"},
        "DUE_DATE": {"text": "Due Date : 19-Mar-2002"},
        "SELLER_NAME": {"text": "Berg, Ward and Wright"},
        "SELLER_ADDRESS": {"text": "Address:5776 Whitney Meadow\nWest Adriennebury, ID 40049 US"},
        "SUB_TOTAL": {"text": "SUB_TOTAL : 885.25  USD"},
        "TAX": {"text": "TAX:VAT (3.22%):  28.51 USD"},
        "TOTAL": {"text": "TOTAL : 870.74 USD"},
        "GSTIN_SELLER": {"text": "GSTIN O4AROPO98743125,"},
        "TOTAL_WORDS": {"text": "Total in words:  eight hundred"},
        "TITLE": {"text": "INVOICE"},
        "OTHER": {"text": "the whole page"},
    }

    def test_maps_the_scalar_fields(self):
        mapped = map_annotation(self.ANNOTATION, "Template17_Instance112.json")
        record = mapped.record
        assert record["invoice_number"] == "INV/78-64/532"
        assert record["invoice_date"] == "2022-04-14"
        assert record["due_date"] == "2002-03-19"
        assert record["subtotal"] == pytest.approx(885.25)
        assert record["tax"] == pytest.approx(28.51)
        assert record["total_amount"] == pytest.approx(870.74)

    def test_derives_currency_from_the_amount_strings(self):
        assert map_annotation(self.ANNOTATION, "Template17_Instance112.json").record[
            "currency"
        ] == ("USD")

    def test_carries_the_layout_id(self):
        assert map_annotation(self.ANNOTATION, "Template17_Instance112.json").layout_id == 17

    def test_line_items_are_never_fabricated(self):
        """FATURA annotates the table as a single box with no cell text. An empty list
        here would be a manufactured label, so the key must be absent instead."""
        mapped = map_annotation(self.ANNOTATION, "Template17_Instance112.json")
        assert "line_items" not in mapped.record
        assert "line_items" not in mapped.supervised_fields

    def test_supervised_fields_lists_only_what_was_annotated(self):
        mapped = map_annotation(self.ANNOTATION, "Template17_Instance112.json")
        assert "invoice_number" in mapped.supervised_fields
        assert "payment_terms" not in mapped.supervised_fields
        assert "amount_due" not in mapped.supervised_fields

    def test_absent_class_yields_an_absent_field_not_a_null_label(self):
        mapped = map_annotation(self.ANNOTATION, "Template17_Instance112.json")
        assert "purchase_order_number" not in mapped.record

    def test_gst_rate_class_feeds_tax(self):
        mapped = map_annotation(
            {"OTHER": {"text": "x"}, "GST(18%)": {"text": "GST(18%) : 116.35"}},
            "Template2_Instance1.json",
        )
        assert mapped.record["tax"] == pytest.approx(116.35)

    def test_vat_wins_when_a_document_carries_both_vat_and_gst(self):
        annotation = {
            "OTHER": {"text": "x"},
            "TAX": {"text": "TAX:VAT (4.41%):  28.51 $"},
            "GST(18%)": {"text": "GST(18%) : 116.35"},
        }
        mapped = map_annotation(annotation, "Template29_Instance192.json")
        assert mapped.record["tax"] == pytest.approx(28.51), (
            "summing would invent a number printed nowhere on the page"
        )
        assert "tax:vat_and_gst" in mapped.conflicts

    def test_a_rate_card_of_several_gst_lines_leaves_tax_unsupervised(self):
        """Template25 prints GST at 1/5/12/18/20% of the subtotal - a rate card, not a
        charge. TOTAL minus SUB_TOTAL matches none of them, so no printed figure is the
        tax. Picking one would be a fabricated label."""
        annotation = {
            "OTHER": {"text": "x"},
            "SUB_TOTAL": {"text": "SUB_TOTAL : 901.21  USD"},
            "TOTAL": {"text": "TOTAL : 917.29 USD"},
            "GST(1%)": {"text": "GST(1%) : 9.01"},
            "GST(5%)": {"text": "GST(5%) : 45.06"},
            "GST(12%)": {"text": "GST(12%) : 108.15"},
            "GST(18%)": {"text": "GST(18%) : 162.22"},
            "GST(20%)": {"text": "GST(20%) : 180.24"},
        }
        mapped = map_annotation(annotation, "Template25_Instance11.json")
        assert "tax" not in mapped.record
        assert "tax" not in mapped.supervised_fields
        assert "tax:multiple_gst_rates" in mapped.conflicts

    def test_a_single_gst_line_still_supervises_tax(self):
        mapped = map_annotation(
            {"OTHER": {"text": "x"}, "GST(9%)": {"text": "GST(9%) : 66.30"}},
            "Template13_Instance4.json",
        )
        assert mapped.record["tax"] == pytest.approx(66.30)
        assert "tax:multiple_gst_rates" not in mapped.conflicts

    def test_an_explicit_vat_line_survives_a_gst_rate_card(self):
        annotation = {
            "OTHER": {"text": "x"},
            "TAX": {"text": "TAX:VAT (4.41%):  28.51 $"},
            "GST(1%)": {"text": "GST(1%) : 9.01"},
            "GST(18%)": {"text": "GST(18%) : 162.22"},
        }
        mapped = map_annotation(annotation, "Template29_Instance192.json")
        assert mapped.record["tax"] == pytest.approx(28.51), "VAT is a real printed charge"
        assert "tax:vat_and_gst" in mapped.conflicts

    def test_buyer_precedence_prefers_bill_to_over_ship_to(self):
        annotation = {
            "OTHER": {"text": "x"},
            "BILL_TO": {"text": "BILL_TO: \nSarah Rich \n1 High St"},
            "SEND_TO": {"text": "SHIP_TO: \nIan Mejia \n2 Low St"},
        }
        mapped = map_annotation(annotation, "Template9_Instance1.json")
        assert mapped.record["buyer"]["name"] == "Sarah Rich"
        assert "buyer:multiple_sources" in mapped.conflicts

    def test_conditions_supervises_payment_terms(self):
        annotation = {
            "OTHER": {"text": "x"},
            "CONDITIONS": {"text": "Terms & Conditions\nPayment within 8 days."},
        }
        mapped = map_annotation(annotation, "Template6_Instance3.json")
        assert mapped.record["payment_terms"] == "Payment within 8 days."
        assert "payment_terms" in mapped.supervised_fields

    def test_line_items_is_the_only_structurally_unsupervised_field(self):
        from data.map_fatura import UNSUPERVISED_FIELDS

        assert set(UNSUPERVISED_FIELDS) == {"line_items"}

    def test_unknown_class_is_a_loud_failure(self):
        with pytest.raises(ValueError, match="TOTALLY_NEW"):
            map_annotation({"TOTALLY_NEW": {"text": "?"}}, "Template1_Instance1.json")


class TestMappingReport:
    def test_counts_coverage_and_conflicts(self):
        report = MappingReport()
        report.observe(map_annotation(TestMapAnnotation.ANNOTATION, "Template17_Instance1.json"))
        report.observe(map_annotation(TestMapAnnotation.ANNOTATION, "Template17_Instance2.json"))
        assert report.documents == 2
        assert report.field_coverage["invoice_number"] == 2
        assert report.field_coverage["payment_terms"] == 0

    def test_renders_a_markdown_table_for_the_readme(self):
        report = MappingReport()
        report.observe(map_annotation(TestMapAnnotation.ANNOTATION, "Template17_Instance1.json"))
        markdown = report.to_markdown()
        assert "| FATURA class |" in markdown
        assert "GSTIN_SELLER" in markdown
