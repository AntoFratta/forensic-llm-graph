"""Regression checks for rules, span boundaries and preprocessed NLP input.

Small stand-ins represent spaCy output without requiring a model download.
These checks validate extraction logic, not the model's recognition quality.
"""

from types import SimpleNamespace
import unittest

from src.deterministic import extract_entities


class NLPSpan:
    def __init__(self, text, value, label, pos="PROPN"):
        self.start_char = text.index(value)
        self.end_char = self.start_char + len(value)
        self.text = value
        self.label_ = label
        self.pos = pos

    def __iter__(self):
        return iter([SimpleNamespace(pos_=self.pos)])


class NLPDocument:
    def __init__(self, text, entities=()):
        self.text = text
        self.ents = list(entities)

    def has_annotation(self, name):
        return name == "POS"


class ExtractionTests(unittest.TestCase):
    def assert_offsets(self, text, entities):
        for entity in entities:
            self.assertEqual(text[entity["start"]:entity["end"]], entity["text"])

    def test_explicit_numeric_fiscal_codes_and_vat_fields_remain_distinct(self):
        text = "Codice fiscale 9873462111. C.F. 10112312312. Partita IVA n. 23874333312."
        fiscal = [e for e in extract_entities(text) if e["label"] in {"TAX_CODE", "VAT_NUMBER"}]
        self.assertEqual(
            [(e["text"], e["label"], e["format_valid"]) for e in fiscal],
            [("9873462111", "TAX_CODE", False), ("10112312312", "TAX_CODE", True),
             ("23874333312", "VAT_NUMBER", True)],
        )
        self.assert_offsets(text, fiscal)

    def test_paired_form_fields_keep_separate_values_in_both_orders(self):
        for fields, expected in [
            ("Cognome: De Luca Nome: Anna Maria", ["De Luca", "Anna Maria"]),
            ("Nome Anna Maria Cognome D' Andrea", ["Anna Maria", "D' Andrea"]),
        ]:
            with self.subTest(fields=fields):
                text = f"Il sottoscritto / a : {fields} nato / a a Torino"
                entities = extract_entities(text)
                people = [e for e in entities if e["label"] == "PERSON"]
                self.assertEqual([e["text"] for e in people], expected)
                self.assertTrue(all(e["needs_review"] for e in people))
                self.assert_offsets(text, entities)

    def test_field_labels_are_removed_from_conflicting_nlp_span(self):
        text = "Il sottoscritto Cognome Bianchi Nome Elena Codice Fiscale"
        prediction = NLPSpan(text, "Cognome Bianchi Nome Elena", "LOC")
        entities = extract_entities(text, doc=NLPDocument(text, [prediction]))
        self.assertEqual([(e["text"], e["label"]) for e in entities],
                         [("Bianchi", "PERSON"), ("Elena", "PERSON")])
        for entity in entities:
            self.assertEqual(entity["alternative_predictions"][0]["text"], prediction.text)
            self.assertIn("form_name_prediction_adjusted", entity["review_reasons"])
        self.assert_offsets(text, entities)

    def test_unexplained_nlp_words_are_retained_for_review(self):
        text = "Cognome Bianchi Nome Elena nato a Roma"
        prediction = NLPSpan(text, "Cognome Bianchi Nome Elena nato", "PER")
        entities = extract_entities(text, doc=NLPDocument(text, [prediction]))
        original = next(e for e in entities if e["text"] == prediction.text)
        self.assertIn("possible_form_fields_in_name", original["review_reasons"])
        self.assertIn("overlapping_candidates", original["review_reasons"])

    def test_unlabelled_form_name_still_reconciles_sex_fields(self):
        text = "Il sottoscritto Marco Verdi M F nato a Torino"
        doc = NLPDocument(text, [NLPSpan(text, "Marco Verdi M F", "LOC")])
        entities = extract_entities(text, doc=doc)
        self.assertEqual([(e["text"], e["label"]) for e in entities],
                         [("Marco Verdi", "PERSON")])

    def test_comma_alone_does_not_complete_an_unlabelled_person_field(self):
        text = "Il sottoscritto Avvocato Anna Bianchi, dichiara quanto segue."
        self.assertEqual(extract_entities(text), [])

    def test_residence_prefix_does_not_extend_an_address(self):
        text = "Residente in Torino via Giuseppe Verdi n. 12"
        doc = NLPDocument(text, [NLPSpan(text, "Residente in Torino", "LOC"),
                                 NLPSpan(text, "Giuseppe Verdi", "PER")])
        entities = extract_entities(text, doc=doc)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]["text"], "Torino via Giuseppe Verdi n. 12")
        absorbed = entities[0]["absorbed_entities"]
        self.assertEqual({e["text"] for e in absorbed}, {"Torino", "Giuseppe Verdi"})
        self.assert_offsets(text, entities)

    def test_dates_remain_as_written_and_identifiers_are_protected(self):
        text = "Date 5/15/21 e 03/08/2021. Conto corrente postale n. 1234567890."
        entities = extract_entities(text)
        self.assertEqual([e["text"] for e in entities if e["label"] == "TIME"],
                         ["5/15/21", "03/08/2021"])
        account = next(e for e in entities if e["label"] == "BANK_ACCOUNT")
        self.assertEqual(account["text"], "1234567890")
        self.assertEqual(account["subtype"], "postal_account")
        self.assert_offsets(text, entities)

    def test_grammatical_review_signal_is_preserved(self):
        text = "Richiesta"
        doc = NLPDocument(text, [NLPSpan(text, text, "LOC", pos="NOUN")])
        entities = extract_entities(text, doc=doc)
        self.assertIn("name_without_proper_noun", entities[0]["review_reasons"])

    def test_account_context_variants_do_not_capture_payment_or_document_ids(self):
        for prefix in ("numero di conto", "bonifico sul conto", "contratto relativo al conto", "conto postale"):
            with self.subTest(prefix=prefix):
                entities = extract_entities(f"{prefix} 001234567890.")
                accounts = [e for e in entities if e["label"] == "BANK_ACCOUNT"]
                self.assertEqual([e["text"] for e in accounts], ["001234567890"])
        for prefix in ("copia del contratto", "a mezzo", "assegno n.", "bonifico CRO"):
            with self.subTest(prefix=prefix):
                self.assertFalse(any(e["label"] == "BANK_ACCOUNT"
                                     for e in extract_entities(f"{prefix} 001234567890.")))

    def test_single_call_and_preprocessed_document_agree(self):
        text = "Anna vive a Torino."
        doc = NLPDocument(text, [NLPSpan(text, "Anna", "PER"), NLPSpan(text, "Torino", "LOC")])
        self.assertEqual(extract_entities(text, nlp=lambda value: doc),
                         extract_entities(text, doc=doc))

    def test_conflicting_or_modified_nlp_input_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_entities("Original", doc=NLPDocument("Modified"))
        with self.assertRaises(ValueError):
            extract_entities("Original", nlp=lambda value: NLPDocument(value),
                             doc=NLPDocument("Original"))


if __name__ == "__main__":
    unittest.main()
