"""Offline checks for passage coverage, routing and grounded LLM decisions."""

import json
import importlib.util
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.deterministic import extract_entities
from src.llm_filter import merge_reviews, parse_entities, review_messages
from src.passages import build_passages, select_passages, smoke_sample


class PassageTests(unittest.TestCase):
    def test_cores_cover_original_text_and_each_candidate_has_one_owner(self):
        text = ("Un testo italiano da conservare. " * 20) + "Email: anna@example.org"
        document = {"id": "one", "text": text, "entities": extract_entities(text)}
        passages = build_passages([document], 120, 25)
        self.assertEqual("".join(text[p["core_start"]:p["core_end"]] for p in passages), text)
        self.assertEqual(sum(len(p["candidates"]) for p in passages), len(document["entities"]))
        for passage in passages:
            self.assertEqual(passage["text"], text[passage["start"]:passage["end"]])
            for entity in passage["candidates"]:
                self.assertLessEqual(entity["end"], passage["end"])

    def test_context_extends_to_preserve_long_owned_candidate(self):
        text = "A" * 300
        entity = {"text": text[90:250], "start": 90, "end": 250}
        passages = build_passages([{"id": "one", "text": text, "entities": [entity]}], 100, 0)
        self.assertEqual(passages[0]["end"], 250)
        self.assertEqual(sum(len(p["candidates"]) for p in passages), 1)

    def test_short_final_remainder_is_absorbed_without_losing_text(self):
        text = "Un testo italiano da dividere. " * 4
        passages = build_passages([{"id": "one", "text": text, "entities": []}], 115, 10)
        self.assertEqual(len(passages), 1)
        self.assertEqual(passages[0]["core_end"], len(text))

    def test_smoke_sampling_is_balanced_reproducible_and_uses_distinct_documents(self):
        documents = [{"id": f"{source}-{index}", "source": source,
                      "text": "A long example. " * 20, "entities": []}
                     for source in ("one", "two") for index in range(3)]
        passages = build_passages(documents, 120, 20)
        selected = smoke_sample(passages, 2, 42)
        self.assertEqual(selected, smoke_sample(passages, 2, 42))
        self.assertEqual(len({p["document_id"] for p in selected}), 4)
        self.assertEqual([p["source"] for p in selected].count("one"), 2)
        self.assertEqual([p["source"] for p in selected].count("two"), 2)

    def test_routing_can_find_passages_without_candidates_and_respects_budget(self):
        passages = [{"id": str(i), "text": "Testo italiano senza candidati. " * 5, "candidates": []}
                    for i in range(5)]
        scores = {p["id"]: {"status": "ok", "perplexity": 20 + i, "mean_nll": 3 + i}
                  for i, p in enumerate(passages)}
        settings = {"perplexity_threshold": 22, "max_review_fraction": 0.4, "max_review_passages": 3}
        routing = select_passages(passages, scores, settings)
        self.assertEqual(sum(d["selected"] for d in routing["decisions"].values()), 2)
        self.assertIn("possible_omission", routing["decisions"]["0"]["reasons"])
        self.assertTrue(routing["decisions"]["4"]["selected"])

    def test_sample_percentile_is_not_used_to_calibrate_an_extended_run(self):
        settings = {"perplexity_threshold": None, "diagnostic_percentile": 0.8,
                    "max_review_fraction": 0.2, "max_review_passages": 10}
        with self.assertRaises(ValueError):
            select_passages([], {}, settings)
        self.assertEqual(select_passages([], {}, settings, smoke_test=True)["threshold_mode"],
                         "smoke_sample_percentile")


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.text = "Anna vive a Roma. Anna lavora a Roma."
        self.passage = {"id": "one@0", "document_id": "one", "start": 10,
                        "core_start": 10, "core_end": 10 + len(self.text),
                        "text": self.text, "candidates": []}

    def parse(self, entities):
        return parse_entities(json.dumps({"entities": entities}), self.passage, "test-model")

    def test_repeated_mentions_map_to_the_correct_original_offsets(self):
        entities = self.parse([{"text": "Anna", "label": "PERSON", "occurrence": 2}])
        self.assertEqual(entities[0]["start"], 10 + self.text.rindex("Anna"))
        self.assertTrue(entities[0]["llm_reviewed"])

    def test_unique_match_resolves_an_incorrect_index_with_a_recorded_flag(self):
        self.passage["text"] = "Elena vive qui."
        entity = self.parse([{"text": "Elena", "label": "PERSON", "occurrence": 2}])[0]
        self.assertEqual(entity["start"], self.passage["start"])
        self.assertIn("occurrence_index_resolved_from_unique_match", entity["review_reasons"])
        self.assertTrue(entity["needs_review"])

    def test_invented_values_invalid_labels_and_ambiguous_occurrences_fail(self):
        for item in ({"text": "Elena", "label": "PERSON"},
                     {"text": "Anna", "label": "NEW_CATEGORY", "occurrence": 1},
                     {"text": "Anna", "label": "PERSON"},
                     {"text": "Anna", "label": "PERSON", "occurrence": 3},
                     {"text": "Anna", "label": "PERSON", "occurrence": True}):
            with self.subTest(item=item), self.assertRaises(ValueError):
                self.parse([item])

    def test_mentions_in_context_are_owned_by_the_adjacent_core(self):
        self.passage["core_start"] = 20
        self.assertEqual(self.parse([{"text": "Anna", "label": "PERSON", "occurrence": 1}]), [])

    def test_response_requires_a_complete_json_object(self):
        for response in ('{"entities":[', '{"entities":[],"explanation":"extra"}', 'Text {"entities":[]}'):
            with self.subTest(response=response), self.assertRaises(ValueError):
                parse_entities(response, self.passage, "test-model")
        self.assertEqual(parse_entities('```json\n{"entities":[]}\n```', self.passage, "test-model"), [])

    def test_prompt_does_not_include_gold_or_run_metadata(self):
        self.passage["annotations"] = [{"secret_gold": True}]
        self.passage["candidates"] = [{"text": "invented candidate", "label": "PERSON"}]
        self.assertNotIn("secret_gold", json.dumps(review_messages(self.passage)))
        self.assertNotIn("invented candidate", json.dumps(review_messages(self.passage)))

    def test_only_successful_cores_replace_candidates_and_input_is_unchanged(self):
        text = "Email: anna@example.org. Email: elena@example.org."
        document = {"id": "one", "text": text, "entities": extract_entities(text)}
        before = json.dumps(document, sort_keys=True)
        passage = {"id": "one@0", "document_id": "one", "core_start": 0, "core_end": 25}
        failed = merge_reviews([document], [passage], {"one@0": {"status": "invalid_response", "entities": []}})
        self.assertEqual(failed[0]["entities"], document["entities"])
        successful = merge_reviews([document], [passage], {"one@0": {"status": "ok", "entities": []}})
        self.assertEqual([e["text"] for e in successful[0]["entities"]], ["elena@example.org"])
        self.assertEqual(json.dumps(document, sort_keys=True), before)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch arithmetic check runs in the Colab environment.")
class PerplexityArithmeticTests(unittest.TestCase):
    def test_shift_and_padding_against_hand_calculated_probabilities(self):
        import torch
        from src.language_models import perplexity_batches

        class Batch(dict):
            def to(self, device):
                # Keep the numerical fixture on CPU; it needs no GPU or model weights.
                return self

        class Tokenizer:
            def __call__(self, texts, **kwargs):
                return self.pad({"input_ids": [[1, 2] if text == "short" else [1, 2, 3] for text in texts]})

            def pad(self, inputs, **kwargs):
                maximum = max(len(tokens) for tokens in inputs["input_ids"])
                return Batch(
                    input_ids=torch.tensor([tokens + [0] * (maximum - len(tokens)) for tokens in inputs["input_ids"]]),
                    attention_mask=torch.tensor([[1] * len(tokens) + [0] * (maximum - len(tokens))
                                                 for tokens in inputs["input_ids"]]),
                )

        def model(input_ids, attention_mask, use_cache):
            rows = [[0.1, 0.1, 0.5, 0.3], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]
            probabilities = torch.tensor(rows[:input_ids.shape[1]])
            return SimpleNamespace(logits=probabilities.log().unsqueeze(0).repeat(input_ids.shape[0], 1, 1))

        model.device = torch.device("cpu")
        passages = [{"id": "short", "text": "short"}, {"id": "long", "text": "long"}]
        with patch("src.language_models.require_cuda", return_value=torch), patch.object(torch.cuda, "synchronize"):
            batch = next(perplexity_batches(passages, {"max_tokens": 8, "batch_size": 2}, model, Tokenizer()))
        self.assertAlmostEqual(batch["short"]["perplexity"], 2.0, places=5)
        self.assertAlmostEqual(batch["long"]["perplexity"], math.sqrt(8), places=5)
        self.assertEqual(batch["short"]["scored_tokens"], 1)
        self.assertEqual(batch["long"]["scored_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
