import unittest

from src.relation_graph import build_graph


def document(text, named=()):
    entities = []
    for value, label in named:
        start = text.index(value)
        entities.append({"text": value, "label": label, "start": start,
                         "end": start + len(value), "needs_review": False})
    return {"id": "example", "text": text, "entities": entities}


class RelationGraphTests(unittest.TestCase):
    def test_public_body_does_not_become_a_company_node(self):
        text = "Lea Blu, dipendente del Comune di Roma."
        graph = build_graph([document(text, (("Lea Blu", "PERSON"), ("Comune di Roma", "COMPANY")))])
        self.assertFalse(any(node.get("label") == "COMPANY" for node in graph["nodes"]))
        self.assertFalse(any(edge["type"] == "EXPLICIT_RELATION" for edge in graph["edges"]))

    def test_descriptive_subject_replaces_mistagged_place_in_relation(self):
        text = "Una 42enne di Carpi, legale rappresentante della società Alfa."
        graph = build_graph([document(text, (("Carpi", "PERSON"), ("società Alfa", "COMPANY")))])
        nodes = {node["id"]: node for node in graph["nodes"]}
        edges = [edge for edge in graph["edges"] if edge["type"] == "EXPLICIT_RELATION"]

        self.assertEqual(len(edges), 1)
        edge = edges[0]
        self.assertEqual(nodes[edge["source"]]["name"], "Una 42enne di Carpi")
        self.assertEqual(nodes[edge["source"]]["mention_kind"], "description")
        self.assertEqual(nodes[edge["target"]]["name"], "società Alfa")
        self.assertNotIn("Carpi", [node.get("name") for node in nodes.values()])
        evidence = edge["evidence"][0]
        self.assertEqual(text[evidence["predicate_start"]:evidence["predicate_end"]],
                         "legale rappresentante")
        self.assertEqual(text[evidence["start"]:evidence["end"]], text[:-1])

    def test_repeated_descriptions_are_distinct_occurrences(self):
        text = "Una donna è la direttrice dell'azienda. Una donna è la direttrice dell'azienda."
        graph = build_graph([document(text)])
        edges = [edge for edge in graph["edges"] if edge["type"] == "EXPLICIT_RELATION"]

        self.assertEqual(len(edges), 2)
        self.assertNotEqual(edges[0]["source"], edges[1]["source"])
        self.assertNotEqual(edges[0]["target"], edges[1]["target"])

    def test_named_people_and_an_unnamed_target(self):
        text = "L'avvocata Lea Blu, difensore di un imputato, ha parlato con Ada Neri."
        graph = build_graph([document(text, (("Lea Blu", "PERSON"), ("Ada Neri", "PERSON")))])
        nodes = {node["id"]: node for node in graph["nodes"]}
        edges = [edge for edge in graph["edges"] if edge["type"] == "EXPLICIT_RELATION"]

        self.assertEqual(len(edges), 1)
        self.assertEqual(nodes[edges[0]["source"]]["name"], "Lea Blu")
        self.assertEqual(nodes[edges[0]["target"]]["name"], "un imputato")
        self.assertEqual(nodes[edges[0]["target"]]["mention_kind"], "description")
        self.assertTrue(any(edge["type"] == "CO_OCCURS_WITH" for edge in graph["edges"]))

    def test_cooccurrence_and_negation_do_not_create_a_relation(self):
        text = "Erano presenti Lea Blu e Ada Neri. Lea Blu non è la direttrice della società Alfa."
        graph = build_graph([document(text, (("Lea Blu", "PERSON"), ("Ada Neri", "PERSON"),
                                            ("società Alfa", "COMPANY")))])

        self.assertTrue(any(edge["type"] == "CO_OCCURS_WITH" for edge in graph["edges"]))
        self.assertFalse(any(edge["type"] == "EXPLICIT_RELATION" for edge in graph["edges"]))

    def test_unnamed_role_subject_with_two_targets(self):
        text = "Il difensore di Ada Neri e Delta S.r.l. ha risposto."
        graph = build_graph([document(text, (("Ada Neri", "PERSON"), ("Delta S.r.l.", "COMPANY")))])
        nodes = {node["id"]: node for node in graph["nodes"]}
        edges = [edge for edge in graph["edges"] if edge["type"] == "EXPLICIT_RELATION"]

        self.assertEqual(len(edges), 2)
        self.assertEqual({nodes[edge["source"]]["name"] for edge in edges}, {"Il difensore"})
        self.assertEqual({nodes[edge["target"]]["name"] for edge in edges}, {"Ada Neri", "Delta S.r.l."})

    def test_employment_verb_with_descriptive_organisation(self):
        text = "La donna lavora presso un centro estetico."
        graph = build_graph([document(text)])
        nodes = {node["id"]: node for node in graph["nodes"]}
        edges = [edge for edge in graph["edges"] if edge["type"] == "EXPLICIT_RELATION"]

        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["relation"], "lavora presso")
        self.assertEqual(nodes[edges[0]["target"]]["name"], "un centro estetico")


if __name__ == "__main__":
    unittest.main()
