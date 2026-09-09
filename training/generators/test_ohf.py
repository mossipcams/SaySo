"""Semantic checks for the pinned OHF English grammar adapter."""
import unittest

from generators.ohf import render_call


class OhfGrammarTest(unittest.TestCase):
    def test_slots_are_not_rewritten_by_grammar_cleanup(self):
        for seed in range(30):
            name = "the 1 minutes  Lamp"
            text = render_call({"name": "HassTurnOn", "arguments": {"name": name, "domain": ["light"]}}, name, seed)
            self.assertIn(name, text)
            self.assertNotIn("the the", text)

    def test_timers_are_complete_requests(self):
        for seed in range(40):
            text = render_call({"name": "HassStartTimer", "arguments": {"minutes": 20, "name": "pasta"}}, "", seed)
            self.assertRegex(text, r"^(start|set|create|begin) ")
            self.assertNotRegex(text, r"\d+-minutes\b")

    def test_possessive_names_do_not_gain_an_article(self):
        for seed in range(30):
            text = render_call({"name": "HassTurnOn", "arguments": {"name": "Bram's Lamp", "domain": ["light"]}}, "Bram's Lamp", seed)
            self.assertNotRegex(text, r"(?:the|my|our) Bram's")
    def test_generated_actions_have_a_verb_and_explicit_scope_domain(self):
        for seed in range(100):
            text = render_call({"name": "HassLightSet", "arguments": {"floor": "Upstairs", "domain": ["light"], "brightness": 35}}, "", seed)
            self.assertIsNotNone(text)
            self.assertRegex(text, r"^(set|change|turn|increase|decrease|make) ")
            self.assertRegex(text, r"light|lamp")
            self.assertIn("brightness", text)
            off = render_call({"name": "HassTurnOff", "arguments": {"name": "reading lamp", "domain": ["light"]}}, "reading lamp", seed)
            self.assertRegex(off, r"^(turn|switch|change|bring) ")
            self.assertIn("off", off)

    def test_provenance_identifies_actual_selected_source(self):
        provenance = []
        call = {"name": "HassTurnOn", "arguments": {"name": "Desk Lamp", "domain": ["light"]}}
        self.assertIsNotNone(render_call(call, "Desk Lamp", 2, provenance=provenance))
        self.assertEqual(len(provenance), 1)
        self.assertTrue(provenance[0]["source"].startswith("sentences/en/HassTurnOn/"))
        self.assertIn("{name}", provenance[0]["template"])
        self.assertEqual(len(provenance[0]["revision"]), 40)

    def test_named_and_scoped_calls_keep_constraints(self):
        for args, target, words in [
            ({"name": "Desk Lamp", "domain": ["light"]}, "reading light", ["reading light"]),
            ({"area": "Kitchen", "domain": ["light"]}, "", ["Kitchen"]),
            ({"floor": "Upstairs", "domain": ["light"]}, "", ["Upstairs"]),
            ({"name": "Desk Lamp", "area": "Office", "domain": ["light"]}, "Desk Lamp", ["Desk Lamp", "Office"]),
        ]:
            for seed in range(20):
                text = render_call({"name": "HassTurnOn", "arguments": args}, target, seed)
                self.assertIsNotNone(text)
                for word in words:
                    self.assertIn(word, text)

    def test_operation_values_and_timer_durations_survive(self):
        for name, args, words in [
            ("HassLightSet", {"name": "Desk Lamp", "brightness": 35}, ["Desk Lamp", "35"]),
            ("HassClimateSetTemperature", {"name": "Thermostat", "temperature": 22}, ["Thermostat", "22"]),
            ("HassSetVolume", {"name": "TV", "volume_level": 40}, ["TV", "40"]),
            ("HassStartTimer", {"name": "pasta", "hours": 1, "minutes": 20, "seconds": 5}, ["pasta", "1", "20", "5"]),
            ("HassIncreaseTimer", {"name": "pasta", "minutes": 3}, ["pasta", "3"]),
        ]:
            for seed in range(20):
                text = render_call({"name": name, "arguments": args}, args.get("name", ""), seed)
                self.assertIsNotNone(text, name)
                for word in words:
                    self.assertIn(word, text)

    def test_unsupported_combinations_fail_explicitly(self):
        for args in [
            {"name": "Lamp", "brightness": 35, "color": "red"},
            {"name": "Lamp", "brightness": 35, "unexpected": 8},
            {"name": "Lamp", "brightness": 35, "area": "Kitchen", "floor": "Upstairs"},
        ]:
            self.assertIsNone(render_call({"name": "HassLightSet", "arguments": args}, "Lamp", 1))

    def test_repeatable_diverse_and_literal_names(self):
        call = {"name": "HassTurnOn", "arguments": {"name": "A [B] (C)", "domain": ["light"]}}
        outputs = {render_call(call, "A [B] (C)", seed) for seed in range(80)}
        self.assertGreater(len(outputs), 10)
        self.assertTrue(all("A [B] (C)" in text for text in outputs))
        self.assertEqual(render_call(call, "A [B] (C)", 7), render_call(call, "A [B] (C)", 7))


if __name__ == "__main__":
    unittest.main()
