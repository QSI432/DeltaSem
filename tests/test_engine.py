import random
import tempfile
import unittest
from pathlib import Path
from deltasem import DeltaSem, Occurrence, Anchor


class MaintenanceTests(unittest.TestCase):
    def test_deleted_input_can_refine_old_certificate(self):
        engine = DeltaSem(lambda key, text: int(text), [
            Occurrence("d", "d", 5, preimage="0"),
            Occurrence("a", "a", 6, preimage="1"),
            Occurrence("b", "b", 6, preimage="1")])
        engine.anchors["group"] = [Anchor("initial", tuple(engine.snapshot.active.values()), 8, 12)]
        engine.delete("d")
        self.assertEqual(engine.intervals()["group"], (3, 12))
        engine.resolve("d")
        self.assertEqual(engine.intervals()["group"], (8, 12))

    def test_random_updates_publish_matches_exact_and_restart(self):
        rng = random.Random(37)
        engine = DeltaSem(lambda key, text: int(text), seed=28)
        labels = {}
        for epoch in range(180):
            if engine.snapshot.active and rng.random() < .4:
                engine.delete(rng.choice(list(engine.snapshot.active)))
            else:
                key = str(rng.randrange(20))
                labels.setdefault(key, rng.randrange(2))
                engine.insert(Occurrence(str(epoch), key, rng.randint(1, 8),
                                         str(rng.randrange(3)), str(labels[key])))
            output = engine.publish(epsilon=.001, k=min(2, len(engine.intervals())) or None, batch=7)
            truth = {g: sum(x.weight*labels[x.semantic_key] for x in engine.snapshot.active.values()
                            if x.group == g) for g in output["intervals"]}
            for g, value in truth.items():
                lo, hi = output["intervals"][g]
                self.assertLessEqual(lo, value)
                self.assertLessEqual(value, hi)
            if output["topk"] is not None:
                expected = sorted(truth, key=lambda g: (-truth[g], g))[:len(output["topk"])]
                self.assertEqual(set(expected), set(output["topk"]))
            if epoch == 90:
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)/"state.json"
                    engine.save(path)
                    restored = DeltaSem.load(path, engine.oracle)
                    self.assertEqual(engine.intervals(), restored.intervals())
                    self.assertEqual(engine.budget.to_state(), restored.budget.to_state())
                    engine = restored

    def test_semantic_key_reuse_and_invalid_replacement(self):
        engine = DeltaSem(lambda key, text: 1)
        engine.insert(Occurrence("a", "same", 1))
        engine.insert(Occurrence("b", "same", 5))
        self.assertEqual(engine.publish(batch=2)["estimates"]["group"], 6)
        self.assertEqual(engine.oracle_calls, 1)
        with self.assertRaises(ValueError):
            engine.replace("a", Occurrence("b", "different", 1))
        self.assertIn("a", engine.snapshot.active)


if __name__ == "__main__":
    unittest.main()
