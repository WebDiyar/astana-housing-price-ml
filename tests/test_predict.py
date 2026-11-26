"""Сервис прогноза: сценарии с координатами и с районом, чувствительность, некорректный ввод, командная строка."""
import json
import subprocess
import sys
import unittest

from predict import predict
from src import config as C
from src.pipeline import BUNDLE, load_bundle

FLAT = {"area": 62, "rooms": 2, "floor": 7, "total_floors": 12, "year_built": 2019, "house_type": "монолитный",
        "latitude": 51.1283, "longitude": 71.4305}


class Predict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle()      # нет файла — понятная ошибка: сначала «python -m src.pipeline»

    def test_scenario_coordinates(self):
        r = predict(FLAT, bundle=self.bundle)
        self.assertEqual((r["model"], r["feature_set"], r["district"]), ("full", "E", "Есиль"))
        self.assertTrue(15e6 < r["price"] < 200e6)
        self.assertTrue(r["interval_90"][0] <= r["interval_80"][0] < r["price"] < r["interval_80"][1] <= r["interval_90"][1])
        self.assertEqual(len(r["factors"]), 5)
        self.assertTrue(all({"feature", "label", "value", "effect_%"} <= set(f) for f in r["factors"]))

    def test_scenario_district_only(self):
        r = predict({"area": 45, "rooms": 1, "year_built": 1985, "house_type": "панельный", "district": "Сарыарка"}, bundle=self.bundle)
        self.assertEqual((r["model"], r["feature_set"]), ("district", "B"))
        self.assertTrue(5e6 < r["price"] < 60e6)
        self.assertTrue(any("координаты не заданы" in w for w in r["warnings"]))

    def test_scenario_minimal_input(self):
        r = predict({"area": 80, "rooms": 3, "year_built": 2024, "district": "Нура"}, bundle=self.bundle)
        self.assertGreater(r["price"], 0)
        self.assertTrue(any("«floor»" in w for w in r["warnings"]))

    def test_bigger_flat_costs_more_and_old_centre_differs(self):
        small = predict({**FLAT, "area": 45, "rooms": 1}, bundle=self.bundle)["price"]
        big = predict({**FLAT, "area": 110, "rooms": 3}, bundle=self.bundle)["price"]
        self.assertGreater(big, small * 1.5)
        old = predict({**FLAT, "year_built": 1975, "house_type": "панельный", "floor": 3, "total_floors": 5}, bundle=self.bundle)["price"]
        self.assertLess(old, predict(FLAT, bundle=self.bundle)["price"])

    def test_deterministic(self):
        self.assertEqual(predict(FLAT, bundle=self.bundle), predict(FLAT, bundle=self.bundle))

    def test_invalid_input(self):
        bad = [
            {**FLAT, "area": -5}, {**FLAT, "area": 1000}, {**FLAT, "rooms": 0}, {**FLAT, "rooms": 2.5},
            {**FLAT, "floor": 15}, {**FLAT, "house_type": "деревянный"}, {**FLAT, "area": "много"},
            {**FLAT, "latitude": 43.24, "longitude": 76.9},                       # точка в Алматы
            {k: v for k, v in FLAT.items() if k != "longitude"},                   # широта без долготы
            {k: v for k, v in FLAT.items() if k not in ("latitude", "longitude")},  # ни координат, ни района
            {"area": 60, "rooms": 2, "year_built": 2010, "district": "Медеуский"},  # район другого города
            {k: v for k, v in FLAT.items() if k != "area"}, {**FLAT, "year_built": 2040},
        ]
        for case in bad:
            with self.assertRaises(ValueError, msg=str(case)):
                predict(case, bundle=self.bundle)

    def test_bundle_contents(self):
        b = self.bundle
        self.assertEqual(b["meta"]["seed"], C.SEED)
        self.assertEqual(set(b["models"]), {"full", "district"})
        for name in b["models"]:
            q = b["intervals"][name]
            self.assertTrue(q["q05"] < q["q10"] < 1 < q["q90"] < q["q95"])
            self.assertTrue(0.7 < b["coverage_80"][name] < 0.9)
        self.assertLess(b["metrics"]["full"]["MAPE_%"], b["metrics"]["district"]["MAPE_%"])


class CommandLine(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, "predict.py", *args], capture_output=True, text=True, cwd=C.ROOT)

    def test_json_output(self):
        r = self.run_cli("--area", "62", "--rooms", "2", "--floor", "7", "--total-floors", "12", "--year", "2019",
                         "--house-type", "монолитный", "--lat", "51.1283", "--lon", "71.4305", "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertGreater(json.loads(r.stdout)["price"], 0)

    def test_invalid_input_exit_code(self):
        r = self.run_cli("--area", "62", "--rooms", "2", "--year", "2019", "--floor", "20", "--total-floors", "9", "--district", "Есиль")
        self.assertEqual(r.returncode, 2)
        self.assertIn("Ошибка ввода", r.stderr)


if __name__ == "__main__":
    BUNDLE.exists() or sys.exit("нет файла модели: сначала «python -m src.pipeline»")
    unittest.main()
