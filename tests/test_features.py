"""Отсутствие утечки целевых столбцов в признаки, состав наборов A–E, геопризнаки и пространственные фолды."""
import unittest

import numpy as np

from src import config as C
from src import features as F
from src import osm, validation as V
from src.geo import GEO_FEATURES, GeoIndex
from src.models import build_estimator
from src.pipeline import prepare

FORBIDDEN = {"price_local", "price_usd", "price_per_m2", "price_per_m2_kzt", "listing_id", "district_listing", "id", "url"}


class FeatureSets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.journal, cls.geo = prepare()

    def test_no_target_or_leak_columns_in_any_set(self):
        for name, cols in {**F.FEATURE_SETS, **F.EXTRA_SETS}.items():
            self.assertFalse(FORBIDDEN & set(cols), f"набор {name}")
            self.assertEqual(len(cols), len(set(cols)), f"повтор признака в наборе {name}")
            self.assertTrue(set(cols) <= set(self.data.columns), f"набор {name}")

    def test_leak_columns_absent_from_processed_data(self):
        self.assertFalse({"price_usd", "price_per_m2"} & set(self.data.columns))

    def test_no_feature_duplicates_the_target(self):
        num = [f for f in F.FEATURE_SETS["E"] if f not in F.CATEGORICAL]
        for target in (C.TARGET, "price_per_m2_kzt"):
            corr = self.data[num].corrwith(self.data[target], method="spearman").abs()
            self.assertLess(corr.max(), 0.95, f"{corr.idxmax()} почти повторяет {target}")

    def test_ablation_ladder(self):
        s = {k: set(v) for k, v in F.FEATURE_SETS.items()}
        self.assertTrue(s["A"] < s["B"] < s["D"] < s["E"])
        self.assertTrue(s["A"] < s["C"] < s["D"])
        self.assertEqual(s["E"] - s["D"], set(GEO_FEATURES))
        self.assertFalse(s["A"] & (set(GEO_FEATURES) | {"latitude", "longitude", "district"}))
        self.assertNotIn("residential_complex_name", s["E"])

    def test_complex_name_is_encoded_inside_pipeline(self):
        est = build_estimator("hgb", F.EXTRA_SETS["E+ЖК"])
        names = [type(t).__name__ for _, t, _ in est.steps[0][1].transformers]
        self.assertIn("TargetEncoder", names)       # кодирование обучается вместе с моделью, т. е. внутри фолда

    def test_dictionary_covers_all_features(self):
        d = F.feature_dictionary(self.data)
        self.assertEqual(set(d["признак"]), set(F.EXTRA_SETS["E+ЖК"]))
        self.assertEqual(set(d["группа"]), {"объектные", "локационные", "геоинфраструктурные"})

    def test_geo_features_are_sane(self):
        g = self.data[GEO_FEATURES]
        self.assertEqual(int(g.isna().sum().sum()), 0)
        self.assertTrue((g >= 0).all().all())
        for k in osm.POINT_LAYERS + ["park"]:
            self.assertTrue((g[f"n_{k}_1km"] >= g[f"n_{k}_500m"]).all(), k)
            self.assertTrue(((g[f"dist_{k}_km"] <= 0.5) == (g[f"n_{k}_500m"] > 0)).all(), k)


class Geo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geo = GeoIndex.from_osm()

    def test_haversine(self):
        self.assertAlmostEqual(osm.haversine_km(51.0, 71.4, 52.0, 71.4), 111.195, places=2)
        self.assertAlmostEqual(osm.haversine_km(51.1, 71.4, 51.1, 71.4), 0.0, places=6)

    def test_district_by_point(self):
        d = self.geo.district([C.CENTER[0], 43.24], [C.CENTER[1], 76.9])
        self.assertEqual(d[0], "Есиль")       # «Байтерек» стоит в районе Есиль
        self.assertIsNone(d[1])                # Алматы (город) — вне Астаны

    def test_center_distance_is_zero_at_center(self):
        f = self.geo.features([C.CENTER[0]], [C.CENTER[1]])
        self.assertAlmostEqual(f["dist_center_km"].iloc[0], 0.0, places=6)
        self.assertEqual(set(f.columns), set(GEO_FEATURES))


class SpatialFolds(unittest.TestCase):
    def test_cells_do_not_cross_folds(self):
        rng = np.random.default_rng(0)
        lat, lon = 51.05 + rng.random(3000) * 0.15, 71.3 + rng.random(3000) * 0.25
        cells = V.spatial_cells(lat, lon)
        splits = V.spatial_splits(cells, 5)
        seen = np.zeros(len(lat), int)
        for tr, te in splits:
            self.assertFalse(set(cells[tr]) & set(cells[te]))
            seen[te] += 1
        self.assertTrue((seen == 1).all())
        sizes = [len(te) for _, te in splits]
        self.assertLess(max(sizes) - min(sizes), 0.1 * len(lat))


if __name__ == "__main__":
    unittest.main()
