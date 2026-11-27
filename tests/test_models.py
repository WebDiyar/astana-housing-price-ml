"""Метрики, статистические поправки и воспроизводимость при фиксированном seed."""
import unittest

import numpy as np

from src import config as C
from src import features as F
from src import validation as V
from src.models import PriceModel, metrics
from src.pipeline import prepare

FAST = {"ridge": {"alpha": 1.0}, "rf": {"n_estimators": 30}, "hgb": {"max_iter": 60},
        "catboost": {"iterations": 80, "depth": 6}}


class Metrics(unittest.TestCase):
    def test_known_values(self):
        m = metrics([10e6, 20e6, 40e6], [11e6, 18e6, 40e6])
        self.assertAlmostEqual(m["MAE_млн"], 1.0)
        self.assertAlmostEqual(m["MAPE_%"], (10 + 10 + 0) / 3)
        self.assertAlmostEqual(m["MdAPE_%"], 10.0)
        self.assertAlmostEqual(m["RMSE_млн"], np.sqrt(5 / 3))

    def test_perfect_forecast(self):
        m = metrics([10e6, 20e6], [10e6, 20e6])
        self.assertEqual((m["MAE_млн"], m["MAPE_%"], m["R2"]), (0.0, 0.0, 1.0))

    def test_holm(self):
        np.testing.assert_allclose(V.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertTrue((V.holm([0.5, 0.9]) <= 1).all())


class Reproducibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data, _, _ = prepare()
        cls.data = data.sample(2500, random_state=C.SEED).reset_index(drop=True)
        cls.y = cls.data[C.TARGET].to_numpy(float)
        cls.tr, cls.te = V.holdout_split(cls.data)

    def fit_predict(self, kind, seed=C.SEED):
        m = PriceModel(kind, F.FEATURE_SETS["E"], FAST.get(kind), seed).fit(self.data.iloc[self.tr], self.y[self.tr])
        return m.predict(self.data.iloc[self.te])

    def test_same_seed_same_forecast(self):
        for kind in ("baseline", "ridge", "hgb", "catboost"):
            a, b = self.fit_predict(kind), self.fit_predict(kind)
            np.testing.assert_array_equal(a, b, err_msg=kind)
        # Random Forest усредняет деревья в нескольких потоках: порядок сложения меняет последние разряды (~1e-14)
        np.testing.assert_allclose(self.fit_predict("rf"), self.fit_predict("rf"), rtol=1e-11, err_msg="rf")

    def test_other_seed_changes_stochastic_models(self):
        for kind in ("rf", "catboost"):
            self.assertFalse(np.array_equal(self.fit_predict(kind, 1), self.fit_predict(kind, 2)), kind)

    def test_splits_are_fixed(self):
        tr2, te2 = V.holdout_split(self.data)
        np.testing.assert_array_equal(self.te, te2)
        self.assertFalse(set(self.tr) & set(self.te))
        self.assertEqual(len(self.te), round(0.2 * len(self.data)))
        a, b = V.repeated_splits(100), V.repeated_splits(100)
        self.assertEqual(len(a), 15)
        self.assertTrue(all(np.array_equal(x[1], z[1]) for x, z in zip(a, b)))
        cells = V.spatial_cells(self.data["latitude"], self.data["longitude"])
        s1, s2 = V.spatial_splits(cells), V.spatial_splits(cells)
        self.assertTrue(all(np.array_equal(x[1], z[1]) for x, z in zip(s1, s2)))

    def test_models_beat_shuffled_target(self):
        """Проверка процедуры оценки: на перемешанных ценах модель не должна ничего «угадывать», на настоящих — должна."""
        y_perm = np.random.default_rng(C.SEED).permutation(self.y[self.tr])
        m = PriceModel("hgb", F.FEATURE_SETS["E"], FAST["hgb"]).fit(self.data.iloc[self.tr], y_perm)
        self.assertLess(metrics(self.y[self.te], m.predict(self.data.iloc[self.te]))["R2_log"], 0.05)
        m = PriceModel("hgb", F.FEATURE_SETS["E"], FAST["hgb"]).fit(self.data.iloc[self.tr], self.y[self.tr])
        self.assertGreater(metrics(self.y[self.te], m.predict(self.data.iloc[self.te]))["R2_log"], 0.8)

    def test_forecast_is_in_tenge_and_positive(self):
        p = self.fit_predict("hgb")
        self.assertTrue((p > 3e6).all() and (p < 2e9).all())


if __name__ == "__main__":
    unittest.main()
