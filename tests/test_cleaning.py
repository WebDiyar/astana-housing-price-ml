"""Корректность очистки: правила на маленьком искусственном наборе и инварианты на реальных данных."""
import unittest

import numpy as np
import pandas as pd

from src import cleaning
from src import config as C
from src.geo import GeoIndex


def row(**kw):
    base = {"id": 1, "listing_id": 1, "url": "u", "source": "krisha.kz", "residential_complex_id": np.nan,
            "residential_complex_name": "ЖК", "rooms": 2, "area": 60.0, "living_area": np.nan, "kitchen_area": 10.0,
            "floor": 5.0, "total_floors": 9.0, "price_usd": 70000, "price_local": 36_000_000, "price_per_m2": 1166,
            "house_type": "кирпичный", "year_built": 2015, "building_series": np.nan, "ceiling_height": 2.7,
            "condition": np.nan, "bathroom": np.nan, "balcony": np.nan, "parking": np.nan, "furniture": np.nan,
            "floor_type": np.nan, "heating": np.nan, "security": "домофон, охрана", "internet": np.nan, "documents": np.nan,
            "offer_type": np.nan, "address": "a", "district": "Есильский р-н", "latitude": 51.1283, "longitude": 71.4305,
            "parsed_at": "2026-01-03", "updated_at": np.nan, "is_active": 1}
    return {**base, **kw}


class CleaningRules(unittest.TestCase):
    def clean(self, rows):
        df = pd.DataFrame([{**r, "id": i, "listing_id": 100 + i} for i, r in enumerate(rows)])
        return cleaning.clean(df)

    def test_exact_duplicate_removed(self):
        out, j = self.clean([row(), row(), row(area=61.0)])
        self.assertEqual(len(out), 2)
        self.assertEqual(int(j.loc[j["действие"].str.startswith("повтор"), "удалено_строк"].iloc[0]), 1)

    def test_same_object_other_price_removed_only_with_known_floor(self):
        out, _ = self.clean([row(), row(price_local=37_000_000), row(area=70, floor=np.nan, total_floors=np.nan),
                             row(area=70, floor=np.nan, total_floors=np.nan, price_local=45_000_000)])
        self.assertEqual(len(out), 3)

    def test_point_outside_city_removed(self):
        out, _ = self.clean([row(), row(latitude=43.24, longitude=76.9, area=50)])
        self.assertEqual(len(out), 1)

    def test_outliers_removed(self):
        rows = [row(), row(area=10, price_local=6_000_000), row(area=400, price_local=240_000_000),
                row(area=80, price_local=4_000_000), row(area=81, price_local=400_000_000), row(area=120, rooms=8, price_local=70_000_000)]
        out, _ = self.clean(rows)
        self.assertEqual(len(out), 1)
        ppm2 = out[C.TARGET] / out["area"]
        self.assertTrue(ppm2.between(C.PPM2_MIN, C.PPM2_MAX).all())

    def test_impossible_values_become_missing(self):
        out, _ = self.clean([row(ceiling_height=27.0), row(area=50, ceiling_height=9.0), row(area=40, kitchen_area=40.0)])
        self.assertAlmostEqual(out.loc[0, "ceiling_height"], 2.7)
        self.assertTrue(np.isnan(out.loc[1, "ceiling_height"]))
        self.assertTrue(np.isnan(out.loc[2, "kitchen_area"]))

    def test_floor_above_total_removed(self):
        out, _ = self.clean([row(), row(area=55, floor=10.0, total_floors=9.0)])
        self.assertEqual(len(out), 1)

    def test_missing_and_derived_columns(self):
        out, _ = self.clean([row(residential_complex_name=np.nan, security=np.nan, year_built=2027)])
        r = out.iloc[0]
        self.assertEqual(r["condition"], cleaning.NO_VALUE)
        self.assertEqual((r["in_complex"], r["under_construction"], r["security_n"]), (0, 1, 0))
        out, _ = self.clean([row()])
        self.assertEqual((out.loc[0, "sec_guard"], out.loc[0, "sec_intercom"], out.loc[0, "sec_video"], out.loc[0, "security_n"]), (1, 1, 0, 2))

    def test_journal_is_consistent(self):
        _, j = self.clean([row(), row(), row(area=10, price_local=6_000_000)])
        self.assertTrue((j["строк_было"] - j["удалено_строк"] == j["строк_стало"]).all())
        steps = j.iloc[1:-1]
        self.assertTrue((steps["строк_было"].to_numpy()[1:] == steps["строк_стало"].to_numpy()[:-1]).all())


class CleaningOnRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = cleaning.load_raw()
        cls.geo = GeoIndex.from_osm()
        cls.df, cls.journal = cleaning.clean(cls.raw, cls.geo)

    def test_raw_shape(self):
        self.assertEqual(self.raw.shape, (18388, 37))

    def test_no_duplicates_left(self):
        self.assertEqual(int(self.df.duplicated(cleaning.DUP_KEY).sum()), 0)
        known = self.df[self.df["floor"].notna()]
        self.assertEqual(int(known.duplicated(cleaning.OBJ_KEY).sum()), 0)

    def test_values_in_bounds(self):
        d = self.df
        self.assertTrue(d["area"].between(C.AREA_MIN, C.AREA_MAX).all())
        self.assertTrue(d["rooms"].between(1, C.ROOMS_MAX).all())
        self.assertTrue((d[C.TARGET] / d["area"]).between(C.PPM2_MIN, C.PPM2_MAX).all())
        self.assertTrue(d["ceiling_height"].dropna().between(C.CEIL_MIN, C.CEIL_MAX).all())
        self.assertFalse((d["floor"] > d["total_floors"]).any())

    def test_all_points_inside_astana_districts(self):
        self.assertTrue(self.df["district"].isin(C.DISTRICTS).all())
        self.assertEqual(self.df["district"].nunique(), 6)

    def test_journal_totals(self):
        j = self.journal
        self.assertEqual(int(j["строк_стало"].iloc[-1]), len(self.df))
        self.assertEqual(int(j["удалено_строк"].iloc[1:-1].sum()), len(self.raw) - len(self.df))
        self.assertEqual(int(j.loc[j["действие"].str.startswith("повтор"), "удалено_строк"].iloc[0]), 1082)

    def test_no_missing_in_categoricals(self):
        self.assertEqual(int(self.df[cleaning.CAT_COLS + ["district", "residential_complex_name"]].isna().sum().sum()), 0)


if __name__ == "__main__":
    unittest.main()
