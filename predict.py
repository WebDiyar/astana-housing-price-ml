#!/usr/bin/env python3
"""Прототип сервиса оценки: прогноз цены квартиры в Астане, интервал и главные факторы.

Примеры:
    python predict.py --area 62 --rooms 2 --floor 7 --total-floors 12 --year 2019 --house-type монолитный --lat 51.1283 --lon 71.4305
    python predict.py --area 45 --rooms 1 --year 1985 --house-type панельный --district Сарыарка --json

Из кода:
    from predict import predict
    predict({"area": 62, "rooms": 2, "floor": 7, "total_floors": 12, "year_built": 2019,
             "house_type": "монолитный", "latitude": 51.1283, "longitude": 71.4305})
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

from src import config as C
from src import features as F
from src.analysis import shap_values
from src.cleaning import NO_VALUE, SECURITY
from src.pipeline import load_bundle

_BUNDLE = None
OPTIONAL_CAT = ["house_type", "condition", "bathroom", "balcony", "parking", "furniture", "floor_type"]


def _bundle():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = load_bundle()
    return _BUNDLE


def _number(a, key, lo, hi, required=False, integer=False):
    v = a.get(key)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        if required:
            raise ValueError(f"не задано обязательное поле «{key}»")
        return np.nan
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"поле «{key}» должно быть числом, получено: {a.get(key)!r}") from None
    if not np.isfinite(v) or not lo <= v <= hi:
        raise ValueError(f"поле «{key}» = {v:g} вне допустимого диапазона {lo:g}–{hi:g}")
    if integer and v != int(v):
        raise ValueError(f"поле «{key}» должно быть целым, получено: {v:g}")
    return v


def build_row(a, bundle):
    """Проверяет ввод и собирает строку признаков. Возвращает (таблица из одной строки, имя модели, предупреждения)."""
    meta, geo, warn = bundle["meta"], bundle["geo"], []
    row = {
        "area": _number(a, "area", C.AREA_MIN, C.AREA_MAX, required=True),
        "rooms": _number(a, "rooms", 1, C.ROOMS_MAX, required=True, integer=True),
        "floor": _number(a, "floor", 1, 60, integer=True),
        "total_floors": _number(a, "total_floors", 1, 60, integer=True),
        "year_built": _number(a, "year_built", 1940, meta["year_range"][1], required=True, integer=True),
        "ceiling_height": _number(a, "ceiling_height", C.CEIL_MIN, C.CEIL_MAX),
        "kitchen_area": _number(a, "kitchen_area", C.KITCHEN_MIN, C.AREA_MAX),
    }
    if row["floor"] > row["total_floors"]:
        raise ValueError(f"этаж ({row['floor']:g}) больше этажности дома ({row['total_floors']:g})")
    if row["kitchen_area"] > C.KITCHEN_SHARE_MAX * row["area"]:
        raise ValueError("площадь кухни больше 60 % общей площади")
    if np.isnan(row["floor"]) != np.isnan(row["total_floors"]):
        warn.append("этаж и этажность нужно задавать вместе; неполная пара учтена как пропуск")
        row["floor"] = row["total_floors"] = np.nan
    row["floor_ratio"] = row["floor"] / row["total_floors"]
    row["is_first_floor"] = np.nan if np.isnan(row["floor"]) else float(row["floor"] == 1)
    row["is_last_floor"] = np.nan if np.isnan(row["floor"]) else float(row["floor"] == row["total_floors"])
    row["under_construction"] = float(row["year_built"] >= C.COLLECT_YEAR)
    row["in_complex"] = float(bool(a.get("in_complex", False)))
    sec = a.get("security") or []                       # список или строка через запятую
    sec = [s.strip() for s in (sec.split(",") if isinstance(sec, str) else sec)]
    for token, col in SECURITY.items():
        row[col] = float(token in sec)
    unknown = [s for s in sec if s and s not in SECURITY]
    if unknown:
        raise ValueError(f"неизвестные средства безопасности: {unknown}; допустимо: {list(SECURITY)}")
    row["security_n"] = float(sum(row[c] for c in SECURITY.values()))
    for c in OPTIONAL_CAT:
        v = a.get(c) or NO_VALUE
        if v not in meta["categories"][c]:
            raise ValueError(f"поле «{c}» = {v!r}; допустимо: {meta['categories'][c]}")
        row[c] = v

    lat, lon, district = a.get("latitude"), a.get("longitude"), a.get("district")
    if (lat is None) != (lon is None):
        raise ValueError("широту и долготу нужно задавать вместе")
    if lat is not None:
        lat, lon = _number(a, "latitude", -90, 90), _number(a, "longitude", -180, 180)
        d = geo.district([lat], [lon])[0]
        if d is None:
            raise ValueError(f"точка ({lat:g}, {lon:g}) лежит вне границ районов Астаны")
        if district and district != d:
            warn.append(f"указан район «{district}», по координатам — «{d}»; использован район по координатам")
        row.update({"latitude": lat, "longitude": lon, "district": d})
        row.update(geo.features([lat], [lon]).iloc[0].to_dict())
        name = "full"
    elif district:
        if district not in meta["districts"]:
            raise ValueError(f"неизвестный район {district!r}; допустимо: {meta['districts']}")
        row["district"] = district
        name = "district"
        warn.append("координаты не заданы: использована модель «объектные признаки + район», её точность ниже")
    else:
        raise ValueError("нужно задать координаты (latitude, longitude) или район (district)")
    for k in ("floor", "ceiling_height", "kitchen_area"):
        if np.isnan(row[k]):
            warn.append(f"поле «{k}» не задано — модель использует его как пропуск")
    return pd.DataFrame([row]), name, warn


def predict(apartment, top=5, bundle=None):
    """Прогноз цены: {'price', 'price_per_m2', 'interval_80', 'interval_90', 'factors', 'model', 'district', 'warnings'}."""
    bundle = bundle or _bundle()
    X, name, warn = build_row(dict(apartment), bundle)
    model, q = bundle["models"][name], bundle["intervals"][name]
    price = float(model.predict(X)[0])
    sv, base = shap_values(model, X)
    order = np.argsort(-np.abs(sv[0]))[:top]
    factors = []
    for i in order:
        f = model.features[i]
        v = X[f].iloc[0]
        value = v if isinstance(v, str) else (None if pd.isna(v) else round(float(v), 3))      # None — признак не задан
        factors.append({"feature": f, "label": F.SHORT.get(f, f), "value": value, "effect_%": round(float(np.expm1(sv[0][i]) * 100), 1)})
    return {
        "price": round(price, -3), "price_per_m2": round(price / X["area"].iloc[0], -2),
        "interval_80": [round(price * q["q10"], -3), round(price * q["q90"], -3)],
        "interval_90": [round(price * q["q05"], -3), round(price * q["q95"], -3)],
        "base_price": round(float(np.exp(base)), -3),
        "model": name, "feature_set": bundle["metrics"][name]["feature_set"], "district": X["district"].iloc[0],
        "test_MAPE_%": round(bundle["metrics"][name]["MAPE_%"], 2), "factors": factors, "warnings": warn,
    }


def _fmt(v):
    return f"{v / 1e6:.2f} млн ₸"


def main(argv=None):
    p = argparse.ArgumentParser(description="Оценка цены квартиры в Астане (прототип, НИРМ 03).")
    p.add_argument("--area", type=float, required=True, help="общая площадь, м²")
    p.add_argument("--rooms", type=float, required=True, help="число комнат")
    p.add_argument("--year", dest="year_built", type=float, required=True, help="год постройки")
    p.add_argument("--floor", type=float)
    p.add_argument("--total-floors", dest="total_floors", type=float)
    p.add_argument("--house-type", dest="house_type", help="монолитный | кирпичный | панельный | иной")
    p.add_argument("--condition", help="состояние (как в объявлении)")
    p.add_argument("--ceiling", dest="ceiling_height", type=float, help="высота потолков, м")
    p.add_argument("--kitchen", dest="kitchen_area", type=float, help="площадь кухни, м²")
    p.add_argument("--in-complex", dest="in_complex", action="store_true", help="дом входит в жилой комплекс")
    p.add_argument("--lat", dest="latitude", type=float)
    p.add_argument("--lon", dest="longitude", type=float)
    p.add_argument("--district", help=" | ".join(C.DISTRICTS))
    p.add_argument("--top", type=int, default=5, help="сколько факторов показать")
    p.add_argument("--json", action="store_true", help="вывести результат в JSON")
    args = vars(p.parse_args(argv))
    top, as_json = args.pop("top"), args.pop("json")
    try:
        res = predict({k: v for k, v in args.items() if v is not None}, top=top)
    except ValueError as ex:
        print(f"Ошибка ввода: {ex}", file=sys.stderr)
        return 2
    if as_json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 0
    print(f"Прогноз цены: {_fmt(res['price'])} ({res['price_per_m2'] / 1e3:.0f} тыс. ₸ за м²), район: {res['district']}")
    print(f"Интервал 80 %: {_fmt(res['interval_80'][0])} – {_fmt(res['interval_80'][1])}; "
          f"90 %: {_fmt(res['interval_90'][0])} – {_fmt(res['interval_90'][1])}")
    print(f"Модель: набор {res['feature_set']}, MAPE на тесте {res['test_MAPE_%']} %")
    print("Главные факторы (вклад в цену относительно среднего уровня):")
    for f in res["factors"]:
        print(f"  {f['label']:<24} = {'не задано' if f['value'] is None else f['value']!s:<14} {f['effect_%']:+.1f} %")
    for w in res["warnings"]:
        print(f"Замечание: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
