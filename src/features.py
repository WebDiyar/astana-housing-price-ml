"""Признаки: производные объектные, геопризнаки OSM, группы и наборы A–E для абляции, словарь признаков."""
import numpy as np
import pandas as pd

from . import config as C
from .cleaning import CAT_COLS, SECURITY
from .geo import COUNT_LAYERS, GEO_FEATURES
from .osm import LAYERS

OBJ_NUM = ["area", "rooms", "floor", "total_floors", "floor_ratio", "year_built", "ceiling_height", "kitchen_area", "security_n"]
OBJ_BIN = ["is_first_floor", "is_last_floor", "under_construction", "in_complex"] + list(SECURITY.values())
OBJ_CAT = list(CAT_COLS)
OBJECT = OBJ_NUM + OBJ_BIN + OBJ_CAT
COORDS = ["latitude", "longitude"]
DISTRICT = ["district"]
COMPLEX = ["residential_complex_name"]
CATEGORICAL = OBJ_CAT + DISTRICT + COMPLEX

FEATURE_SETS = {
    "A": OBJECT,
    "B": OBJECT + DISTRICT,
    "C": OBJECT + COORDS,
    "D": OBJECT + DISTRICT + COORDS,
    "E": OBJECT + DISTRICT + COORDS + GEO_FEATURES,
}
SET_LABELS = {
    "A": "A: объектные", "B": "B: A + район", "C": "C: A + координаты",
    "D": "D: A + район + координаты", "E": "E: D + геопризнаки OSM",
}
EXTRA_SETS = {   # дополнительные наборы вне основной лестницы A–E
    "A+OSM": OBJECT + GEO_FEATURES,
    "E+ЖК": FEATURE_SETS["E"] + COMPLEX,
}
GROUPS = {"объектные": OBJECT, "локационные": DISTRICT + COORDS, "геоинфраструктурные": GEO_FEATURES}


def group_of(feature):
    for g, cols in GROUPS.items():
        if feature in cols:
            return g
    return "локационные" if feature in COMPLEX else "—"


def add_object_features(df):
    """Производные объектные признаки (не зависят от цены)."""
    df = df.copy()
    df["floor_ratio"] = df["floor"] / df["total_floors"]
    known = df["floor"].notna()
    df["is_first_floor"] = np.where(known, (df["floor"] == 1).astype(float), np.nan)
    df["is_last_floor"] = np.where(known, (df["floor"] == df["total_floors"]).astype(float), np.nan)
    return df


def build_dataset(clean_df, geo):
    """Очищенный набор + производные и геопризнаки + цена за м² (для анализа, не признак)."""
    df = add_object_features(clean_df)
    g = geo.features(df["latitude"], df["longitude"])
    g.index = df.index
    df = pd.concat([df, g], axis=1)
    df["price_per_m2_kzt"] = df[C.TARGET] / df["area"]
    return df


_RU_LAYER = {"school": "школы", "kindergarten": "детского сада", "healthcare": "больницы или поликлиники",
             "university": "вуза или колледжа", "mall": "торгового центра", "supermarket": "супермаркета",
             "pharmacy": "аптеки", "food": "кафе или ресторана", "bus_stop": "остановки", "park": "парка"}
_RU_LAYER_PL = {"school": "школ", "kindergarten": "детских садов", "healthcare": "больниц и поликлиник",
                "university": "вузов и колледжей", "mall": "торговых центров", "supermarket": "супермаркетов",
                "pharmacy": "аптек", "food": "кафе и ресторанов", "bus_stop": "остановок", "park": "парков"}

DESCRIPTIONS = {
    "area": "общая площадь, м²", "rooms": "число комнат", "floor": "этаж", "total_floors": "этажность дома",
    "floor_ratio": "относительный этаж (этаж / этажность)", "year_built": "год постройки дома",
    "ceiling_height": "высота потолков, м", "kitchen_area": "площадь кухни, м²",
    "security_n": "число указанных средств безопасности", "is_first_floor": "первый этаж (0/1)",
    "is_last_floor": "последний этаж (0/1)", "under_construction": "дом строится: год постройки 2026 и позже (0/1)",
    "in_complex": "указан жилой комплекс (0/1)", "sec_guard": "охрана (0/1)", "sec_concierge": "консьерж (0/1)",
    "sec_video": "видеонаблюдение (0/1)", "sec_intercom": "домофон (0/1)", "sec_videointercom": "видеодомофон (0/1)",
    "sec_codelock": "кодовый замок (0/1)", "sec_alarm": "сигнализация (0/1)", "sec_bars": "решётки на окнах (0/1)",
    "house_type": "тип дома (материал стен)", "condition": "состояние, ремонт", "bathroom": "санузел",
    "balcony": "балкон, лоджия", "parking": "парковка", "furniture": "мебель", "floor_type": "покрытие пола",
    "district": "район города по координатам (границы OSM)", "latitude": "широта, град.", "longitude": "долгота, град.",
    "residential_complex_name": "название жилого комплекса (вне основных наборов; кодируется внутри фолдов)",
    "dist_center_km": "расстояние до центра (монумент «Байтерек»), км",
    "dist_river_km": "расстояние до русла Есиля, км", "near_river": "до русла Есиля не более 500 м (0/1)",
}
for _k in COUNT_LAYERS:
    DESCRIPTIONS[f"dist_{_k}_km"] = f"расстояние до ближайшего(-ей) {_RU_LAYER[_k]}, км"
    DESCRIPTIONS[f"n_{_k}_500m"] = f"число {_RU_LAYER_PL[_k]} в радиусе 500 м"
    DESCRIPTIONS[f"n_{_k}_1km"] = f"число {_RU_LAYER_PL[_k]} в радиусе 1 км"

SHORT = {   # короткие подписи для рисунков
    "area": "Площадь", "rooms": "Комнат", "floor": "Этаж", "total_floors": "Этажность", "floor_ratio": "Относительный этаж",
    "year_built": "Год постройки", "ceiling_height": "Высота потолков", "kitchen_area": "Площадь кухни",
    "security_n": "Средств безопасности", "is_first_floor": "Первый этаж", "is_last_floor": "Последний этаж",
    "under_construction": "Дом строится", "in_complex": "Указан ЖК", "sec_guard": "Охрана", "sec_concierge": "Консьерж",
    "sec_video": "Видеонаблюдение", "sec_intercom": "Домофон", "sec_videointercom": "Видеодомофон",
    "sec_codelock": "Кодовый замок", "sec_alarm": "Сигнализация", "sec_bars": "Решётки на окнах",
    "house_type": "Тип дома", "condition": "Состояние", "bathroom": "Санузел", "balcony": "Балкон", "parking": "Парковка",
    "furniture": "Мебель", "floor_type": "Пол", "district": "Район", "latitude": "Широта", "longitude": "Долгота",
    "residential_complex_name": "Название ЖК", "dist_center_km": "До центра, км", "dist_river_km": "До Есиля, км",
    "near_river": "У берега Есиля",
}
_SH = {"school": "школ", "kindergarten": "дет. сад", "healthcare": "больниц", "university": "вуз", "mall": "ТРЦ",
       "supermarket": "супермаркет", "pharmacy": "аптек", "food": "кафе", "bus_stop": "остановк", "park": "парк"}
_SH_TO = {"school": "школы", "kindergarten": "дет. сада", "healthcare": "больницы", "university": "вуза", "mall": "ТРЦ",
          "supermarket": "супермаркета", "pharmacy": "аптеки", "food": "кафе", "bus_stop": "остановки", "park": "парка"}
_SH_N = {"school": "Школ", "kindergarten": "Дет. садов", "healthcare": "Больниц", "university": "Вузов", "mall": "ТРЦ",
         "supermarket": "Супермаркетов", "pharmacy": "Аптек", "food": "Кафе", "bus_stop": "Остановок", "park": "Парков"}
for _k in COUNT_LAYERS:
    SHORT[f"dist_{_k}_km"] = f"До {_SH_TO[_k]}, км"
    SHORT[f"n_{_k}_500m"] = f"{_SH_N[_k]} в 500 м"
    SHORT[f"n_{_k}_1km"] = f"{_SH_N[_k]} в 1 км"


def feature_dictionary(df):
    """Словарь признаков: признак, группа, тип, описание, доля пропусков."""
    rows = []
    for f in OBJECT + DISTRICT + COORDS + GEO_FEATURES + COMPLEX:
        kind = "категориальный" if f in CATEGORICAL else ("двоичный" if f in OBJ_BIN or f == "near_river" else "числовой")
        rows.append({"признак": f, "группа": group_of(f), "тип": kind, "описание": DESCRIPTIONS[f],
                     "доля_пропусков_%": round(float(df[f].isna().mean() * 100), 2),
                     "уникальных": int(df[f].nunique())})
    return pd.DataFrame(rows)
