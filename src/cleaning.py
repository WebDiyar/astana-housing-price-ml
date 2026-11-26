"""Очистка объявлений с журналом «было -> стало» по каждому шагу."""
import numpy as np
import pandas as pd

from . import config as C

EMPTY_COLS = ["residential_complex_id", "building_series", "heating", "internet", "documents", "offer_type"]
CONST_COLS = ["source", "is_active"]
SPARSE_COLS = ["living_area", "updated_at"]          # заполнены у 0,4 % и 19 % строк
TECH_COLS = ["id", "url", "address", "parsed_at"]    # служебные поля и свободный текст адреса
DUP_KEY = ["latitude", "longitude", "area", "floor", "rooms", C.TARGET]
OBJ_KEY = ["latitude", "longitude", "area", "floor", "rooms"]
CAT_COLS = ["house_type", "condition", "bathroom", "balcony", "parking", "furniture", "floor_type"]
NO_VALUE = "не указано"
SECURITY = {   # значение в поле security -> признак
    "охрана": "sec_guard", "консьерж": "sec_concierge", "видеонаблюдение": "sec_video", "домофон": "sec_intercom",
    "видеодомофон": "sec_videointercom", "кодовый замок": "sec_codelock", "сигнализация": "sec_alarm",
    "решетки на окнах": "sec_bars",
}


def load_raw(path=C.RAW_CSV):
    return pd.read_csv(path)


class Journal:
    """Журнал очистки: строки до и после шага, сколько удалено, сколько значений изменено."""

    def __init__(self):
        self.rows = []

    def add(self, step, action, before, after, cells=0, note=""):
        self.rows.append({"шаг": len(self.rows), "этап": step, "действие": action, "строк_было": before,
                          "строк_стало": after, "удалено_строк": before - after, "изменено_значений": int(cells),
                          "пояснение": note})

    def frame(self):
        return pd.DataFrame(self.rows)


def clean(raw, geo=None):
    """Возвращает (очищенный набор, журнал). geo — GeoIndex: район и принадлежность городу по границам OSM."""
    j = Journal()
    df = raw.copy()
    n0 = len(df)
    j.add("исходный набор", "загрузка CSV", n0, n0, note=f"{df.shape[1]} столбцов")

    # 1. Столбцы без информации
    drop = [c for c in EMPTY_COLS + CONST_COLS + SPARSE_COLS + TECH_COLS + C.LEAK_COLS if c in df.columns]
    df = df.drop(columns=drop)
    j.add("столбцы", "удалены пустые, постоянные, служебные и производные от цены столбцы", n0, len(df), len(drop),
          f"пустые: {len(EMPTY_COLS)}; постоянные: {len(CONST_COLS)}; почти пустые: {len(SPARSE_COLS)}; "
          f"служебные: {len(TECH_COLS)}; производные от цены: {len(C.LEAK_COLS)}")

    # 2. Полные дубли объявления
    n = len(df)
    df = df.drop_duplicates(DUP_KEY, keep="first")
    j.add("дубли", "повтор по (координаты, площадь, этаж, комнаты, цена)", n, len(df), note="оставлено первое объявление")

    # 3. Тот же объект с другой ценой (этаж известен) — иначе один объект попал бы и в обучение, и в тест
    n = len(df)
    df = df[~(df.duplicated(OBJ_KEY, keep="first") & df["floor"].notna())]
    j.add("дубли", "тот же объект с другой ценой: (координаты, площадь, этаж, комнаты)", n, len(df),
          note="оставлено первое объявление; строки без этажа не сравнивались")

    # 4. Точки вне Астаны
    n = len(df)
    s, w, nn, e = C.BBOX
    in_box = df["latitude"].between(s, nn) & df["longitude"].between(w, e)
    df = df[in_box]
    j.add("география", "координаты вне рамки города", n, len(df), note=f"широта {s}–{nn}, долгота {w}–{e}")
    if geo is not None:
        n = len(df)
        osm_d = pd.Series(geo.district(df["latitude"], df["longitude"]), index=df.index)
        df = df[osm_d.notna()]
        j.add("география", "координаты вне границ районов Астаны (OSM)", n, len(df), note="пригороды и ошибки геокодирования")

    # 5. Невозможные и нетипичные значения
    ppm2 = df[C.TARGET] / df["area"]
    for mask, action, note in [
        (ppm2 < C.PPM2_MIN, f"цена за м² ниже {C.PPM2_MIN:,} ₸".replace(",", " "), "ошибка ввода цены или площади"),
        (ppm2 > C.PPM2_MAX, f"цена за м² выше {C.PPM2_MAX:,} ₸".replace(",", " "), "единичные элитные объекты"),
        (df["area"] < C.AREA_MIN, f"площадь меньше {C.AREA_MIN:g} м²", "комнаты в общежитиях"),
        (df["area"] > C.AREA_MAX, f"площадь больше {C.AREA_MAX:g} м²", "пентхаусы, объединённые квартиры"),
        (df["rooms"] > C.ROOMS_MAX, f"комнат больше {C.ROOMS_MAX}", "единичные объекты"),
    ]:
        n = len(df)
        keep = ~mask.reindex(df.index).fillna(False)
        df = df[keep]
        j.add("выбросы", action, n, len(df), note=note)

    n = len(df)
    dm = df["ceiling_height"].between(20, 40)
    df.loc[dm, "ceiling_height"] = df.loc[dm, "ceiling_height"] / 10
    j.add("невозможные значения", "высота потолков 20–40 пересчитана в метры (делением на 10)", n, n, dm.sum(), "высота введена в дециметрах")
    bad = df["ceiling_height"].notna() & ~df["ceiling_height"].between(C.CEIL_MIN, C.CEIL_MAX)
    df.loc[bad, "ceiling_height"] = np.nan
    j.add("невозможные значения", f"высота потолков вне {C.CEIL_MIN:.1f}–{C.CEIL_MAX:.1f} м заменена пропуском".replace(".", ","), n, n, bad.sum())
    bad = df["kitchen_area"].notna() & ((df["kitchen_area"] < C.KITCHEN_MIN) | (df["kitchen_area"] > C.KITCHEN_SHARE_MAX * df["area"]))
    df.loc[bad, "kitchen_area"] = np.nan
    j.add("невозможные значения", f"кухня меньше {C.KITCHEN_MIN:g} м² или больше {C.KITCHEN_SHARE_MAX * 100:.0f} % площади заменена пропуском", n, n, bad.sum())
    bad = df["floor"] > df["total_floors"]
    df = df[~bad]
    j.add("невозможные значения", "этаж больше этажности дома", n, len(df))

    # 6. Строящиеся дома — оставлены, отмечены признаком
    n = len(df)
    df["under_construction"] = (df["year_built"] >= C.COLLECT_YEAR).astype(int)
    j.add("строящиеся дома", f"год постройки {C.COLLECT_YEAR} и позже: строки оставлены, добавлен признак under_construction",
          n, n, df["under_construction"].sum(), "сегмент первичного рынка; ошибка на нём оценивается отдельно")

    # 7. Район: метка объявления -> район по координатам и границам OSM
    df = df.rename(columns={"district": "district_listing"})
    df["district_listing"] = (df["district_listing"].str.replace("р-н", "", regex=False).str.strip()
                              .replace({"Есильский": "Есиль"}))
    if geo is not None:
        n = len(df)
        df["district"] = geo.district(df["latitude"], df["longitude"])
        miss = df["district_listing"].isna()
        diff = ~miss & (df["district_listing"] != df["district"])
        j.add("район", "пропуск района заполнен по координатам (границы OSM)", n, n, miss.sum())
        j.add("район", "метка района из объявления заменена районом по координатам (границы OSM)", n, n, diff.sum(),
              "метки объявлений расходятся с действующими границами районов")
    else:
        df["district"] = df["district_listing"]

    # 8. Пропуски
    n = len(df)
    na_cat = int(df[CAT_COLS].isna().sum().sum())
    df[CAT_COLS] = df[CAT_COLS].fillna(NO_VALUE)
    j.add("пропуски", f"категориальные признаки: пропуск -> «{NO_VALUE}»", n, n, na_cat, "отсутствие сведения в объявлении — отдельная категория")
    df["in_complex"] = df["residential_complex_name"].notna().astype(int)
    df["residential_complex_name"] = df["residential_complex_name"].fillna(NO_VALUE)
    j.add("пропуски", "название ЖК: пропуск -> «не указано», добавлен признак in_complex", n, n, (df["in_complex"] == 0).sum())
    sec = df["security"].fillna("")
    for token, col in SECURITY.items():
        df[col] = sec.apply(lambda s, t=token: int(t in [x.strip() for x in s.split(",")]))
    df["security_n"] = df[list(SECURITY.values())].sum(axis=1)
    df = df.drop(columns="security")
    j.add("пропуски", "поле security разложено на 8 двоичных признаков и их число; пропуск -> нули", n, n, (sec == "").sum())
    na_num = int(df[["floor", "total_floors", "ceiling_height", "kitchen_area"]].isna().sum().sum())
    j.add("пропуски", "числовые признаки: пропуски оставлены (заполняются внутри модели на обучающей части фолда)", n, n, 0,
          f"пропусков в floor, total_floors, ceiling_height, kitchen_area: {na_num}")

    df = df.reset_index(drop=True)
    j.add("итог", "очищенный набор", n0, len(df), note=f"осталось {len(df) / n0:.1%} строк")
    return df, j.frame()
