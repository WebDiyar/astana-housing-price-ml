"""Сквозной конвейер: загрузка -> очистка -> признаки -> обучение итоговой модели -> сохранение.

    python -m src.pipeline            # собрать набор, обучить и сохранить models/price_model.joblib
"""
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np

from . import cleaning, features as F, osm
from . import config as C
from .geo import GeoIndex
from .models import PriceModel, metrics
from .validation import holdout_split

BUNDLE = C.MODELS / "price_model.joblib"
FINAL_KIND = "catboost"
Q_LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def prepare(log=None):
    """Загрузка, очистка и расчёт признаков. Возвращает (набор, журнал очистки, геоиндекс)."""
    osm.download(log=log or (lambda *_: None))          # скачивает только отсутствующие файлы OSM
    geo = GeoIndex.from_osm()
    clean_df, journal = cleaning.clean(cleaning.load_raw(), geo)
    return F.build_dataset(clean_df, geo), journal, geo


def save_processed(data, journal):
    """Сохраняет обработанный набор, словарь признаков и журнал очистки."""
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    data.to_csv(C.PROCESSED / "astana_features.csv", index=False)
    F.feature_dictionary(data).to_csv(C.PROCESSED / "feature_dictionary.csv", index=False)
    journal.to_csv(C.RESULTS / "cleaning_journal.csv", index=False)


def tuned_params(kind=FINAL_KIND):
    path = C.RESULTS / "hyperparams.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")).get(kind, {}).get("best_params", {})
    return {}


def ratio_quantiles(y, pred):
    """Квантили отношения «факт / прогноз» — основа интервала прогноза."""
    r = np.asarray(y, float) / np.asarray(pred, float)
    return {f"q{int(q * 100):02d}": float(np.quantile(r, q)) for q in Q_LEVELS}


def interval_check(y, pred, seed=C.SEED, lo="q10", hi="q90"):
    """Проверка интервала: квантили по одной половине теста, покрытие — на другой (в обе стороны)."""
    idx = np.random.default_rng(seed).permutation(len(y))
    a, b = idx[: len(y) // 2], idx[len(y) // 2:]
    cover = []
    for cal, chk in ((a, b), (b, a)):
        q = ratio_quantiles(y[cal], pred[cal])
        r = y[chk] / pred[chk]
        cover.append(float(((r >= q[lo]) & (r <= q[hi])).mean()))
    return float(np.mean(cover))


def train_bundle(data, geo, params=None, seed=C.SEED, kind=FINAL_KIND):
    """Обучает две модели (с координатами — набор E; только район — набор B) на обучающей части и собирает пакет сервиса."""
    params = tuned_params(kind) if params is None else params
    tr, te = holdout_split(data, seed=seed)
    y = data[C.TARGET].to_numpy(float)
    bundle = {"geo": geo, "models": {}, "intervals": {}, "metrics": {}, "coverage_80": {}}
    for name, fset in (("full", "E"), ("district", "B")):
        t0 = time.perf_counter()
        model = PriceModel(kind, F.FEATURE_SETS[fset], params, seed).fit(data.iloc[tr], y[tr])
        pred = model.predict(data.iloc[te])
        bundle["models"][name] = model
        bundle["intervals"][name] = ratio_quantiles(y[te], pred)
        bundle["coverage_80"][name] = interval_check(y[te], pred, seed)
        bundle["metrics"][name] = {**metrics(y[te], pred), "fit_s": time.perf_counter() - t0, "feature_set": fset}
    cats = {c: sorted(data[c].unique()) for c in F.OBJ_CAT}
    bundle["meta"] = {
        "seed": seed, "kind": kind, "params": params, "n_train": int(len(tr)), "n_test": int(len(te)),
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(), "categories": cats, "districts": C.DISTRICTS,
        "district_points": data.groupby("district")[["latitude", "longitude"]].median().round(5).T.to_dict(),
        "year_range": [int(data["year_built"].min()), int(data["year_built"].max())],
    }
    return bundle


def save_bundle(bundle, path=BUNDLE):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def load_bundle(path=BUNDLE):
    if not path.exists():
        raise FileNotFoundError(f"нет файла модели {path}: сначала выполните «python -m src.pipeline»")
    return joblib.load(path)


if __name__ == "__main__":
    t0 = time.perf_counter()
    data, journal, geo = prepare(log=print)
    save_processed(data, journal)
    bundle = train_bundle(data, geo)
    path = save_bundle(bundle)
    print(f"набор: {len(data)} строк; модель: {path} ({path.stat().st_size / 1e6:.1f} МБ); {time.perf_counter() - t0:.0f} с")
    for name, m in bundle["metrics"].items():
        print(name, {k: round(v, 3) for k, v in m.items() if k != "feature_set"})
