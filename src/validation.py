"""Схемы проверки (отложенный тест, повторная k-кратная, пространственная), подбор гиперпараметров, статистические тесты."""
import itertools
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.model_selection import KFold, ParameterSampler, RepeatedKFold, train_test_split

from .config import SEED, TARGET
from .models import METRIC_COLS, N_ITER, SEARCH_SPACE, PriceModel, metrics

CELL_KM = 2.0      # сторона пространственной ячейки
LAT0 = 51.13       # широта Астаны для пересчёта градусов долготы в км
# Сколько моделей обучается одновременно (отдельными процессами). На результат не влияет — только на время счёта
N_PROCS = int(os.environ.get("NIRM3_PROCS", max(1, min((os.cpu_count() or 2) // 2, 5))))


def holdout_split(df, test_size=0.2, seed=SEED):
    """Отложенный тест 20 % со стратификацией по району. Возвращает индексы обучения и теста."""
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=df["district"])
    return np.sort(tr), np.sort(te)


def repeated_splits(n, n_splits=5, n_repeats=3, seed=SEED):
    return list(RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(np.arange(n)))


def spatial_cells(lat, lon, cell_km=CELL_KM):
    """Номер пространственной ячейки cell_km × cell_km для каждой точки."""
    i = np.floor(np.asarray(lat) * 110.57 / cell_km).astype(int)
    k = np.floor(np.asarray(lon) * 111.32 * np.cos(np.radians(LAT0)) / cell_km).astype(int)
    return np.array([f"{a}_{b}" for a, b in zip(i, k)])


def spatial_splits(cells, n_splits=5, seed=SEED):
    """Пространственные фолды: ячейка целиком попадает в один фолд; размеры фолдов выравниваются."""
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(cells, return_counts=True)
    order = rng.permutation(len(uniq))
    order = order[np.argsort(-counts[order], kind="stable")]
    load, fold_of = np.zeros(n_splits), {}
    for g in order:
        f = int(np.argmin(load))
        fold_of[uniq[g]] = f
        load[f] += counts[g]
    fold = np.array([fold_of[c] for c in cells])
    return [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(n_splits)]


def group_splits(groups):
    """«Обучение на остальных группах -> проверка на одной» (например, по районам)."""
    groups = np.asarray(groups)
    return [(np.where(groups != g)[0], np.where(groups == g)[0], g) for g in sorted(pd.unique(groups))]


def _columns(features):
    return list(dict.fromkeys(list(features) + ["area", "district"]))      # базовой оценке нужны площадь и район


def _procs(kind, n_procs):
    return 1 if kind in ("baseline", "ridge") else (n_procs or N_PROCS)


def fit_score(kind, features, params, seed, X_tr, y_tr, X_te, y_te, keep=False):
    """Обучение одной модели и метрики на проверочной части (в тенге). keep=True — вернуть модель и прогноз."""
    t0 = time.perf_counter()
    model = PriceModel(kind, features, params, seed).fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(X_te)
    out = {**metrics(y_te, pred), "fit_s": fit_s, "predict_ms_per_1000": (time.perf_counter() - t0) / len(X_te) * 1e6,
           "rmse_log": float(np.sqrt(np.mean((np.log(pred) - np.log(y_te)) ** 2)))}
    return {**out, "model": model, "pred": pred} if keep else out


def evaluate_splits(kind, features, params, df, splits, seed=SEED, extra=None, n_procs=None):
    """Метрики модели по каждому фолду (в тенге). Фолды обучаются параллельными процессами."""
    y = df[TARGET].to_numpy(float)
    X = df[_columns(features)]
    res = Parallel(n_jobs=_procs(kind, n_procs))(
        delayed(fit_score)(kind, features, params, seed, X.iloc[s[0]], y[s[0]], X.iloc[s[1]], y[s[1]]) for s in splits)
    rows = []
    for i, (split, r) in enumerate(zip(splits, res)):
        row = {"fold": i, "n_train": len(split[0]), "n_test": len(split[1]), **{k: v for k, v in r.items() if k not in ("rmse_log", "predict_ms_per_1000")}}
        if len(split) > 2:
            row["group"] = split[2]
        rows.append({**(extra or {}), **row})
    return pd.DataFrame(rows)


def fit_holdout(specs, train, test, seed=SEED, n_procs=None):
    """Обучение нескольких моделей на обучающей части и оценка на тесте. specs: {имя: (модель, признаки, параметры)}."""
    y_tr, y_te = train[TARGET].to_numpy(float), test[TARGET].to_numpy(float)
    res = Parallel(n_jobs=n_procs or N_PROCS)(
        delayed(fit_score)(kind, feats, params, seed, train[_columns(feats)], y_tr, test[_columns(feats)], y_te, True)
        for kind, feats, params in specs.values())
    return dict(zip(specs, res))


def random_search(kind, features, df, n_iter=None, cv=3, seed=SEED, n_procs=None):
    """Случайный поиск гиперпараметров на обучающей части: критерий — RMSE log(цены) по cv фолдам."""
    space = SEARCH_SPACE[kind]
    n_iter = min(n_iter or N_ITER[kind], int(np.prod([len(v) for v in space.values()])))
    y = df[TARGET].to_numpy(float)
    X = df[_columns(features)]
    folds = list(KFold(cv, shuffle=True, random_state=seed).split(df))
    grid = [{k: (v.item() if hasattr(v, "item") else v) for k, v in p.items()} for p in ParameterSampler(space, n_iter=n_iter, random_state=seed)]
    res = Parallel(n_jobs=_procs(kind, n_procs))(
        delayed(fit_score)(kind, features, params, seed, X.iloc[tr], y[tr], X.iloc[va], y[va]) for params in grid for tr, va in folds)
    rows = []
    for i, params in enumerate(grid):
        part = res[i * cv:(i + 1) * cv]
        errs = [r["rmse_log"] for r in part]
        rows.append({"model": kind, "config": i, **{f"param_{k}": v for k, v in params.items()}, "rmse_log": float(np.mean(errs)),
                     "rmse_log_std": float(np.std(errs)), "MAPE_%": float(np.mean([r["MAPE_%"] for r in part])),
                     "fit_s_mean": float(np.mean([r["fit_s"] for r in part])), "params": params})
    res = pd.DataFrame(rows).sort_values("rmse_log").reset_index(drop=True)
    return res.loc[0, "params"], res.drop(columns="params")


def summarize(folds, by):
    """Среднее и стандартное отклонение метрик по фолдам."""
    g = folds.groupby(by, sort=False)[METRIC_COLS]
    mean, std = g.mean(), g.std(ddof=1)
    out = pd.concat({"mean": mean, "std": std}, axis=1).swaplevel(axis=1)
    out = out[[(m, s) for m in METRIC_COLS for s in ("mean", "std")]]
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["n_folds"] = g.size()
    return out.reset_index()


# ---------- статистика ----------

def holm(pvals):
    """Поправка Холма на множественные сравнения."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def corrected_ttest(a, b, n_train, n_test):
    """Парный t-критерий с поправкой Надо — Бенжио на зависимость фолдов повторной перекрёстной проверки."""
    d = np.asarray(a, float) - np.asarray(b, float)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt(var * (1 / k + n_test / n_train))
    return float(t), float(2 * stats.t.sf(abs(t), k - 1))


def friedman(wide):
    """Критерий Фридмана: строки — фолды, столбцы — модели (меньше — лучше)."""
    stat, p = stats.friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
    ranks = wide.rank(axis=1).mean()
    return float(stat), float(p), ranks


def pairwise_tests(wide, n_train=None, n_test=None, pairs=None):
    """Попарный критерий Уилкоксона с поправкой Холма (+ t-критерий Надо — Бенжио, если заданы размеры фолда)."""
    pairs = pairs or list(itertools.combinations(wide.columns, 2))
    rows = []
    for a, b in pairs:
        d = wide[a] - wide[b]
        w = stats.wilcoxon(wide[a], wide[b]) if (d != 0).any() else None
        row = {"A": a, "B": b, "mean_A": wide[a].mean(), "mean_B": wide[b].mean(), "diff_A_minus_B": d.mean(),
               "diff_rel_%": d.mean() / wide[a].mean() * 100, "folds_B_better": int((d > 0).sum()), "n_folds": len(d),
               "wilcoxon_W": float(w.statistic) if w else np.nan, "p_wilcoxon": float(w.pvalue) if w else 1.0}
        if n_train:
            row["t_NB"], row["p_NB"] = corrected_ttest(wide[a], wide[b], n_train, n_test)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_wilcoxon"])
    if n_train:
        out["p_NB_holm"] = holm(out["p_NB"])
    return out
