"""Интерпретация модели и анализ ошибок: перестановочная важность, SHAP, частная зависимость, ошибки по сегментам."""
import numpy as np
import pandas as pd

from .config import SEED, TARGET
from .features import CATEGORICAL, GROUPS, SHORT, group_of
from .models import metrics


def permutation_importance(model, X, y, columns=None, n_repeats=10, seed=SEED):
    """Рост ошибки на тесте при перемешивании признака (или группы признаков целиком).

    columns: {название: [столбцы]}; по умолчанию — каждый признак модели отдельно.
    """
    rng = np.random.default_rng(seed)
    columns = columns or {f: [f] for f in model.features}
    base = metrics(y, model.predict(X))
    rows = []
    for name, cols in columns.items():
        d_mape, d_mae = [], []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
            m = metrics(y, model.predict(Xp))
            d_mape.append(m["MAPE_%"] - base["MAPE_%"])
            d_mae.append(m["MAE_млн"] - base["MAE_млн"])
        rows.append({"feature": name, "dMAPE_пп_mean": np.mean(d_mape), "dMAPE_пп_std": np.std(d_mape, ddof=1),
                     "dMAE_млн_mean": np.mean(d_mae), "dMAE_млн_std": np.std(d_mae, ddof=1)})
    return pd.DataFrame(rows).sort_values("dMAPE_пп_mean", ascending=False).reset_index(drop=True)


def shap_values(model, X):
    """Значения SHAP для log(цены), точный алгоритм для деревьев. Возвращает (матрица в порядке model.features, базовое значение)."""
    Xf = model._frame(X)
    if model.kind == "catboost":
        from catboost import Pool
        cats = [c for c in model.features if c in CATEGORICAL]
        sv = model.est_.get_feature_importance(Pool(Xf, cat_features=cats), type="ShapValues")
        return sv[:, :-1], float(sv[0, -1])
    import shap                                    # Random Forest и HistGradientBoosting: признаки после кодирования
    pre, est = model.est_.steps[0][1], model.est_.steps[-1][1]
    names = [c for _, _, cols in pre.transformers_ for c in cols]
    expl = shap.TreeExplainer(est)
    sv = pd.DataFrame(expl.shap_values(pre.transform(Xf)), columns=names)[model.features].to_numpy()
    return sv, float(np.ravel(expl.expected_value)[0])


def shap_summary(sv, features):
    """Средний |SHAP| по признакам и доля в сумме, плюс группа признака."""
    imp = np.abs(sv).mean(axis=0)
    out = pd.DataFrame({"feature": features, "mean_abs_shap": imp, "share_%": imp / imp.sum() * 100,
                        "group": [group_of(f) for f in features], "label": [SHORT.get(f, f) for f in features]})
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def group_importance(model, X, y, sv, n_repeats=10, seed=SEED):
    """Важность групп признаков: совместная перестановка группы и доля группы в сумме среднего |SHAP|."""
    groups = {g: [f for f in cols if f in model.features] for g, cols in GROUPS.items()}
    groups = {g: c for g, c in groups.items() if c}
    perm = permutation_importance(model, X, y, groups, n_repeats, seed).set_index("feature")
    imp = pd.Series(np.abs(sv).mean(axis=0), index=model.features)
    rows = []
    for g, cols in groups.items():
        rows.append({"group": g, "n_features": len(cols), "shap_share_%": imp[cols].sum() / imp.sum() * 100,
                     "perm_dMAPE_пп": perm.loc[g, "dMAPE_пп_mean"], "perm_dMAPE_пп_std": perm.loc[g, "dMAPE_пп_std"],
                     "perm_dMAE_млн": perm.loc[g, "dMAE_млн_mean"]})
    return pd.DataFrame(rows)


def partial_dependence(model, X, feature, grid=None, n_grid=25, sample=2000, seed=SEED):
    """Частная зависимость: средний прогноз цены (млн ₸) при принудительной подстановке значения признака."""
    Xs = X.sample(min(sample, len(X)), random_state=seed)
    if grid is None:
        if feature in CATEGORICAL:
            grid = list(X[feature].value_counts().index)
        else:
            lo, hi = X[feature].quantile([0.02, 0.98])
            grid = np.unique(np.round(np.linspace(lo, hi, n_grid), 3))
    rows = []
    for v in grid:
        Xv = Xs.copy()
        Xv[feature] = v
        p = model.predict(Xv) / 1e6
        rows.append({"feature": feature, "value": v, "pd_mean_млн": p.mean(), "pd_median_млн": float(np.median(p))})
    return pd.DataFrame(rows)


def error_table(y, pred, keys, label, order=None):
    """Ошибки по сегментам: число объектов, MAPE, медианная ошибка, MAE, смещение. order — порядок строк."""
    d = pd.DataFrame({"key": np.asarray(keys), "y": np.asarray(y, float), "p": np.asarray(pred, float)})
    d["ape"] = (d.p - d.y).abs() / d.y * 100
    d["pe"] = (d.p - d.y) / d.y * 100
    d["ae"] = (d.p - d.y).abs() / 1e6
    g = d.groupby("key", sort=False, observed=True)
    out = pd.DataFrame({"n": g.size(), "MAPE_%": g.ape.mean(), "MdAPE_%": g.ape.median(), "MAE_млн": g.ae.mean(),
                        "смещение_%": g.pe.mean(), "медиана_цены_млн": g.y.median() / 1e6}).reset_index()
    order = sorted(out["key"]) if order is None else [k for k in order if k in set(out["key"])]
    out = out.set_index("key").loc[order].reset_index()
    out.insert(0, "segment", label)
    return out.rename(columns={"key": "value"})


YEAR_LABELS = ["до 1990", "1990–2009", "2010–2019", "2020–2025", "строится (2026+)"]
QUINTILES = ["Q1 (дешёвые)", "Q2", "Q3", "Q4", "Q5 (дорогие)"]
AGE_LABELS = ["старый фонд (до 2020)", "новостройки (2020+)"]


def year_bin(year):
    return pd.cut(year, bins=[0, 1989, 2009, 2019, 2025, 9999], labels=YEAR_LABELS)


def segment_errors(df, y, pred):
    """MAPE по районам, числу комнат, ценовым квинтилям и периоду постройки."""
    q = pd.qcut(y, 5, labels=QUINTILES)
    parts = [
        error_table(y, pred, df["district"], "район"),
        error_table(y, pred, df["rooms"].astype(int), "комнаты"),
        error_table(y, pred, q, "ценовой квинтиль", QUINTILES),
        error_table(y, pred, year_bin(df["year_built"]), "период постройки", YEAR_LABELS),
        error_table(y, pred, np.where(df["year_built"] >= 2020, AGE_LABELS[1], AGE_LABELS[0]), "новостройки / старый фонд", AGE_LABELS),
    ]
    out = pd.concat(parts, ignore_index=True)
    out["value"] = out["value"].astype(str)
    return out


def worst_cases(df, y, pred, n=10):
    cols = ["listing_id", "district", "residential_complex_name", "rooms", "area", "floor", "total_floors", "year_built",
            "house_type", "condition", "dist_center_km"]
    d = df[cols].copy()
    d["цена_млн"] = np.asarray(y) / 1e6
    d["прогноз_млн"] = np.asarray(pred) / 1e6
    d["ошибка_%"] = (d["прогноз_млн"] - d["цена_млн"]) / d["цена_млн"] * 100
    d["цена_за_м2_тыс"] = np.asarray(y) / df["area"].to_numpy() / 1e3
    return d.reindex(d["ошибка_%"].abs().sort_values(ascending=False).index).head(n).reset_index(drop=True)
