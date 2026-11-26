"""Модели прогноза цены. Обучение на log(цены), прогноз и метрики — в тенге."""
import os

import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder
from threadpoolctl import threadpool_limits

from .config import SEED
from .features import CATEGORICAL, COMPLEX

SKLEARN_VERSION = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
N_JOBS = int(os.environ.get("NIRM3_THREADS", 2))   # потоков на одну модель; фолды считаются параллельными процессами (validation.N_PROCS)
MODEL_LABELS = {"baseline": "Базовая (медиана ₸/м² района × площадь)", "ridge": "Ridge", "rf": "Random Forest",
                "hgb": "HistGradientBoosting", "catboost": "CatBoost"}
MODEL_ORDER = ["baseline", "ridge", "rf", "hgb", "catboost"]

# Исходные гиперпараметры (до подбора)
DEFAULTS = {
    "ridge": {"alpha": 1.0},
    "rf": {"n_estimators": 300, "max_features": 0.5, "min_samples_leaf": 1, "max_depth": None},
    "hgb": {"learning_rate": 0.08, "max_iter": 600, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.0},
    "catboost": {"iterations": 700, "learning_rate": 0.1, "depth": 6, "l2_leaf_reg": 3.0, "rsm": 0.5},
}
# Пространства случайного поиска. У CatBoost rsm = 0,5 (половина признаков на разбиение) закреплён: вдвое быстрее без потери точности
SEARCH_SPACE = {
    "ridge": {"alpha": [float(a) for a in np.round(np.logspace(-2, 3, 16), 4)]},
    "rf": {"n_estimators": [300], "max_features": [0.3, 0.5, 0.7, 1.0], "min_samples_leaf": [1, 2, 4], "max_depth": [None, 20]},
    "hgb": {"learning_rate": [0.05, 0.08, 0.12], "max_iter": [300, 600], "max_leaf_nodes": [31, 63],
            "min_samples_leaf": [10, 20], "l2_regularization": [0.0, 1.0], "max_features": [0.5, 1.0]},
    "catboost": {"iterations": [700], "learning_rate": [0.08, 0.1, 0.15], "depth": [6, 8],
                 "l2_leaf_reg": [1.0, 3.0, 10.0], "rsm": [0.5]},
}
N_ITER = {"ridge": 16, "rf": 8, "hgb": 8, "catboost": 6}


def metrics(y_true, y_pred):
    """Метрики в исходной шкале: MAE и RMSE в млн ₸, MAPE и медианная абсолютная ошибка в %, R²."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ape = np.abs(y_pred - y_true) / y_true * 100
    return {
        "MAE_млн": mean_absolute_error(y_true, y_pred) / 1e6,
        "RMSE_млн": float(np.sqrt(np.mean((y_true - y_pred) ** 2))) / 1e6,
        "MAPE_%": float(ape.mean()),
        "MdAPE_%": float(np.median(ape)),
        "R2": r2_score(y_true, y_pred),
        "R2_log": r2_score(np.log(y_true), np.log(y_pred)),
    }


METRIC_COLS = ["MAE_млн", "RMSE_млн", "MAPE_%", "MdAPE_%", "R2", "R2_log"]


def _log_feature(f):
    return f in ("area", "kitchen_area") or f.startswith("dist_") or f.startswith("n_")


def _split(features):
    cat_high = [f for f in features if f in COMPLEX]
    cat = [f for f in features if f in CATEGORICAL and f not in COMPLEX]
    num = [f for f in features if f not in CATEGORICAL]
    return num, cat, cat_high


def _target_encoder(seed):
    # Среднее log(цены) по категории; при обучении считается по внутренним фолдам, поэтому своя цена в код строки не попадает
    if SKLEARN_VERSION >= (1, 9):       # с версии 1.9 перемешивание задаётся генератором разбиений (результат тот же)
        return TargetEncoder(target_type="continuous", smooth="auto", cv=KFold(5, shuffle=True, random_state=seed))
    return TargetEncoder(target_type="continuous", smooth="auto", cv=5, random_state=seed)


def build_estimator(kind, features, params=None, seed=SEED):
    """Оценщик sklearn/CatBoost для log(цены) по таблице с именованными столбцами."""
    p = {**DEFAULTS[kind], **(params or {})}
    num, cat, cat_high = _split(features)
    if kind == "ridge":
        num_log = [f for f in num if _log_feature(f)]
        num_lin = [f for f in num if not _log_feature(f)]
        steps = [
            ("log", make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                                  StandardScaler()), num_log),
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num_lin),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False), cat),
        ]
        if cat_high:
            steps.append(("te", make_pipeline(_target_encoder(seed), StandardScaler()), cat_high))
        return make_pipeline(ColumnTransformer(steps), Ridge(alpha=p["alpha"]))
    if kind in ("rf", "hgb"):
        steps = [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), cat),
                 ("num", "passthrough", num)]
        if cat_high:
            steps.append(("te", _target_encoder(seed), cat_high))
        pre = ColumnTransformer(steps)
        if kind == "rf":
            est = RandomForestRegressor(random_state=seed, n_jobs=N_JOBS, **p)
        else:
            est = HistGradientBoostingRegressor(random_state=seed, categorical_features=list(range(len(cat))),
                                                early_stopping=False, **p)
        return make_pipeline(pre, est)
    if kind == "catboost":
        return CatBoostRegressor(loss_function="RMSE", random_seed=seed, thread_count=N_JOBS, verbose=0,
                                 allow_writing_files=False, cat_features=cat + cat_high, **p)
    raise ValueError(f"неизвестная модель: {kind}")


class DistrictMedianBaseline:
    """Базовая оценка: медиана цены за м² по району (на обучении) × площадь."""

    def fit(self, X, price):
        ppm2 = np.asarray(price, float) / X["area"].to_numpy(float)
        self.global_ = float(np.median(ppm2))
        self.by_district_ = pd.Series(ppm2).groupby(X["district"].to_numpy()).median().to_dict()
        return self

    def predict(self, X):
        m = X["district"].map(self.by_district_).fillna(self.global_).to_numpy(float)
        return m * X["area"].to_numpy(float)


class PriceModel:
    """Единый интерфейс: fit(таблица, цена в ₸) и predict(таблица) -> цена в ₸. Внутри — log(цены)."""

    def __init__(self, kind, features, params=None, seed=SEED):
        self.kind, self.features, self.params, self.seed = kind, list(features), dict(params or {}), seed

    def _frame(self, X):
        X = X[self.features].copy()
        for c in X.columns:
            if c in CATEGORICAL:
                X[c] = X[c].astype(object).where(X[c].notna(), "не указано").astype(str)
            else:
                X[c] = X[c].astype(float)
        return X

    def fit(self, X, price):
        if self.kind == "baseline":
            self.est_ = DistrictMedianBaseline().fit(X, price)
        else:
            self.est_ = build_estimator(self.kind, self.features, self.params, self.seed)
            with threadpool_limits(limits=N_JOBS):
                self.est_.fit(self._frame(X), np.log(np.asarray(price, float)))
        return self

    def predict_log(self, X):
        if self.kind == "baseline":
            return np.log(self.est_.predict(X))
        with threadpool_limits(limits=N_JOBS):
            return np.asarray(self.est_.predict(self._frame(X)), float)

    def predict(self, X):
        return np.exp(self.predict_log(X))
