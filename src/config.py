"""Пути, seed и константы эксперимента (единое место, чтобы блокнот, тесты и сервис совпадали)."""
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "astana_apartments.csv"
OSM_DIR = ROOT / "data" / "raw" / "osm"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

# Рамка запросов к OpenStreetMap (юг, запад, север, восток) и центр города (монумент «Байтерек»)
BBOX = (50.95, 71.15, 51.35, 71.75)
CENTER = (51.1283, 71.4305)

TARGET = "price_local"                      # цена предложения, тенге
LEAK_COLS = ["price_usd", "price_per_m2"]   # производные от цены: в признаки нельзя
USD_RATE = 508.91                           # курс, зашитый автором набора (price_local / price_usd)

# Границы допустимых значений (обоснование — data/README.md и раздел «Очистка» блокнота)
AREA_MIN, AREA_MAX = 15.0, 300.0            # общая площадь, м²
ROOMS_MAX = 6
PPM2_MIN, PPM2_MAX = 200_000, 2_000_000     # цена за м², тенге
CEIL_MIN, CEIL_MAX = 2.3, 5.0               # высота потолков, м
KITCHEN_MIN, KITCHEN_SHARE_MAX = 3.0, 0.6   # кухня: не меньше 3 м² и не больше 60 % площади
COLLECT_YEAR = 2026                         # год сбора объявлений (январь 2026)

DISTRICTS = ["Алматы", "Байконур", "Есиль", "Нура", "Сарайшык", "Сарыарка"]
