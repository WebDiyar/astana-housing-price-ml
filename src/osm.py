"""Загрузка и разбор открытых геоданных OpenStreetMap (Overpass API).

Данные: © участники OpenStreetMap, лицензия ODbL 1.0 (https://www.openstreetmap.org/copyright).
Сырые ответы сохраняются без изменений в data/raw/osm/ (два запроса: точечные объекты и геометрия),
сведения о запросах — в _manifest.json.

    python -m src.osm            # скачать недостающие файлы
    python -m src.osm --force    # скачать заново
"""
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np

from .config import BBOX, OSM_DIR

LICENSE_NOTE = "© участники OpenStreetMap, ODbL 1.0 — https://www.openstreetmap.org/copyright"
ENDPOINTS = [   # (адрес, время ожидания ответа в секундах)
    ("https://overpass-api.de/api/interpreter", 300),
    ("https://overpass.kumi.systems/api/interpreter", 90),
    ("https://maps.mail.ru/osm/tools/overpass/api/interpreter", 90),
    ("https://overpass.private.coffee/api/interpreter", 90),
]
USER_AGENT = "osm-poi-research-script/1.0"   # без личных данных
PAUSE = 15                                    # пауза между запросами, с (при отказе удваивается)

# Слой -> (название, условие отбора по тегам). Точечные слои идут в признаки «расстояние / число в радиусе».
LAYERS = {
    "school":       ("школы",                          lambda e, t: t.get("amenity") == "school"),
    "kindergarten": ("детские сады",                   lambda e, t: t.get("amenity") == "kindergarten"),
    "healthcare":   ("больницы и поликлиники",         lambda e, t: t.get("amenity") in ("hospital", "clinic")),
    "university":   ("вузы и колледжи",                lambda e, t: t.get("amenity") in ("university", "college")),
    "mall":         ("торговые центры",                lambda e, t: t.get("shop") == "mall"),
    "supermarket":  ("супермаркеты",                   lambda e, t: t.get("shop") == "supermarket"),
    "pharmacy":     ("аптеки",                         lambda e, t: t.get("amenity") == "pharmacy"),
    "food":         ("кафе и рестораны",               lambda e, t: t.get("amenity") in ("restaurant", "cafe", "fast_food")),
    "bus_stop":     ("остановки общественного транспорта", lambda e, t: e["type"] == "node" and t.get("highway") == "bus_stop"),
    "park":         ("парки",                          lambda e, t: e["type"] != "node" and t.get("leisure") == "park"),
    "river":        ("реки",                           lambda e, t: e["type"] == "way" and t.get("waterway") == "river"),
    "districts":    ("административные границы 6-го уровня", lambda e, t: e["type"] == "relation" and t.get("boundary") == "administrative" and t.get("admin_level") == "6"),
}
POINT_LAYERS = ["school", "kindergarten", "healthcare", "university", "mall", "supermarket", "pharmacy", "food", "bus_stop"]
GEOM_LAYERS = ["park", "river", "districts"]
# Файл -> (запрос Overpass QL без рамки, режим вывода)
REQUESTS = {
    "osm_poi": ('nwr["amenity"~"^(school|kindergarten|hospital|clinic|university|college|pharmacy|restaurant|cafe|fast_food)$"];'
                'nwr["shop"~"^(mall|supermarket)$"];node["highway"="bus_stop"];', "tags center"),
    "osm_geometry": ('way["leisure"="park"];relation["leisure"="park"];way["waterway"="river"];'
                     'relation["boundary"="administrative"]["admin_level"="6"];', "body geom"),   # body: у отношений нужны участники
}
RIVER_NAME = ("есіл", "есиль", "ишим", "esil", "ishim", "yesil")
DISTRICT_NAMES = {   # name:ru в OSM -> короткое название района
    "район Алматы": "Алматы", "район Байконур": "Байконур", "район Есиль": "Есиль",
    "район Нура": "Нура", "район Сарайшык": "Сарайшык", "район Сарыарка": "Сарыарка",
}


def build_query(name):
    flt, mode = REQUESTS[name]
    s, w, n, e = BBOX
    return f"[out:json][timeout:240][bbox:{s},{w},{n},{e}];({flt});out {mode};"


def _post(endpoint, query, timeout):
    req = urllib.request.Request(endpoint, data=urllib.parse.urlencode({"data": query}).encode(),
                                 headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch(name, rounds=5, log=print):
    """Возвращает (сырой ответ, адрес зеркала). При отказе — следующее зеркало, затем повтор с удвоенной паузой."""
    query, last, pause = build_query(name), None, PAUSE
    for _ in range(rounds):
        for ep, timeout in ENDPOINTS:
            try:
                raw = _post(ep, query, timeout)
                data = json.loads(raw)
                if "elements" not in data or "error" in data.get("remark", ""):
                    raise ValueError(data.get("remark", "нет поля elements"))
                return raw, ep
            except (urllib.error.URLError, TimeoutError, ValueError, ConnectionError, OSError) as ex:
                last = ex
                log(f"  {name}: {ep.split('/')[2]} — {type(ex).__name__}: {str(ex)[:80]}")
                time.sleep(5)
        log(f"  {name}: пауза {pause} с")
        time.sleep(pause)
        pause *= 2
    raise RuntimeError(f"{name} не получен: {last}")


def download(force=False, log=print):
    """Скачивает недостающие файлы, пишет _manifest.json (дата запроса, зеркало, запрос, число объектов, SHA-256)."""
    OSM_DIR.mkdir(parents=True, exist_ok=True)
    mpath = OSM_DIR / "_manifest.json"
    man = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {"license": LICENSE_NOTE, "bbox_s_w_n_e": BBOX, "requests": {}}
    for name in REQUESTS:
        path = OSM_DIR / f"{name}.json"
        if path.exists() and name in man["requests"] and not force:
            continue
        raw, ep = fetch(name, log=log)
        path.write_bytes(raw)
        data = json.loads(raw)
        man["requests"][name] = {
            "file": path.name, "query": build_query(name), "endpoint": ep,
            "requested_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "osm_base": data.get("osm3s", {}).get("timestamp_osm_base"),
            "elements": len(data["elements"]), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        }
        mpath.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"{name}: {len(data['elements'])} объектов ({ep.split('/')[2]})")
        time.sleep(PAUSE)
    man["layers"] = {k: {"title": LAYERS[k][0], "elements": len(load_raw(k))} for k in LAYERS}
    mpath.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    return man


_CACHE = {}


def load_raw(layer):
    """Объекты слоя из сырого ответа (отбор по тегам)."""
    name = "osm_geometry" if layer in GEOM_LAYERS else "osm_poi"
    if name not in _CACHE:
        _CACHE[name] = json.loads((OSM_DIR / f"{name}.json").read_text(encoding="utf-8"))["elements"]
    cond = LAYERS[layer][1]
    return [e for e in _CACHE[name] if cond(e, e.get("tags", {}))]


def manifest():
    return json.loads((OSM_DIR / "_manifest.json").read_text(encoding="utf-8"))


# ---------- разбор ----------

EARTH_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


def _latlon(el):
    if el["type"] == "node":
        return el.get("lat"), el.get("lon")
    c = el.get("center") or {}
    return c.get("lat"), c.get("lon")


def parse_points(layer, dedup_m=150.0):
    """Точки слоя (lat, lon). Объект, отмеченный и точкой, и контуром (одно название, ближе dedup_m), считается один раз."""
    rows = []
    for el in sorted(load_raw(layer), key=lambda e: (e["type"] != "node", e["id"])):
        lat, lon = _latlon(el)
        if lat is None:
            continue
        name = (el.get("tags", {}).get("name") or "").strip().lower()
        rows.append((lat, lon, name))
    raw_n = len(rows)
    kept, by_name = [], {}
    for lat, lon, name in rows:
        if layer != "bus_stop" and name:
            near = by_name.setdefault(name, [])
            if any(haversine_km(lat, lon, a, b) * 1000 < dedup_m for a, b in near):
                continue
            near.append((lat, lon))
        kept.append((lat, lon))
    return np.array(kept, dtype=float), raw_n


def _densify(coords, step_m=25.0):
    """Добавляет точки на отрезках ломаной не реже чем через step_m метров."""
    out = [coords[0]]
    for (a, b), (c, d) in zip(coords[:-1], coords[1:]):
        k = int(haversine_km(a, b, c, d) * 1000 // step_m)
        for i in range(1, k + 1):
            t = i / (k + 1)
            out.append((a + (c - a) * t, b + (d - b) * t))
        out.append((c, d))
    return out


def _ways(el):
    """Ломаные объекта: у линии — её геометрия, у отношения — геометрии внешних участников."""
    if el["type"] == "way":
        return [[(p["lat"], p["lon"]) for p in el.get("geometry", [])]]
    return [[(p["lat"], p["lon"]) for p in m["geometry"]] for m in el.get("members", [])
            if m["type"] == "way" and m.get("role") != "inner" and m.get("geometry")]


def _ring_area_ha(ring):
    lat0 = np.radians(np.mean([p[0] for p in ring]))
    x = np.array([p[1] for p in ring]) * np.cos(lat0) * 111.32
    y = np.array([p[0] for p in ring]) * 110.57
    return abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2 * 100   # км² -> га


def parse_parks(min_ha=1.0):
    """Парки не меньше min_ha: точки границы (через 25 м) с номером парка. Скверы меньше порога не учитываются."""
    pts, ids, n_all, n_kept = [], [], 0, 0
    for el in load_raw("park"):
        ways = [w for w in _ways(el) if len(w) >= 3]
        if not ways:
            continue
        n_all += 1
        if sum(_ring_area_ha(w) for w in ways) < min_ha:
            continue
        for w in ways:
            d = _densify(w)
            pts += d
            ids += [n_kept] * len(d)
        n_kept += 1
    return np.array(pts, dtype=float), np.array(ids), n_all, n_kept


def parse_river():
    """Точки русла Есиля (Ишима) через 25 м."""
    pts, n = [], 0
    for el in load_raw("river"):
        t = el.get("tags", {})
        names = " ".join(str(t.get(k, "")) for k in ("name", "name:ru", "name:kk", "name:en")).lower()
        if any(s in names for s in RIVER_NAME) and el.get("geometry"):
            pts += _densify([(p["lat"], p["lon"]) for p in el["geometry"]])
            n += 1
    return np.array(pts, dtype=float), n


def _stitch(ways):
    """Собирает замкнутые кольца из отрезков границы (совпадение концов)."""
    segs, rings = [list(w) for w in ways if len(w) >= 2], []
    while segs:
        ring = segs.pop(0)
        while ring[0] != ring[-1]:
            for i, s in enumerate(segs):
                if s[0] == ring[-1]:
                    ring += s[1:]
                elif s[-1] == ring[-1]:
                    ring += s[-2::-1]
                else:
                    continue
                segs.pop(i)
                break
            else:
                break                      # кольцо не замкнулось — отбрасываем
        if ring[0] == ring[-1] and len(ring) >= 4:
            rings.append(ring)
    return rings


def parse_districts():
    """Границы шести районов Астаны: {район: [кольцо (lat, lon), ...]}."""
    out = {}
    for el in load_raw("districts"):
        name = DISTRICT_NAMES.get(el.get("tags", {}).get("name:ru"))
        if name:
            out[name] = [np.array(r, dtype=float) for r in _stitch(_ways(el))]
    return out


def layer_counts():
    """Сколько объектов каждого слоя получено и сколько использовано в признаках."""
    man = manifest()["layers"]
    rows = []
    for layer in POINT_LAYERS:
        pts, raw_n = parse_points(layer)
        rows.append((layer, LAYERS[layer][0], man[layer]["elements"], len(pts), "без повторов «точка + контур» одного объекта"))
    _, _, n_all, n_kept = parse_parks()
    rows.append(("park", LAYERS["park"][0], man["park"]["elements"], n_kept, "парки площадью от 1 га"))
    _, n_riv = parse_river()
    rows.append(("river", LAYERS["river"][0], man["river"]["elements"], n_riv, "отрезки русла Есиля (Ишима)"))
    rows.append(("districts", LAYERS["districts"][0], man["districts"]["elements"], len(parse_districts()), "районы Астаны"))
    return rows


if __name__ == "__main__":
    download(force="--force" in sys.argv)
    for r in layer_counts():
        print(r)
