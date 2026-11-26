"""Геопризнаки по данным OpenStreetMap: расстояния до ближайших объектов и число объектов в радиусе (BallTree, haversine)."""
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from sklearn.neighbors import BallTree

from . import osm
from .config import CENTER

EARTH_KM = osm.EARTH_KM
RADII_KM = {"500m": 0.5, "1km": 1.0}
NEAR_RIVER_KM = 0.5            # «у берега Есиля»: не дальше 500 м от русла
COUNT_LAYERS = osm.POINT_LAYERS + ["park"]


def _tree(pts):
    return BallTree(np.radians(pts), metric="haversine")


class GeoIndex:
    """Индекс объектов OSM. Считает геопризнаки и район для любых координат — и для набора, и для сервиса прогноза."""

    def __init__(self, points, park_pts, park_ids, river_pts, districts):
        self.points, self.park_pts, self.park_ids = points, park_pts, park_ids
        self.river_pts, self.districts = river_pts, districts
        self._build()

    def _build(self):
        self._trees = {k: _tree(v) for k, v in self.points.items()}
        self._park_tree, self._river_tree = _tree(self.park_pts), _tree(self.river_pts)
        self._paths = {d: [MplPath(r[:, ::-1]) for r in rings] for d, rings in self.districts.items()}

    @classmethod
    def from_osm(cls):
        points = {layer: osm.parse_points(layer)[0] for layer in osm.POINT_LAYERS}
        park_pts, park_ids, _, _ = osm.parse_parks()
        river_pts, _ = osm.parse_river()
        return cls(points, park_pts, park_ids, river_pts, osm.parse_districts())

    def __getstate__(self):      # деревья и контуры пересобираются при загрузке
        return {k: getattr(self, k) for k in ("points", "park_pts", "park_ids", "river_pts", "districts")}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._build()

    def district(self, lat, lon):
        """Район Астаны по границам OSM; None — точка вне города."""
        lat, lon = np.atleast_1d(np.asarray(lat, float)), np.atleast_1d(np.asarray(lon, float))
        xy = np.column_stack([lon, lat])
        out = np.full(len(lat), None, dtype=object)
        for name, paths in self._paths.items():
            inside = np.zeros(len(lat), dtype=bool)
            for p in paths:
                inside |= p.contains_points(xy)
            out[inside & (out == None)] = name      # noqa: E711
        return out

    def features(self, lat, lon):
        """Таблица геопризнаков для точек (lat, lon)."""
        lat, lon = np.atleast_1d(np.asarray(lat, float)), np.atleast_1d(np.asarray(lon, float))
        q = np.radians(np.column_stack([lat, lon]))
        f = {"dist_center_km": osm.haversine_km(lat, lon, *CENTER)}
        f["dist_river_km"] = self._river_tree.query(q, k=1)[0][:, 0] * EARTH_KM
        f["near_river"] = (f["dist_river_km"] <= NEAR_RIVER_KM).astype(int)
        for layer, tree in self._trees.items():
            f[f"dist_{layer}_km"] = tree.query(q, k=1)[0][:, 0] * EARTH_KM
            for tag, r in RADII_KM.items():
                f[f"n_{layer}_{tag}"] = tree.query_radius(q, r=r / EARTH_KM, count_only=True)
        f["dist_park_km"] = self._park_tree.query(q, k=1)[0][:, 0] * EARTH_KM
        for tag, r in RADII_KM.items():       # парк считается один раз, сколько бы точек его границы ни попало в радиус
            idx = self._park_tree.query_radius(q, r=r / EARTH_KM)
            f[f"n_park_{tag}"] = np.array([len(np.unique(self.park_ids[i])) for i in idx])
        return pd.DataFrame(f)


GEO_FEATURES = (["dist_center_km", "dist_river_km", "near_river"]
                + [f"dist_{k}_km" for k in COUNT_LAYERS]
                + [f"n_{k}_{t}" for k in COUNT_LAYERS for t in RADII_KM])
