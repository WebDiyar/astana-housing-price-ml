"""Запуск всех автотестов с записью итога в results/tests.json:  python tests/run_tests.py"""
import json
import pathlib
import sys
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def flatten(suite):
    for t in suite:
        yield from flatten(t) if isinstance(t, unittest.TestSuite) else [t]


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    per_module = {}
    for t in flatten(suite):
        name = t.id().split(".")[1]
        per_module[name] = per_module.get(name, 0) + 1
    t0 = time.perf_counter()
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    bad = len(res.failures) + len(res.errors)
    out = {"total": res.testsRun, "passed": res.testsRun - bad - len(res.skipped), "failed": len(res.failures),
           "errors": len(res.errors), "skipped": len(res.skipped), "time_s": round(time.perf_counter() - t0, 1),
           "per_module": dict(sorted(per_module.items()))}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "tests.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"всего {out['total']}, пройдено {out['passed']}, не пройдено {bad}, пропущено {out['skipped']}")
    sys.exit(1 if bad else 0)
