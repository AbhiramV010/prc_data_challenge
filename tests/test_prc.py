import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import prc


def s(values, dtype="string"):
    return pd.Series(values, dtype=dtype)


class TestDotenvAndKeys(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        # minio_keys looks next to prc.__file__; point it at a temp folder, never the real repo.
        patcher = mock.patch.object(prc, "__file__", str(self.dir / "prc.py"))
        patcher.start()
        self.addCleanup(patcher.stop)
        env = {k: v for k, v in os.environ.items() if k not in ("PRC_ACCESS_KEY", "PRC_SECRET_KEY")}
        env_patch = mock.patch.dict(os.environ, env, clear=True)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_read_dotenv_formats(self):
        p = self.dir / ".env"
        p.write_text('# comment\nPRC_ACCESS_KEY = "abc"\nexport PRC_SECRET_KEY=\'xyz\'\n\nNOEQUALS\n',
                     encoding="utf-8")
        self.assertEqual(prc.read_dotenv(p), {"PRC_ACCESS_KEY": "abc", "PRC_SECRET_KEY": "xyz"})

    def test_read_dotenv_missing_file(self):
        self.assertEqual(prc.read_dotenv(self.dir / "nope.env"), {})

    def test_keys_from_dotenv(self):
        (self.dir / ".env").write_text("PRC_ACCESS_KEY=a\nPRC_SECRET_KEY=b\n", encoding="utf-8")
        self.assertEqual(prc.minio_keys(), ("a", "b"))

    def test_environment_beats_dotenv(self):
        (self.dir / ".env").write_text("PRC_ACCESS_KEY=a\nPRC_SECRET_KEY=b\n", encoding="utf-8")
        with mock.patch.dict(os.environ, {"PRC_ACCESS_KEY": "envA", "PRC_SECRET_KEY": "envB"}):
            self.assertEqual(prc.minio_keys(), ("envA", "envB"))

    def test_rest_credentials_are_rejected(self):
        (self.dir / "credentials.json").write_text('{"clientId": "x", "clientSecret": "y"}')
        with self.assertRaises(SystemExit) as cm:
            prc.minio_keys()
        self.assertIn("clientId", str(cm.exception))
        self.assertNotIn('"y"', str(cm.exception))

    def test_no_keys_exits(self):
        with self.assertRaises(SystemExit):
            prc.minio_keys()


class TestTimeHelpers(unittest.TestCase):
    def test_to_utc_naive_aware_and_strings(self):
        naive = pd.Series(pd.to_datetime(["2025-01-01 00:00"]))
        aware = pd.Series(pd.to_datetime(["2025-01-01 01:00"]).tz_localize("Europe/Paris"))
        text = pd.Series(["2025-01-01T00:00:00Z", "garbage"])
        for series in (naive, aware, text):
            out = prc.to_utc(series)
            self.assertEqual(str(out.dt.tz), "UTC")
            self.assertEqual(out.iloc[0], pd.Timestamp("2025-01-01", tz="UTC"))
        self.assertTrue(pd.isna(prc.to_utc(text).iloc[1]))

    def test_to_secs(self):
        ts = pd.Series([pd.Timestamp("1970-01-01 00:01:40", tz="UTC"), pd.NaT])
        out = prc.to_secs(ts)
        self.assertEqual(out.iloc[0], 100.0)
        self.assertTrue(np.isnan(out.iloc[1]))


class TestCounting(unittest.TestCase):
    ev_key = s(["A", "A", "A", "B"])
    ev_t = pd.Series([0.0, 100.0, 1000.0, 50.0])

    def test_count_le_sides_and_keys(self):
        q_key, q_t = s(["A", "A", "B", "C"]), pd.Series([100.0, 500.0, 50.0, 10.0])
        np.testing.assert_array_equal(prc.count_le(self.ev_key, self.ev_t, q_key, q_t), [2, 2, 1, 0])
        np.testing.assert_array_equal(prc.count_le(self.ev_key, self.ev_t, q_key, q_t, "left"),
                                      [1, 2, 0, 0])

    def test_count_le_invalid_queries_are_nan(self):
        out = prc.count_le(self.ev_key, self.ev_t, s(["A", None]), pd.Series([np.nan, 5.0]))
        self.assertTrue(np.isnan(out).all())

    def test_window_counts(self):
        q_key, q_t = s(["A", "B", "C", None]), pd.Series([100.0] * 4)
        out = prc.window_counts(self.ev_key, self.ev_t, q_key, q_t, prc.DEP_WINDOWS)
        np.testing.assert_array_equal(out["dep_prev10"], [1, 1, 0, np.nan])  # B event at 50
        np.testing.assert_array_equal(out["dep_next15"], [1, 0, 0, np.nan])
        np.testing.assert_array_equal(out["dep_pm30"], [3, 1, 0, np.nan])

    def test_window_means(self):
        vals = pd.Series([10.0, 20.0, 30.0, 40.0])
        q_key, q_t = s(["A", "A", "C"]), pd.Series([100.0, 1001.0, 10.0])
        out = prc.window_means(self.ev_key, self.ev_t, vals, q_key, q_t, [("past", -1000, 0), ("all", -2000, 2000)])
        # A events at 0, 100, 1000 with values 10, 20, 30; the window end is open, so time 100 itself is out.
        np.testing.assert_array_equal(out["past"], [10.0, 25.0, np.nan])
        np.testing.assert_array_equal(out["all"], [20.0, 20.0, np.nan])

    def test_neighbor_gaps_excludes_self_and_other_keys(self):
        q_key, q_t = s(["A", "A", "B", "C"]), pd.Series([100.0, 500.0, 50.0, 10.0])
        prev, nxt = prc.neighbor_gaps(self.ev_key, self.ev_t, q_key, q_t)
        np.testing.assert_array_equal(prev, [100, 400, np.nan, np.nan])
        np.testing.assert_array_equal(nxt, [900, 500, np.nan, np.nan])


def tiny_departures():
    """Three departures and one arrival on runway A|R; times in seconds."""
    n = 3
    dep = pd.DataFrame({
        "airport": s(["A"] * n), "RUNWAY_mvt": s(["R"] * n), "STAND_mvt": s(["S1", "S2", "S1"]),
        "AIRCRAFT_TYPE_mvt": s([None, "A320", "B738"]), "AIRCRAFT_TYPE_flt": s(["A21N", "X", "X"]),
        "ADES_mvt": s(["EGLL"] * n), "ADES_flt": s(["EGLL"] * n),
        "FLIGHT_RULE_mvt": s(["I"] * n), "FLIGHT_RULE_flt": s(["I"] * n),
        "WK_TBL_CAT_flt": s(["M"] * n), "MARKET_SEGMENT_flt": s(["Mainline"] * n),
        "AIRCRAFT_OPERATOR_flt": s(["OP1"] * n), "FLIGHT_TYPE_flt": s(["S"] * n),
        "FLIGHT_mvt": s(["RYR1234", "KL1521", None]),
        "hour": [0.0] * n, "dow": [0] * n, "unmatched": [0.0, 1.0, 0.0],
        "t": [600.0, 700.0, 1000.0], "obt": [0.0, 100.0, 650.0],
        "SCHED_TIME_UTC_mvt": [0.0] * n, "EOBT_1_flt": [0.0] * n, "IOBT_flt": [0.0] * n,
        "AOBT_3_flt": [0.0] * n, "LOBT_flt": [0.0] * n,
    })
    events = pd.DataFrame({
        "airport": s(["A"] * 4), "runway": s(["R"] * 4), "stand": s(["S1", "S2", "S1", "S1"]),
        "dep": [True, True, True, False],
        "t": [600.0, 700.0, 1000.0, 300.0], "obt": [0.0, 100.0, 650.0, np.nan],
        "inblk": [np.nan, np.nan, np.nan, 500.0], "arr_stale": [np.nan, np.nan, np.nan, 1.0],
    })
    return dep, events


class TestBuildFeatures(unittest.TestCase):
    def setUp(self):
        dep, events = tiny_departures()
        self.X = prc.build_features(dep, events, use_obt=True)

    def test_categorical_fallbacks(self):
        self.assertEqual(self.X["actype"].tolist(), ["A21N", "A320", "B738"])
        self.assertEqual(self.X["apt_rwy"].iloc[0], "A|R")
        self.assertEqual(self.X["apt_stand"].iloc[1], "A|S2")
        self.assertEqual(self.X["operator"].tolist(), ["OP1"] * 3)
        self.assertEqual(self.X["airline"].tolist()[:2], ["RYR", "KL"])
        self.assertTrue(pd.isna(self.X["airline"].iloc[2]))

    def test_queue_features_for_middle_departure(self):
        r = self.X.iloc[1]  # pushback 100, takeoff 700
        self.assertEqual(r["taxi_window"], 600)
        self.assertEqual(r["rwy_deps_in_taxi"], 1)   # first departure took off at 600
        self.assertEqual(r["rwy_arrs_in_taxi"], 1)   # arrival landed at 300
        self.assertEqual(r["rwy_gap_prev_dep"], 100)
        self.assertEqual(r["rwy_gap_next_dep"], 300)
        self.assertEqual(r["deps_taxiing_at_obt"], 2)  # itself and the first departure
        self.assertEqual(r["deps_taxiing_at_t"], 1)
        self.assertEqual(r["arrs_taxiing_at_obt"], 0)
        self.assertEqual(r["arrs_taxiing_at_t"], 0)   # in-block at 500
        self.assertEqual(r["pushbacks_prev15"], 2)

    def test_arrival_stale_shares(self):
        # The one arrival (stand S1, landed at 300) was logged stale; it precedes every departure.
        np.testing.assert_array_equal(self.X["arr_stale_apt_m180"], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(self.X["arr_stale_stand_m720"], [1.0, np.nan, 1.0])

    def test_first_departure_has_no_previous_gap(self):
        self.assertTrue(np.isnan(self.X["rwy_gap_prev_dep"].iloc[0]))

    def test_duplicate_record_flags(self):
        self.assertEqual(self.X["actype_mvt_missing"].tolist(), [1.0, 0.0, 0.0])
        # Only the first two departures have callsigns, each used once.
        self.assertEqual(self.X["cs_deps_same_day"].tolist()[:2], [1.0, 1.0])
        self.assertTrue(np.isnan(self.X["cs_deps_same_day"].iloc[2]))
        self.assertTrue(self.X["cs_gap_nearest"].isna().all())

    def test_obt_features_only_with_flag(self):
        dep, events = tiny_departures()
        self.assertNotIn("mvt_minus_aobt3", prc.build_features(dep, events, use_obt=False))
        self.assertIn("mvt_minus_aobt3", self.X)


class TestModelHelpers(unittest.TestCase):
    def test_encode_cats_drops_rare_and_unseen(self):
        X = pd.DataFrame({c: s(["x"] * 40 + ["rare", "new"]) for c in prc.CAT_FEATS})
        fit = np.array([True] * 41 + [False])
        out = prc.encode_cats(X, fit)
        col = out["airport"]
        self.assertIsInstance(col.dtype, pd.CategoricalDtype)
        self.assertEqual(list(col.cat.categories), ["x"])
        self.assertTrue(col.iloc[40:].isna().all())

    def test_fit_masks(self):
        dep = pd.DataFrame({prc.TARGET: [np.nan, -5.0, 0.0, 600.0, 3600.0, 3601.0, 900.0, 700.0],
                            "stale": [np.nan, 0, 0, 0, 0, 0, 1, np.nan]})
        reg, labelled = prc.fit_masks(dep, pd.Series([True] * 8), False, 3600)
        # Regressor: normal (non-stale) rows with 0 < y <= max_target.
        np.testing.assert_array_equal(reg, [False, False, False, True, True, False, False, False])
        # Classifier: every row with a target and a stale label.
        np.testing.assert_array_equal(labelled, [False, True, True, True, True, True, True, False])

    def test_predict_mix(self):
        class Fixed:
            def __init__(self, values=(), raw=()):
                self.values, self.raw = np.array(values, dtype=float), np.array(raw, dtype=float)

            def predict(self, X, raw_score=False):
                return self.raw if raw_score else self.values

        X = pd.DataFrame({"mvt_minus_sched": [1000.0, 5000.0, np.nan, -100.0]})
        reg = Fixed([800.0, 800.0, 700.0, -50.0])
        pred, p, normal = prc.predict_mix({"stale": Fixed([0.0, 1.0, 0.5, 1.0]), "reg": reg}, X)
        # Stale rows take takeoff - scheduled; no schedule falls back to the normal model.
        np.testing.assert_allclose(pred, [800.0, 5000.0, 700.0, 0.0])
        np.testing.assert_array_equal(normal, [800.0, 800.0, 700.0, 0.0])

        # With the blend booster, P(stale) is the sigmoid of the stale and mix logits summed.
        models = {"stale": Fixed(raw=[0.0] * 4), "mix": Fixed(raw=[0.0, np.log(3), 0.0, 0.0]), "reg": reg}
        pred, p, _ = prc.predict_mix(models, X)
        np.testing.assert_allclose(p, [0.5, 0.75, 0.5, 0.5])
        np.testing.assert_allclose(pred, [900.0, 3950.0, 700.0, 0.0])

    def test_predict_mix_late_unmatched_uses_airport_rates(self):
        class Fixed:
            def __init__(self, values):
                self.values = np.array(values, dtype=float)

            def predict(self, X, raw_score=False):
                return self.values

        late = prc.LATE_UNMATCHED_SEC + 1000.0
        X = pd.DataFrame({"mvt_minus_sched": [late, late, late, 5000.0],
                          "airport": ["LIRF", "EDDF", "LEMD", "LIRF"],
                          "unmatched": [1.0, 1.0, 1.0, 1.0]})
        models = {"stale": Fixed([0.0, 0.9, 0.9, 0.0]), "reg": Fixed([1000.0, 900.0, 800.0, 700.0]),
                  "late_rates": {"LIRF": (0.5, 0.25), "EDDF": (0.0, 0.0)}}
        pred, _, _ = prc.predict_mix(models, X)
        lirf = 0.5 * late + 0.25 * (prc.DAY_SEC + 1000.0) + 0.25 * 1000.0
        # LIRF and EDDF use their rates; LEMD has none so keeps the classifier; row 4 is not late.
        np.testing.assert_allclose(pred, [lirf, 900.0, 0.9 * late + 0.1 * 800.0, 700.0])

    def test_late_unmatched_rates(self):
        late = prc.LATE_UNMATCHED_SEC + 1.0
        X = pd.DataFrame({"mvt_minus_sched": [late, late, late, late, 100.0, late],
                          "airport": ["LIRF", "LIRF", "LIRF", "LIRF", "LIRF", "EDDF"],
                          "unmatched": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]})
        y = pd.Series([late, late, prc.DAY_SEC + 900.0, 1200.0, 50000.0, 800.0])
        stale = pd.Series([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        rates = prc.late_unmatched_rates(X, y, stale, np.ones(6, dtype=bool))
        # Row 5 is not late, so LIRF has 4 rows: 2 stale, 1 a day early, 1 normal.
        self.assertEqual(rates, {"LIRF": (0.5, 0.25), "EDDF": (0.0, 0.0)})
        # Matched flights and rows outside the mask are ignored.
        X["unmatched"] = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        mask = np.array([True, True, True, True, True, False])
        self.assertEqual(prc.late_unmatched_rates(X, y, stale, mask), {"LIRF": (1 / 3, 1 / 3)})

    def test_blend_parts_cap_bounds_stale_term(self):
        n, m = prc.blend_parts(np.array([500.0, 500.0, 500.0]), np.array([np.nan, 90000.0, 800.0]), cap=3600)
        np.testing.assert_array_equal(m, [500.0, 3600.0, 800.0])
        n, m = prc.blend_parts(np.array([500.0]), np.array([90000.0]))
        self.assertEqual(m[0], 90000.0)

    def test_predict_mix_cap_keeps_predictions_bounded(self):
        class Fixed:
            def __init__(self, values):
                self.values = np.array(values, dtype=float)

            def predict(self, X, raw_score=False):
                return self.values

        X = pd.DataFrame({"mvt_minus_sched": [90000.0, 800.0]})
        models = {"stale": Fixed([0.5, 0.5]), "reg": Fixed([700.0, 700.0])}
        uncapped, _, _ = prc.predict_mix(models, X)
        capped, _, _ = prc.predict_mix({**models, "cap": 3600.0}, X)
        np.testing.assert_allclose(uncapped, [45350.0, 750.0])
        np.testing.assert_allclose(capped, [2150.0, 750.0])  # the stale term is limited to max_target

    def test_predict_mix_p_scale_shrinks_the_stale_term(self):
        class Fixed:
            def __init__(self, values):
                self.values = np.array(values, dtype=float)

            def predict(self, X, raw_score=False):
                return self.values

        X = pd.DataFrame({"mvt_minus_sched": [90000.0, 800.0]})
        models = {"stale": Fixed([0.5, 0.5]), "reg": Fixed([700.0, 700.0]), "p_scale": 0.5}
        pred, p, _ = prc.predict_mix(models, X)
        np.testing.assert_allclose(p, [0.25, 0.25])
        np.testing.assert_allclose(pred, [0.25 * 90000 + 0.75 * 700, 0.25 * 800 + 0.75 * 700])

    def test_submit_rounds_rescales_for_learning_rate(self):
        best = {"reg": 269, "stale": 75, "mix": 79, "lr": 0.1}
        # validate --fast (lr 0.1) then a full fit (lr 0.05): twice the rounds, blend booster unchanged.
        self.assertEqual(prc.submit_rounds(best, fast=False), {"reg": 591, "stale": 165, "mix": 86})
        self.assertEqual(prc.submit_rounds(best, fast=True), {"reg": 295, "stale": 82, "mix": 86})
        old = {"reg": 269, "stale": 75, "mix": 79}  # no "lr": no rescaling
        self.assertEqual(prc.submit_rounds(old, fast=False), {"reg": 295, "stale": 82, "mix": 86})
        self.assertEqual(prc.submit_rounds({"reg": 10, "stale": 10, "mix": 10, "lr": 0.1}, fast=True),
                         {"reg": 50, "stale": 50, "mix": 50})

    def test_mix_rows(self):
        X = pd.DataFrame({"unmatched": [1.0, 1.0, 0.0, 1.0],
                          "mvt_minus_sched": [prc.LATE_UNMATCHED_SEC + 1, 100.0, 100.0, 100.0]})
        y = pd.Series([50000.0, 800.0, 90000.0, -5.0])
        mask = np.ones(4, dtype=bool)
        # Capped mode keeps only 0 < y <= max_target; tail mode drops only late unmatched rows.
        np.testing.assert_array_equal(prc.mix_rows(X, y, mask, False, 3600), [False, True, False, False])
        np.testing.assert_array_equal(prc.mix_rows(X, y, mask, True, 3600), [False, True, True, True])

    def test_stack_features(self):
        X = pd.DataFrame({"mvt_minus_sched": [900.0, 5000.0, np.nan]})
        out = prc.stack_features(X, np.array([1000.0, -20.0, 700.0]))
        np.testing.assert_array_equal(out["n_hat"], [1000.0, 0.0, 700.0])  # normal prediction is clipped at 0
        np.testing.assert_array_equal(out["ms_minus_n"], [-100.0, 5000.0, np.nan])
        np.testing.assert_array_equal(out["abs_ms_minus_n"], [100.0, 5000.0, np.nan])
        self.assertNotIn("n_hat", X)  # the input frame is left alone

    def test_blend_parts(self):
        n, m = prc.blend_parts(np.array([-5.0, 100.0, 200.0]), np.array([np.nan, -30.0, 500.0]))
        np.testing.assert_array_equal(n, [0.0, 100.0, 200.0])
        np.testing.assert_array_equal(m, [0.0, 0.0, 500.0])  # no schedule -> normal; clipped at 0

    def test_rmse(self):
        self.assertAlmostEqual(prc.rmse([0, 0], [3, 4]), np.sqrt(12.5))

    def test_reporting_airport_with_flight_fallback(self):
        df = pd.DataFrame({"ADEP_mvt": ["EDDF", None], "ADES_mvt": ["KJFK", "EGLL"],
                           "ADEP_flt": ["X", "LFPG"], "ADES_flt": ["Y", "Z"]})
        out = prc.reporting_airport(df, pd.Series([True, True]))
        self.assertEqual(out.tolist(), ["EDDF", "LFPG"])
        out = prc.reporting_airport(df, pd.Series([False, False]))
        self.assertEqual(out.tolist(), ["KJFK", "EGLL"])


def run_cli(*args, root):
    return subprocess.run([sys.executable, str(ROOT / "prc.py"), "--root", str(root), *args],
                          capture_output=True, text=True, cwd=root, check=False)


RESULTS_CSV = ROOT / "results.csv"
RESULT_FIELDS = ["run_at", "dataset", "rows_clean", "model_rmse_clean", "baseline_rmse_clean",
                 "model_rmse_all", "baseline_rmse_all", "best_iter"]
RMSE_RESULTS: list[dict] = []
FINAL_RMSE: list[float] = []  # competition-data model RMSE (clean rows), shown after all tests


def report_rmse(label, stdout):
    """Keep validate's RMSE values for the results.csv rows written after all tests."""
    clean = re.search(r"RMSE clean \(.*?([\d,]+) rows\): model ([\d.]+)\s.*?airport-mean baseline ([\d.]+)",
                      stdout)
    every = re.search(r"RMSE all rows \([\d,]+\): model ([\d.]+)\s.*?airport-mean baseline ([\d.]+)", stdout)
    best = re.search(r"best_iter\.json: \{'reg': (\d+)", stdout)  # normal-taxi regressor's best round
    assert clean and every and best, f"validate output format changed:\n{stdout}"
    RMSE_RESULTS.append({
        "dataset": label, "rows_clean": int(clean.group(1).replace(",", "")),
        "model_rmse_clean": float(clean.group(2)), "baseline_rmse_clean": float(clean.group(3)),
        "model_rmse_all": float(every.group(1)), "baseline_rmse_all": float(every.group(2)),
        "best_iter": int(best.group(1)),
    })


def rmse_summary():
    """The competition-data RMSE, printed after all tests (see conftest.py)."""
    return f"Final RMSE: {FINAL_RMSE[-1]:.2f}" if FINAL_RMSE else ""


def tearDownModule():
    if "pytest" not in sys.modules and FINAL_RMSE:  # under pytest, conftest.py prints it so capturing can't hide it
        print("\n" + rmse_summary(), file=sys.stderr)
    if RMSE_RESULTS:
        run_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        new = not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0
        with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
            if new:
                writer.writeheader()
            writer.writerows({"run_at": run_at, **row} for row in RMSE_RESULTS)
        print(f"Results appended to {RESULTS_CSV}", file=sys.stderr)


class TestEndToEnd(unittest.TestCase):
    """Synthetic data through validate, submit and upload's pre-flight checks."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        r = run_cli("synth", "--rows", "1500", root=cls.root)
        assert r.returncode == 0, r.stderr

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_validate_then_submit_matches_template(self):
        r = run_cli("validate", "--fast", "--use-actual-obt", root=self.root)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("RMSE clean", r.stdout)
        report_rmse("Synthetic test data (fake; only shows the pipeline works)", r.stdout)
        best = json.loads((self.root / "artifacts" / "best_iter.json").read_text())
        self.assertEqual(set(best), {"reg", "stale", "mix", "lr"})

        r = run_cli("submit", "--name", "team", "--fast", "--use-actual-obt", root=self.root)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = pd.read_parquet(self.root / "submissions" / "team_v1.parquet")
        tpl = pd.read_parquet(self.root / "data" / "submitting.parquet")
        self.assertEqual(list(out.columns), list(tpl.columns))
        np.testing.assert_array_equal(out[prc.ID].to_numpy(), tpl[prc.ID].to_numpy())
        self.assertFalse(out[prc.TARGET].isna().any())
        self.assertTrue((out[prc.TARGET] >= 0).all())

    def test_tail_pricing_is_default_and_no_tail_caps(self):
        r = run_cli("validate", "--fast", "--use-actual-obt", root=self.root)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Late unmatched", r.stdout)
        self.assertIn("Squared error by row group", r.stdout)
        r = run_cli("validate", "--fast", "--use-actual-obt", "--no-tail", root=self.root)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("Late unmatched", r.stdout)
        self.assertIn("RMSE all rows", r.stdout)

    def test_diagnose_what_if_reproduces_validate(self):
        r = run_cli("validate", "--fast", "--use-actual-obt", root=self.root)
        self.assertEqual(r.returncode, 0, r.stderr)
        r = run_cli("diagnose", root=self.root)
        self.assertEqual(r.returncode, 0, r.stderr)
        head = re.search(r"RMSE all ([\d.]+)", r.stdout)
        first = re.search(r"^\s+none\s+1\.0\s+([\d.]+)", r.stdout, re.M)
        assert head and first, r.stdout
        self.assertEqual(first.group(1), head.group(1))

    def test_load_labels_stale_blocks_on_training_rows_only(self):
        dep, _ = prc.load(self.root / "data")
        train, rank = dep[dep["split"].eq("train")], dep[dep["split"].eq("rank")]
        self.assertTrue(rank["stale"].isna().all())  # ranking block times are blank
        self.assertTrue(train["stale"].isin([0.0, 1.0]).all())
        f = prc.training_files(self.root / "data")[0]
        raw = pd.read_parquet(f)
        raw = raw[raw["PHASE_mvt"].eq("DEP")]
        gap = (raw["BLOCK_TIME_UTC_mvt"] - raw["SCHED_TIME_UTC_mvt"]).dt.total_seconds().abs()
        loaded = train[train[prc.ID].isin(raw[prc.ID])]
        self.assertAlmostEqual(loaded["stale"].mean(), (gap <= prc.STALE_SEC).mean())
        self.assertTrue(dep["unmatched"].eq(0).all())  # synthetic flights all have FLIGHT_ID

    def test_submit_aborts_on_unknown_template_id(self):
        root = self.root / "bad"
        (root / "data").mkdir(parents=True)
        for f in (self.root / "data").glob("*.parquet"):
            (root / "data" / f.name).write_bytes(f.read_bytes())
        tpl = pd.read_parquet(root / "data" / "submitting.parquet")
        extra = pd.DataFrame({prc.ID: [tpl[prc.ID].max() + 10 ** 6], prc.TARGET: [np.nan]})
        pd.concat([tpl, extra], ignore_index=True).to_parquet(root / "data" / "submitting.parquet")
        r = run_cli("submit", "--name", "team", "--fast", "--rounds", "5", root=root)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no prediction", r.stderr)
        self.assertFalse((root / "submissions" / "team_v1.parquet").exists())

    def test_upload_rejects_bad_name_and_bad_file_before_network(self):
        tpl = self.root / "data" / "submitting.parquet"
        r = run_cli("upload", str(tpl), "--as", "wrong.parquet", root=self.root)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("does not follow", r.stderr)

        r = run_cli("upload", str(tpl), "--as", "team_v1.parquet", root=self.root)
        self.assertNotEqual(r.returncode, 0)  # template target column is all null
        self.assertIn("Not uploading", r.stderr)


DATA = ROOT / "data"
class TestScores(unittest.TestCase):
    def test_record_score_keeps_one_line_per_version_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            prc.record_score(tmp, 9)
            prc.record_score(tmp, 8, "319.7223")
            prc.record_score(tmp, 10)
            prc.record_score(tmp, 9, "300.5")
            prc.record_score(tmp, 8)  # pending does not erase a recorded score
            text = (Path(tmp) / "scores.txt").read_text()
        self.assertEqual(text.splitlines(), ["v8 319.7223", "v9 300.5", "v10 pending"])

    def test_score_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(run_cli("score", "8", "319.7223", root=tmp).returncode, 0)
            r = run_cli("score", "9", root=tmp)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("v9 pending", r.stdout)
            bad = run_cli("score", "9", "abc", root=tmp)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("not a number", bad.stderr)
            text = (Path(tmp) / "scores.txt").read_text()
        self.assertEqual(text.splitlines(), ["v8 319.7223", "v9 pending"])


class TestDiagnose(unittest.TestCase):
    def test_groups_rows_and_splits_clean_rows_by_stale_reach(self):
        v = pd.DataFrame({
            prc.ID: list("abcdef"), "airport": "EDDF",
            "y": [600.0, 900.0, 9000.0, 100000.0, 5000.0, -30.0],
            "pred": [620.0, 1500.0, 4000.0, 2000.0, 700.0, 100.0],
            "normal": [620.0, 950.0, 600.0, 600.0, 700.0, 100.0],
            "p_stale": 0.1, "mvt_minus_sched": [700.0, 7000.0, 9000.0, 50000.0, 100.0, 100.0],
            "stale": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], "unmatched": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]})
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "artifacts").mkdir()
            v.to_parquet(Path(tmp) / "artifacts" / "val_predictions.parquet", index=False)
            r = run_cli("diagnose", root=tmp)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("6 validation rows", r.stdout)
        for text in ["late unmatched", "stale", "other", "y <= 0", "3600-7200s", "> 86400s"]:
            self.assertIn(text, r.stdout)
        self.assertIn("takeoff - scheduled > 3600s: 1 rows, RMSE model 600.0, normal-taxi model only 50.0",
                      r.stdout)
        self.assertIn("Largest squared errors", r.stdout)

    def test_requires_validate_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = run_cli("diagnose", root=tmp)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Run validate first", r.stderr)


HAS_DATA = (DATA / "ranking.parquet").exists() and (DATA / "submitting.parquet").exists()


@unittest.skipUnless(HAS_DATA, "competition data not downloaded (python prc.py download)")
class TestCompetitionData(unittest.TestCase):
    """Checks on the real files in data/. Outputs go to a temp folder, never artifacts/."""

    @classmethod
    def setUpClass(cls):
        cls.template = pd.read_parquet(DATA / "submitting.parquet")
        rank = pd.read_parquet(DATA / "ranking.parquet",
                               columns=[prc.ID, "PHASE_mvt", prc.TARGET, "BLOCK_TIME_UTC_mvt"])
        cls.rank_deps = rank[rank["PHASE_mvt"].eq("DEP")]

    def test_all_files_present(self):
        self.assertEqual(len(prc.training_files(DATA)), 12)

    def test_template_ids_are_exactly_the_ranking_departures(self):
        tid = self.template[prc.ID]
        self.assertFalse(tid.duplicated().any())
        self.assertEqual(set(tid), set(self.rank_deps[prc.ID]))
        self.assertEqual(len(tid), len(self.rank_deps))

    def test_ranking_departure_targets_are_blanked(self):
        self.assertTrue(self.rank_deps[prc.TARGET].isna().all())
        self.assertTrue(self.rank_deps["BLOCK_TIME_UTC_mvt"].isna().all())

    def test_training_target_is_takeoff_minus_offblock(self):
        f = prc.training_files(DATA)[0]
        d = pd.read_parquet(f, columns=["PHASE_mvt", prc.TARGET, "MVT_TIME_UTC_mvt",
                                        "BLOCK_TIME_UTC_mvt"])
        d = d[d["PHASE_mvt"].eq("DEP")]
        implied = prc.to_secs(prc.to_utc(d["MVT_TIME_UTC_mvt"])) - prc.to_secs(
            prc.to_utc(d["BLOCK_TIME_UTC_mvt"]))
        self.assertGreater(((implied - d[prc.TARGET]).abs() <= 1).mean(), 0.999)

    def test_existing_submission_files_are_valid(self):
        files = sorted((ROOT / "submissions").glob("*_v*.parquet"))
        if not files:
            self.skipTest("no submission files yet")
        for f in files:
            with self.subTest(file=f.name):
                out = pd.read_parquet(f)
                self.assertEqual(list(out.columns), list(self.template.columns))
                np.testing.assert_array_equal(out[prc.ID].to_numpy(), self.template[prc.ID].to_numpy())
                self.assertFalse(out[prc.TARGET].isna().any())
                self.assertTrue((out[prc.TARGET] >= 0).all())

    def test_validate_on_competition_data(self):
        """Fast validation (30% of rows) on Jan + Jul 2025; takes a few minutes."""
        with tempfile.TemporaryDirectory() as tmp:
            r = run_cli("--data", str(DATA), "validate", "--fast", "--use-actual-obt", root=tmp)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        report_rmse("Competition data (fast run, 30% of training rows)", r.stdout)

        m = re.search(r"RMSE clean .*model ([\d.]+) .*baseline ([\d.]+)", r.stdout)
        assert m, r.stdout
        model, baseline = float(m.group(1)), float(m.group(2))
        FINAL_RMSE.append(model)
        self.assertLess(model, baseline)
        # The default prices the tail (leaderboard v8, capped at max_target, scored 319.7 on all rows;
        # on this validation the capped run scored 566.4 all rows / 224.8 clean and the earlier
        # uncapped run 383.1 / 263.5). Loose bounds until the first tail-mode run sets real ones.
        self.assertLess(model, 280)

        m = re.search(r"RMSE all rows .*model ([\d.]+) .*baseline ([\d.]+)", r.stdout)
        assert m, r.stdout
        self.assertLess(float(m.group(1)), float(m.group(2)))
        self.assertLess(float(m.group(1)), 480)  # the capped run's 566.4 must be beaten by a wide margin


if __name__ == "__main__":
    unittest.main()
