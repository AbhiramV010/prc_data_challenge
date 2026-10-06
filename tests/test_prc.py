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
        "hour": [0.0] * n, "dow": [0] * n, "unmatched": [0.0, 1.0, 0.0],
        "t": [600.0, 700.0, 1000.0], "obt": [0.0, 100.0, 650.0],
        "SCHED_TIME_UTC_mvt": [0.0] * n, "EOBT_1_flt": [0.0] * n, "IOBT_flt": [0.0] * n,
        "AOBT_3_flt": [0.0] * n, "LOBT_flt": [0.0] * n,
    })
    events = pd.DataFrame({
        "airport": s(["A"] * 4), "runway": s(["R"] * 4), "dep": [True, True, True, False],
        "t": [600.0, 700.0, 1000.0, 300.0], "obt": [0.0, 100.0, 650.0, np.nan],
        "inblk": [np.nan, np.nan, np.nan, 500.0],
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

    def test_first_departure_has_no_previous_gap(self):
        self.assertTrue(np.isnan(self.X["rwy_gap_prev_dep"].iloc[0]))

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
            def __init__(self, values):
                self.values = np.array(values, dtype=float)

            def predict(self, X):
                return self.values

        X = pd.DataFrame({"mvt_minus_sched": [1000.0, 5000.0, np.nan, -100.0]})
        models = {"stale": Fixed([0.0, 1.0, 0.5, 1.0]), "reg": Fixed([800.0, 800.0, 700.0, -50.0])}
        pred, _, normal = prc.predict_mix(models, X)
        # Stale rows take takeoff - scheduled; no schedule falls back to the normal model.
        np.testing.assert_array_equal(pred, [800.0, 5000.0, 700.0, 0.0])
        np.testing.assert_array_equal(normal, [800.0, 800.0, 700.0, 0.0])

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
RMSE_REPORT: list[str] = []
RMSE_RESULTS: list[dict] = []


def report_rmse(label, stdout):
    """Keep validate's RMSE lines for the summary and results.csv written after all tests."""
    RMSE_REPORT.append(label)
    RMSE_REPORT.extend("  " + line for line in stdout.splitlines() if line.startswith("RMSE"))
    clean = re.search(r"RMSE clean \(.*?([\d,]+) rows\): model ([\d.]+)\s+\|\s+airport-mean "
                      r"baseline ([\d.]+)", stdout)
    every = re.search(r"RMSE all rows: model ([\d.]+)\s+\|\s+baseline ([\d.]+)\s+\|\s+best_iter (\d+)",
                      stdout)
    if clean and every:
        RMSE_RESULTS.append({
            "dataset": label, "rows_clean": int(clean.group(1).replace(",", "")),
            "model_rmse_clean": float(clean.group(2)), "baseline_rmse_clean": float(clean.group(3)),
            "model_rmse_all": float(every.group(1)), "baseline_rmse_all": float(every.group(2)),
            "best_iter": int(every.group(3)),
        })


def tearDownModule():
    if RMSE_REPORT:
        print("\n\nRMSE summary\n" + "\n".join(RMSE_REPORT), file=sys.stderr)
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
        self.assertEqual(set(best), {"reg", "stale", "mix"})

        r = run_cli("submit", "--name", "team", "--fast", "--use-actual-obt", root=self.root)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = pd.read_parquet(self.root / "submissions" / "team_v1.parquet")
        tpl = pd.read_parquet(self.root / "data" / "submitting.parquet")
        self.assertEqual(list(out.columns), list(tpl.columns))
        np.testing.assert_array_equal(out[prc.ID].to_numpy(), tpl[prc.ID].to_numpy())
        self.assertFalse(out[prc.TARGET].isna().any())
        self.assertTrue((out[prc.TARGET] >= 0).all())

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
        self.assertLess(model, baseline)
        # The competition scores all rows; the blend booster trades a little clean-row RMSE for
        # much lower all-row RMSE (fast run: 383.1 vs 415.3 for the classifier-only blend).
        m = re.search(r"RMSE all rows .*model ([\d.]+) .*classifier blend ([\d.]+)", r.stdout)
        assert m, r.stdout
        self.assertLess(float(m.group(1)), float(m.group(2)))
        self.assertLess(float(m.group(1)), 400)  # guards against regressions


if __name__ == "__main__":
    unittest.main()
