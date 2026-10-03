"""PRC Data Challenge 2026: taxi-out time prediction.

Subcommands: download, audit, validate, submit, synth (synthetic data for testing).
Requires: pip install minio pandas pyarrow lightgbm numpy
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ENDPOINT = "s3.opensky-network.org"
DEFAULT_BUCKET = "prc-2026-datasets"
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LTFM", "LSZH"]
FILE_RE = re.compile(r"^(training_.*|ranking.*|submitting.*)\.parquet$")
TARGET = "TAXITIME_SEC_mvt"
ID = "MVT_ID_mvt"

TIME_COLS = ["MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "EOBT_1_flt", "IOBT_flt"]
OBT_COLS = ["AOBT_3_flt", "LOBT_flt"]
STR_COLS = [
    "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "ADEP_flt", "ADES_flt", "RUNWAY_mvt", "STAND_mvt",
    "AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt", "FLIGHT_RULE_mvt", "FLIGHT_RULE_flt",
    "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt", "AIRCRAFT_OPERATOR_flt", "FLIGHT_TYPE_flt",
]
CAT_FEATS = ["airport", "apt_rwy", "apt_stand", "actype", "dest", "rule", "wake",
             "market", "operator", "ftype"]
MIN_CAT_COUNT = 30

# (name, lo offset s, lo side, hi offset s, hi side); searchsorted sides set open/closed bounds
DEP_WINDOWS = [("dep_prev10", -600, "left", 0, "left"),
               ("dep_prev30", -1800, "left", 0, "left"),
               ("dep_next15", 0, "right", 900, "right"),
               ("dep_pm30", -1800, "left", 1800, "right")]
ARR_WINDOWS = [("arr_pm15", -900, "left", 900, "right"),
               ("arr_pm30", -1800, "left", 1800, "right")]


def paths(args):
    root = Path(args.root)
    data = Path(args.data) if getattr(args, "data", None) else root / "data"
    return data, root / "artifacts", root / "submissions"


def training_files(data):
    files = sorted(data.glob("training_*.parquet"))
    if not files:
        sys.exit(f"No training_*.parquet in {data}. Run `python prc.py download` first.")
    return files


def single_file(data, prefix):
    files = sorted(data.glob(f"{prefix}*.parquet"))
    if not files:
        sys.exit(f"No {prefix}*.parquet in {data}.")
    return files[0]


def read_dotenv(path):
    env = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            key, sep, value = line.strip().removeprefix("export ").partition("=")
            if sep and not key.startswith("#"):
                env[key.strip()] = value.strip().strip("'\"")
    return env


def minio_keys():
    dotenv = read_dotenv(Path(__file__).with_name(".env"))
    access = os.environ.get("PRC_ACCESS_KEY") or dotenv.get("PRC_ACCESS_KEY")
    secret = os.environ.get("PRC_SECRET_KEY") or dotenv.get("PRC_SECRET_KEY")
    if access and secret:
        return access, secret
    cred = Path(__file__).with_name("credentials.json")
    if cred.exists():
        c = json.loads(cred.read_text())
        access = c.get("PRC_ACCESS_KEY") or c.get("accessKey") or c.get("access_key")
        secret = c.get("PRC_SECRET_KEY") or c.get("secretKey") or c.get("secret_key")
        if access and secret:
            return access, secret
        if "clientId" in c:
            sys.exit("credentials.json holds the OSN REST clientId/clientSecret, which do not open "
                     "the MinIO buckets. Set PRC_ACCESS_KEY and PRC_SECRET_KEY (MinIO console keys).")
    sys.exit("Set PRC_ACCESS_KEY and PRC_SECRET_KEY in .env or the environment.")


def cmd_download(args):
    from minio import Minio
    from minio.error import S3Error

    data, _, _ = paths(args)
    data.mkdir(parents=True, exist_ok=True)
    access, secret = minio_keys()
    client = Minio(ENDPOINT, access_key=access, secret_key=secret, secure=True)

    def challenge_objects(bucket):
        return [o for o in client.list_objects(bucket, recursive=True)
                if o.object_name and FILE_RE.match(o.object_name.rsplit("/", 1)[-1])]

    bucket = args.bucket
    if not bucket:
        try:
            names = [b.name for b in client.list_buckets()]
            print("Visible buckets:", ", ".join(names) or "(none)")
        except S3Error as e:
            print(f"Cannot list buckets ({e.code}); trying {DEFAULT_BUCKET}")
            names = []
        for name in names:
            try:
                if any((o.object_name or "").endswith("ranking.parquet") for o in challenge_objects(name)):
                    bucket = name
                    break
            except S3Error:
                continue
        bucket = bucket or DEFAULT_BUCKET
    print("Using bucket:", bucket)

    objs = challenge_objects(bucket)
    if not objs:
        sys.exit(f"No challenge files found in {bucket}. Pass --bucket NAME.")
    for o in objs:
        name, size = o.object_name or "", o.size or 0
        dest = data / name.rsplit("/", 1)[-1]
        if dest.exists() and dest.stat().st_size == size:
            print(f"skip  {dest.name}")
            continue
        print(f"get   {dest.name} ({size / 1e6:.1f} MB)")
        client.fget_object(bucket, name, str(dest))
    print("Done.")


def cmd_upload(args):
    from minio import Minio
    from minio.error import S3Error

    path = Path(args.file)
    key = args.as_name or path.name
    m = re.fullmatch(r"(.+)_v\d+\.parquet", key)
    if not m:
        sys.exit(f"{key} does not follow <team-name>_v<N>.parquet. Use --as NAME.")
    data, _, _ = paths(args)
    sub, tpl = pd.read_parquet(path), pd.read_parquet(single_file(data, "submitting"))
    if (list(sub.columns) != list(tpl.columns) or len(sub) != len(tpl)
            or not (sub[ID].to_numpy() == tpl[ID].to_numpy()).all() or sub[TARGET].isna().any()):
        sys.exit(f"{path} does not match the template rows, IDs or columns, or has nulls. Not uploading.")

    access, secret = minio_keys()
    client = Minio(ENDPOINT, access_key=access, secret_key=secret, secure=True)
    bucket = args.bucket
    if not bucket:
        names = [b.name for b in client.list_buckets()]
        matches = [n for n in names if m.group(1) in n]
        if len(matches) != 1:
            sys.exit(f"Visible buckets: {', '.join(names) or '(none)'}. Pass --bucket NAME.")
        bucket = matches[0]
    try:
        client.stat_object(bucket, key)
        sys.exit(f"{bucket}/{key} already exists. Use the next version number.")
    except S3Error as e:
        if e.code not in ("NoSuchKey", "NoSuchObject"):
            raise
    client.fput_object(bucket, key, str(path))
    print(f"Uploaded {path} -> {bucket}/{key}")


def to_utc(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.tz_convert("UTC") if s.dt.tz is not None else s.dt.tz_localize("UTC")
    return pd.to_datetime(s, utc=True, errors="coerce")


def to_secs(ts):
    return (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()


def read_cols(path, cols):
    have = set(pq.read_schema(path).names)
    df = pq.read_table(path, columns=[c for c in cols if c in have]).to_pandas()
    for c in cols:
        if c not in df:
            df[c] = None
    return df


def reporting_airport(df, is_dep):
    apt = df["ADEP_mvt"].where(is_dep, df["ADES_mvt"]).astype("string")
    flt = df["ADEP_flt"].where(is_dep, df["ADES_flt"]).astype("string")
    return apt.fillna(flt)


def load(data: Path, use_obt=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Departures (train + ranking) with raw fields, and all movements as traffic events."""
    time_cols = TIME_COLS + OBT_COLS
    cols = [ID, TARGET, "BLOCK_TIME_UTC_mvt"] + STR_COLS + time_cols
    files = [("train", f) for f in training_files(data)] + [("rank", single_file(data, "ranking"))]
    deps, events = [], []
    for split, path in files:
        df = read_cols(path, cols)
        is_dep = df["PHASE_mvt"].astype("string").str.upper().eq("DEP").fillna(False)
        df["airport"] = reporting_airport(df, is_dep)
        mvt = to_utc(df["MVT_TIME_UTC_mvt"])
        df["t"] = to_secs(mvt)
        eobt = to_secs(to_utc(df["EOBT_1_flt"]))
        obt = to_secs(to_utc(df["AOBT_3_flt"])).fillna(eobt) if use_obt else eobt
        # Departure block times are blanked in ranking, so only arrival in-block times are used.
        inblk = to_secs(to_utc(df["BLOCK_TIME_UTC_mvt"])).where(~is_dep)
        events.append(pd.DataFrame({"airport": df["airport"],
                                    "runway": df["RUNWAY_mvt"].astype("string"),
                                    "dep": is_dep, "t": df["t"], "obt": obt.where(is_dep),
                                    "inblk": inblk}))
        d = df[is_dep].copy()
        m = mvt[is_dep]
        d["year"], d["month"] = m.dt.year, m.dt.month
        d["hour"], d["dow"] = m.dt.hour + m.dt.minute / 60, m.dt.dayofweek
        for c in time_cols[1:]:
            d[c] = to_secs(to_utc(d[c]))
        for c in STR_COLS:
            d[c] = d[c].astype("string")
        d[TARGET] = pd.to_numeric(d[TARGET], errors="coerce")
        d["split"] = split
        d["obt"] = obt[is_dep]
        deps.append(d.drop(columns=["MVT_TIME_UTC_mvt", "PHASE_mvt", "BLOCK_TIME_UTC_mvt"]))
        print(f"loaded {path.name}: {len(df):,} rows, {int(is_dep.sum()):,} departures")
        del df
    return pd.concat(deps, ignore_index=True), pd.concat(events, ignore_index=True)


SPAN = 10 ** 10  # separates keys in the combined (key, time) sort order


def encode(ev_key: pd.Series, ev_t: pd.Series, q_key: pd.Series, q_t: pd.Series):
    """Sorted (key, time) event codes, query codes, and which queries are valid."""
    ev_key, ev_t = ev_key.reset_index(drop=True), ev_t.reset_index(drop=True)
    q_key, q_t = q_key.reset_index(drop=True), q_t.reset_index(drop=True)
    codes, _ = pd.factorize(pd.concat([ev_key, q_key], ignore_index=True))
    n = len(ev_key)
    ec, qc = codes[:n].astype(np.int64), codes[n:].astype(np.int64)
    et, qt = ev_t.to_numpy(float), q_t.to_numpy(float)
    ok_e = (ec >= 0) & ~np.isnan(et)
    ev = np.sort(ec[ok_e] * SPAN + et[ok_e].astype(np.int64))
    ok_q = (qc >= 0) & ~np.isnan(qt)
    qv = qc[ok_q] * SPAN + qt[ok_q].astype(np.int64)
    return ev, qv, ok_q


def count_le(ev_key, ev_t, q_key, q_t, side: Literal["left", "right"] = "right") -> np.ndarray:
    """Events with the same key at or before (side="right") or before ("left") each query time."""
    ev, qv, ok_q = encode(ev_key, ev_t, q_key, q_t)
    out = np.full(len(q_key), np.nan)
    out[ok_q] = np.searchsorted(ev, qv, side) - np.searchsorted(ev, (qv // SPAN) * SPAN, "left")
    return out


def neighbor_gaps(ev_key, ev_t, q_key, q_t) -> tuple[np.ndarray, np.ndarray]:
    """Seconds since the previous and until the next event with the same key (self excluded)."""
    ev, qv, ok_q = encode(ev_key, ev_t, q_key, q_t)
    prev, nxt = np.full(len(q_key), np.nan), np.full(len(q_key), np.nan)
    i = np.searchsorted(ev, qv, "left") - 1
    j = np.searchsorted(ev, qv, "right")
    gp = np.where(i >= 0, qv - ev[np.clip(i, 0, None)], np.inf)
    gn = np.where(j < len(ev), ev[np.clip(j, None, len(ev) - 1)] - qv, np.inf)
    # A gap of a day or more means the neighbour is missing or belongs to another key.
    prev[ok_q] = np.where(gp < 86400, gp, np.nan)
    nxt[ok_q] = np.where(gn < 86400, gn, np.nan)
    return prev, nxt


def window_counts(ev_key: pd.Series, ev_t: pd.Series, q_key: pd.Series, q_t: pd.Series,
                  windows) -> dict[str, np.ndarray]:
    ev, qv, ok_q = encode(ev_key, ev_t, q_key, q_t)
    out = {}
    for name, lo, lo_side, hi, hi_side in windows:
        col = np.full(len(q_key), np.nan, dtype=np.float32)
        col[ok_q] = np.searchsorted(ev, qv + hi, hi_side) - np.searchsorted(ev, qv + lo, lo_side)
        out[name] = col
    return out


def build_features(dep, events, use_obt=False):
    X = pd.DataFrame(index=dep.index)
    X["airport"] = dep["airport"]
    X["apt_rwy"] = dep["airport"] + "|" + dep["RUNWAY_mvt"]
    X["apt_stand"] = dep["airport"] + "|" + dep["STAND_mvt"]
    X["actype"] = dep["AIRCRAFT_TYPE_mvt"].fillna(dep["AIRCRAFT_TYPE_flt"])
    X["dest"] = dep["ADES_mvt"].fillna(dep["ADES_flt"])
    X["rule"] = dep["FLIGHT_RULE_mvt"].fillna(dep["FLIGHT_RULE_flt"])
    X["wake"] = dep["WK_TBL_CAT_flt"]
    X["market"] = dep["MARKET_SEGMENT_flt"]
    X["operator"] = dep["AIRCRAFT_OPERATOR_flt"]
    X["ftype"] = dep["FLIGHT_TYPE_flt"]

    X["hour"], X["dow"] = dep["hour"], dep["dow"]
    X["mvt_minus_sched"] = dep["t"] - dep["SCHED_TIME_UTC_mvt"]
    X["mvt_minus_eobt1"] = dep["t"] - dep["EOBT_1_flt"]
    X["mvt_minus_iobt"] = dep["t"] - dep["IOBT_flt"]
    if use_obt:
        X["mvt_minus_aobt3"] = dep["t"] - dep["AOBT_3_flt"]
        X["mvt_minus_lobt"] = dep["t"] - dep["LOBT_flt"]

    ev_rwy = events["airport"] + "|" + events["runway"]
    is_d = events["dep"].to_numpy()
    for level, ek, qk in [("apt", events["airport"], X["airport"]), ("rwy", ev_rwy, X["apt_rwy"])]:
        for mask, windows in [(is_d, DEP_WINDOWS), (~is_d, ARR_WINDOWS)]:
            counts = window_counts(ek[mask].reset_index(drop=True),
                                   events["t"][mask].reset_index(drop=True),
                                   qk.reset_index(drop=True), dep["t"].reset_index(drop=True),
                                   windows)
            for name, col in counts.items():
                X[f"{level}_{name}"] = col

    # Queue and taxiway load between pushback (obt) and takeoff.
    t, obt = dep["t"], dep["obt"]
    X["taxi_window"] = t - obt
    dt, at = events["t"][is_d], events["t"][~is_d]
    for level, ek, qk in [("apt", events["airport"], X["airport"]), ("rwy", ev_rwy, X["apt_rwy"])]:
        dk, ak = ek[is_d], ek[~is_d]
        X[f"{level}_deps_in_taxi"] = count_le(dk, dt, qk, t, "left") - count_le(dk, dt, qk, obt, "left")
        X[f"{level}_arrs_in_taxi"] = count_le(ak, at, qk, t, "left") - count_le(ak, at, qk, obt, "left")
        X[f"{level}_gap_prev_dep"], X[f"{level}_gap_next_dep"] = neighbor_gaps(dk, dt, qk, t)
    apt, dk, ak = X["airport"], events["airport"][is_d], events["airport"][~is_d]
    d_obt, a_in = events["obt"][is_d], events["inblk"][~is_d]
    for name, q in [("obt", obt), ("t", t)]:
        X[f"deps_taxiing_at_{name}"] = count_le(dk, d_obt, apt, q) - count_le(dk, dt, apt, q)
        X[f"arrs_taxiing_at_{name}"] = count_le(ak, at, apt, q) - count_le(ak, a_in, apt, q)
    X["pushbacks_prev15"] = count_le(dk, d_obt, apt, obt) - count_le(dk, d_obt, apt, obt - 900)
    return X


def encode_cats(X, fit_mask):
    """Categories come from fitting rows only; rare and unseen values become NaN."""
    for c in CAT_FEATS:
        vc = X.loc[fit_mask, c].value_counts()
        keep = sorted(vc.index[vc >= MIN_CAT_COUNT])
        X[c] = pd.Categorical(X[c].where(X[c].isin(keep)), categories=keep)
    return X


def prepare(args):
    data, _, _ = paths(args)
    dep, events = load(data, args.use_actual_obt)
    X = build_features(dep, events, args.use_actual_obt)
    del events
    return dep, X


def lgb_params(fast):
    return {"objective": "regression", "metric": "rmse", "learning_rate": 0.1 if fast else 0.05,
            "num_leaves": 127 if fast else 255, "min_data_in_leaf": 100, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "cat_smooth": 20,
            "cat_l2": 10, "max_cat_threshold": 64, "verbose": -1, "seed": 0}


def fit(X, y, rounds, fast, Xv=None, yv=None):
    import lightgbm as lgb
    y = y.clip(upper=np.percentile(y, 99.9))
    dtr = lgb.Dataset(X, y, categorical_feature=CAT_FEATS, free_raw_data=True)
    if Xv is None:
        return lgb.train(lgb_params(fast), dtr, rounds,
                         callbacks=[lgb.log_evaluation(0)])
    dv = lgb.Dataset(Xv, yv, reference=dtr)
    return lgb.train(lgb_params(fast), dtr, rounds, valid_sets=[dv], valid_names=["valid"],
                     callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])


def fit_rows(dep, mask, fast, max_target):
    mask = mask & dep[TARGET].notna() & (dep[TARGET] > 0) & (dep[TARGET] <= max_target)
    if fast:
        rng = np.random.default_rng(0)
        mask &= rng.random(len(dep)) < 0.3
    return mask.to_numpy()


def predict(model, X, **kw):
    return np.asarray(model.predict(X, **kw), dtype=float).clip(min=0)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def cmd_validate(args):
    dep, X = prepare(args)
    _, art, _ = paths(args)
    is_train = dep["split"].eq("train")
    hold = dep["month"].isin([1, 7])
    fm = fit_rows(dep, is_train & ~hold, args.fast, args.max_target)
    vm = (is_train & hold & dep[TARGET].notna()).to_numpy()
    clean = (dep[TARGET] > 0) & (dep[TARGET] <= args.max_target)
    cm = vm & clean.to_numpy()
    if not vm.any():
        sys.exit("No Jan/Jul 2025 departures with a target to validate on.")
    X = encode_cats(X, fm)
    y = pd.Series(dep[TARGET], dtype=float)
    print(f"fit rows {fm.sum():,}, validation rows {vm.sum():,}, features {X.shape[1]}")

    model = fit(X.loc[fm], y.loc[fm], 1500 if args.fast else 5000, args.fast, X.loc[cm], y.loc[cm])
    pred = predict(model, X.loc[vm], num_iteration=model.best_iteration)
    yv = y.loc[vm].to_numpy(float)
    c = clean.to_numpy()[vm]

    apt_mean = y.loc[fm].groupby(dep.loc[fm, "airport"]).mean()
    base = dep.loc[vm, "airport"].map(apt_mean).fillna(y.loc[fm].mean()).to_numpy(float)
    print(f"\nRMSE clean (0 < y <= {args.max_target:.0f}s, {c.sum():,} rows): "
          f"model {rmse(pred[c], yv[c]):.2f}  |  airport-mean baseline {rmse(base[c], yv[c]):.2f}")
    print(f"RMSE all rows: model {rmse(pred, yv):.2f}  |  baseline {rmse(base, yv):.2f}"
          f"  |  best_iter {model.best_iteration}")

    per = pd.DataFrame({"airport": dep.loc[vm, "airport"].to_numpy(), "y": yv, "p": pred, "b": base})[c]
    per["rmse"], per["baseline"] = (per.p - per.y) ** 2, (per.b - per.y) ** 2
    tab = per.groupby("airport").agg(n=("y", "size"), rmse=("rmse", "mean"),
                                     baseline=("baseline", "mean"))
    tab[["rmse", "baseline"]] = np.sqrt(tab[["rmse", "baseline"]])
    print("\nPer airport (clean rows):\n" + tab.round(1).to_string())

    imp = pd.Series(model.feature_importance("gain"), index=model.feature_name())
    print("\nTop features (gain share):")
    print((imp / imp.sum()).sort_values(ascending=False).head(25).round(4).to_string())

    art.mkdir(parents=True, exist_ok=True)
    (art / "best_iter.txt").write_text(str(model.best_iteration))
    print(f"\nSaved {art / 'best_iter.txt'}")


def cmd_submit(args):
    data, art, subs = paths(args)
    template = pd.read_parquet(single_file(data, "submitting"))
    dep, X = prepare(args)
    fm = fit_rows(dep, dep["split"].eq("train"), args.fast, args.max_target)
    rm = dep["split"].eq("rank").to_numpy()
    X = encode_cats(X, fm)

    rounds = args.rounds
    if not rounds:
        best = art / "best_iter.txt"
        rounds = int(int(best.read_text()) * 1.1) if best.exists() else (600 if args.fast else 2000)
    print(f"fit rows {fm.sum():,}, ranking departures {rm.sum():,}, rounds {rounds}")
    model = fit(X.loc[fm], dep.loc[fm, TARGET], rounds, args.fast)
    pred = pd.Series(predict(model, X.loc[rm]), index=dep.loc[rm, ID].astype(str).to_numpy())

    tid = template[ID].astype(str)
    if tid.duplicated().any():
        sys.exit("Template has duplicate IDs. Aborting.")
    if pred.index.duplicated().any():
        sys.exit(f"{pred.index.duplicated().sum()} duplicate ranking departure IDs. Aborting.")
    n_missing = len(set(tid) - set(pred.index))
    if n_missing:
        sys.exit(f"{n_missing} template IDs have no prediction. Aborting.")

    out = template[[ID]].copy()
    values = pred.reindex(tid).to_numpy()
    if np.isnan(values).any():
        sys.exit("Null predictions. Aborting.")
    dtype = template[TARGET].dtype
    target = pd.Series(values, index=out.index)
    out[TARGET] = target.round().astype(dtype) if pd.api.types.is_integer_dtype(dtype) else target

    subs.mkdir(parents=True, exist_ok=True)
    path = subs / f"{args.name}_v{args.version}.parquet"
    out.to_parquet(path, index=False)
    print(f"Wrote {path}: {len(out):,} rows, mean {values.mean():.1f}s, "
          f"min {values.min():.1f}s, max {values.max():.1f}s")


def cmd_audit(args):
    data, _, _ = paths(args)
    files = [("train", f) for f in training_files(data)] + [("rank", single_file(data, "ranking"))]
    counts, nonnull, ndep = {}, {"train": {}, "rank": {}}, {"train": 0, "rank": 0}
    taxi, seen, rank_keys, apt_check = [], {}, [], []
    key_cols = {"runway": "RUNWAY_mvt", "stand": "STAND_mvt", "actype": "AIRCRAFT_TYPE_mvt",
                "operator": "AIRCRAFT_OPERATOR_flt"}

    for i, (split, path) in enumerate(files):
        df = pq.read_table(path).to_pandas()
        if i == 0 or split == "rank":
            print(f"\n{path.name} dtypes:\n" + df.dtypes.astype(str).to_string())
        phase = df["PHASE_mvt"].astype("string").str.upper()
        for p, n in phase.value_counts(dropna=False).items():
            counts[(split, p)] = counts.get((split, p), 0) + n
        is_dep = phase.eq("DEP").fillna(False)
        is_arr = phase.eq("ARR").fillna(False)
        apt_check.append((df.loc[is_dep, "ADEP_mvt"].isin(AIRPORTS).sum(), is_dep.sum(),
                          df.loc[is_arr, "ADES_mvt"].isin(AIRPORTS).sum(), is_arr.sum()))
        d = df[is_dep]
        ndep[split] += len(d)
        for c, n in d.notna().sum().items():
            nonnull[split][c] = nonnull[split].get(c, 0) + n
        if split == "train":
            implied = (to_secs(to_utc(d["MVT_TIME_UTC_mvt"])) - to_secs(to_utc(d["BLOCK_TIME_UTC_mvt"])))
            taxi.append(pd.DataFrame({"airport": d["ADEP_mvt"].astype("string"),
                                      "y": pd.to_numeric(d[TARGET], errors="coerce"),
                                      "implied": implied}))
            for k, c in key_cols.items():
                seen.setdefault(k, set()).update(
                    (d["ADEP_mvt"].astype(str) + "|" + d[c].astype(str)).unique())
        else:
            rank_keys.append({k: d["ADEP_mvt"].astype(str) + "|" + d[c].astype(str)
                              for k, c in key_cols.items()})
        del df

    print("\nRows by split and phase:")
    print(pd.Series(counts).rename_axis(["split", "phase"]).to_string())

    a = np.array(apt_check).sum(axis=0)
    print(f"\nDEP rows with ADEP_mvt in the 10 airports: {a[0] / max(a[1], 1):.1%}")
    print(f"ARR rows with ADES_mvt in the 10 airports: {a[2] / max(a[3], 1):.1%}")

    nn = pd.DataFrame({s: pd.Series(nonnull[s], dtype=float) / max(ndep[s], 1) for s in nonnull})
    print("\nNon-null share, departures (train vs ranking):")
    print(nn.round(3).to_string())

    t = pd.concat(taxi, ignore_index=True)
    ok = t["y"].notna() & t["implied"].notna()
    print(f"\nTarget equals MVT - BLOCK within 1 s: {((t.y - t.implied).abs()[ok] <= 1).mean():.1%}"
          f" of {ok.sum():,} rows")
    print(f"Target null: {t.y.isna().mean():.2%}, non-positive: {(t.y <= 0).mean():.2%}")
    stats = t.groupby("airport")["y"].describe(percentiles=[0.5, 0.95])
    print("\nTaxi-out seconds per airport (train):")
    print(stats.round(0).to_string())

    rk = rank_keys[0]
    print("\nRanking departures whose airport|value was seen in training:")
    for k in key_cols:
        print(f"  {k:9s} {rk[k].isin(seen[k]).mean():.1%}")

    for c, share in nn["rank"].reindex(OBT_COLS).fillna(0.0).items():
        if share > 0.5:
            print(f"\nWARNING: {c} is {share:.0%} filled in ranking departures. MVT - {c} may nearly "
                  f"give the target. Confirm with the organizers before using --use-actual-obt.")


def cmd_synth(args):
    data, _, _ = paths(args)
    data.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    next_id = [1]

    def month(year, m, n):
        t0 = pd.Timestamp(year=year, month=m, day=1, tz="UTC")
        mvt = t0 + pd.to_timedelta(rng.uniform(0, 28 * 86400, n), unit="s")
        apt_i = rng.integers(0, len(AIRPORTS), n)
        apt = np.array(AIRPORTS)[apt_i]
        other = rng.choice(["KJFK", "LFPO", "EGKK", "LOWW", "EKCH", "LIMC"], n)
        phase = rng.choice(["DEP", "ARR"], n)
        dep = phase == "DEP"
        stand_n = rng.integers(1, 40, n)
        rwy = rng.choice(["09L", "27R", "18"], n)
        taxi = (300 + 40 * apt_i + 6 * stand_n + 120 * (rwy == "18")
                + 100 * np.isin(mvt.hour, [6, 7, 8, 17, 18]) + rng.gamma(2, 50, n)).round()
        block = mvt - pd.to_timedelta(np.where(dep, taxi, -taxi), unit="s")
        sched = block - pd.to_timedelta(rng.normal(600, 900, n).round(), unit="s")
        ids = np.arange(next_id[0], next_id[0] + n)
        next_id[0] += n
        actype = rng.choice(["A320", "B738", "A359", "E190"], n)
        op = np.char.add("OP", rng.integers(1, 25, n).astype(str))
        return pd.DataFrame({
            "MVT_ID_mvt": ids, "FLIGHT_ID_mvt": ids + 10 ** 7, "FLIGHT_mvt": op,
            "FLIGHT_RULE_mvt": "I", "ADEP_mvt": np.where(dep, apt, other),
            "ADES_mvt": np.where(dep, other, apt), "PHASE_mvt": phase, "MVT_TIME_UTC_mvt": mvt,
            "BLOCK_TIME_UTC_mvt": block, "SCHED_TIME_UTC_mvt": sched, "AIRCRAFT_TYPE_mvt": actype,
            "RUNWAY_mvt": rwy, "STAND_mvt": np.char.add("S", stand_n.astype(str)),
            "TAXITIME_SEC_mvt": taxi, "LOBT_flt": block, "CALLSIGN_flt": op,
            "ADEP_flt": np.where(dep, apt, other), "ADES_flt": np.where(dep, other, apt),
            "ADES_FILED_flt": np.where(dep, other, apt),
            "MARKET_SEGMENT_flt": rng.choice(["Mainline", "Lowcost", "Cargo"], n),
            "IOBT_flt": sched, "FLIGHT_RULE_flt": "I", "FLIGHT_TYPE_flt": "S",
            "AIRCRAFT_TYPE_flt": actype, "WK_TBL_CAT_flt": np.where(actype == "A359", "H", "M"),
            "AIRCRAFT_OPERATOR_flt": op, "EOBT_1_flt": sched + pd.to_timedelta(300, unit="s"),
            "ARVT_1_flt": mvt, "AOBT_3_flt": block, "ARVT_3_flt": mvt,
        })

    for m in range(1, 13):
        month(2025, m, args.rows).to_parquet(data / f"training_2025-{m:02d}-01_synthetic.parquet")
    rank = pd.concat([month(2026, 1, args.rows), month(2026, 7, args.rows)], ignore_index=True)
    dep = rank["PHASE_mvt"].eq("DEP")
    for c in ["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt", "LOBT_flt"]:
        rank[c] = rank[c].where(~dep)
    rank.to_parquet(data / "ranking.parquet")
    sub = rank.loc[dep, [ID, TARGET]].sample(frac=1, random_state=0)
    sub.to_parquet(data / "submitting.parquet", index=False)
    print(f"Wrote synthetic data to {data}")


def main():
    p = argparse.ArgumentParser(description="PRC Data Challenge 2026: taxi-out time prediction.")
    p.add_argument("--root", default=".", help="folder holding data/, artifacts/, submissions/")
    p.add_argument("--data", help="data folder (default: ROOT/data)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download")
    s.add_argument("--bucket")
    s.set_defaults(func=cmd_download)

    s = sub.add_parser("upload", help="upload a submission file to the team bucket")
    s.add_argument("file")
    s.add_argument("--as", dest="as_name", help="object name, e.g. TEAM_v1.parquet")
    s.add_argument("--bucket")
    s.set_defaults(func=cmd_upload)

    sub.add_parser("audit").set_defaults(func=cmd_audit)

    s = sub.add_parser("validate")
    s.add_argument("--max-target", type=float, default=3600,
                   help="drop training rows with taxi-out above this many seconds")
    s.add_argument("--fast", action="store_true")
    s.add_argument("--use-actual-obt", action="store_true")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("submit")
    s.add_argument("--max-target", type=float, default=3600,
                   help="drop training rows with taxi-out above this many seconds")
    s.add_argument("--name", required=True)
    s.add_argument("--version", type=int, default=1)
    s.add_argument("--rounds", type=int)
    s.add_argument("--fast", action="store_true")
    s.add_argument("--use-actual-obt", action="store_true")
    s.set_defaults(func=cmd_submit)

    s = sub.add_parser("synth", help="write small synthetic data for testing")
    s.add_argument("--rows", type=int, default=4000)
    s.set_defaults(func=cmd_synth)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
