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
    "FLIGHT_mvt",
]
CAT_FEATS = ["airport", "apt_rwy", "apt_stand", "actype", "dest", "rule", "wake",
             "market", "operator", "airline", "ftype"]
MIN_CAT_COUNT = 30
# |off-block - scheduled| at or below this means the airport logged the schedule as off-block,
# so the target is simply takeoff minus scheduled time.
STALE_SEC = 60
# Unmatched flights this long after their schedule are mostly logged with the schedule as off-block
# (stale) or a day early, so the target is huge; neither shows up in the normal-taxi model.
LATE_UNMATCHED_SEC = 21600
DAY_SEC = 86400

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
    cols = [ID, TARGET, "BLOCK_TIME_UTC_mvt", "FLIGHT_ID_mvt"] + STR_COLS + time_cols
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
        blk = to_secs(to_utc(d["BLOCK_TIME_UTC_mvt"]))
        gap = (blk - d["SCHED_TIME_UTC_mvt"]).abs()
        d["stale"] = np.where(gap.notna(), (gap <= STALE_SEC).astype(float), np.nan)
        d["unmatched"] = d["FLIGHT_ID_mvt"].isna().astype(float)
        d["split"] = split
        d["obt"] = obt[is_dep]
        deps.append(d.drop(columns=["MVT_TIME_UTC_mvt", "PHASE_mvt", "BLOCK_TIME_UTC_mvt",
                                    "FLIGHT_ID_mvt"]))
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


def window_means(ev_key: pd.Series, ev_t: pd.Series, ev_val, q_key: pd.Series, q_t: pd.Series,
                 windows) -> dict[str, np.ndarray]:
    """Mean of ev_val over events with the same key and time in [q_t + lo, q_t + hi), per (name, lo, hi)."""
    ev_key, ev_t = ev_key.reset_index(drop=True), ev_t.reset_index(drop=True)
    codes, _ = pd.factorize(pd.concat([ev_key, q_key.reset_index(drop=True)], ignore_index=True))
    n = len(ev_key)
    ec, qc = codes[:n].astype(np.int64), codes[n:].astype(np.int64)
    et, val, qt = ev_t.to_numpy(float), np.asarray(ev_val, float), q_t.to_numpy(float)
    ok = (ec >= 0) & ~np.isnan(et) & ~np.isnan(val)
    code = ec[ok] * SPAN + et[ok].astype(np.int64)
    order = np.argsort(code, kind="stable")
    ev, cs = code[order], np.concatenate([[0.0], np.cumsum(val[ok][order])])
    ok_q = (qc >= 0) & ~np.isnan(qt)
    qv = qc[ok_q] * SPAN + qt[ok_q].astype(np.int64)
    out = {}
    for name, lo, hi in windows:
        a, b = np.searchsorted(ev, qv + lo, "left"), np.searchsorted(ev, qv + hi, "left")
        col = np.full(len(qt), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            col[ok_q] = np.where(b > a, (cs[b] - cs[a]) / (b - a), np.nan)
        out[name] = col
    return out


STATE_WINDOWS = [("m60", -3600, 0), ("m180", -10800, 0), ("pm60", -3600, 3600)]


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
    X["airline"] = dep["FLIGHT_mvt"].str.extract(r"^([A-Za-z]+)", expand=False).str.upper()
    X["ftype"] = dep["FLIGHT_TYPE_flt"]

    X["hour"], X["dow"] = dep["hour"], dep["dow"]
    X["unmatched"] = dep["unmatched"]
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

    # Airport state around takeoff: slow arrival taxi-in and long pushback-to-takeoff for other
    # departures mark de-icing, slot holds and storms, when long taxi-outs cluster.
    taxi_in = (events["inblk"] - events["t"]).clip(0, 7200)[~is_d]
    push_to = (events["t"] - events["obt"]).clip(0, 7200)[is_d]
    for prefix, ek, et, val in [("arr_taxiin", ak, at, taxi_in), ("dep_push_to", dk, dt, push_to)]:
        for name, col in window_means(ek, et, val, apt, t, STATE_WINDOWS).items():
            X[f"{prefix}_{name}"] = col

    # Duplicate or ghost movement records: the copy with an early block time has a huge target.
    X["actype_mvt_missing"] = dep["AIRCRAFT_TYPE_mvt"].isna().astype(float)
    cs = (dep["airport"] + "|" + dep["FLIGHT_mvt"]).where(dep["FLIGHT_mvt"].notna())
    day = np.floor(t / 86400)
    g = t.groupby([cs, day])
    X["cs_deps_same_day"] = g.transform("size").reindex(dep.index)
    X["cs_rank_same_day"] = g.rank(method="first").reindex(dep.index)
    prev, nxt = neighbor_gaps(cs, t, cs, t)
    X["cs_gap_nearest"] = np.fmin(prev, nxt)
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


def lgb_params(fast, objective="regression"):
    metric = "rmse" if objective == "regression" else "binary_logloss"
    return {"objective": objective, "metric": metric, "learning_rate": 0.1 if fast else 0.05,
            "num_leaves": 127 if fast else 255, "min_data_in_leaf": 100, "feature_fraction": 0.6,
            "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "cat_smooth": 20,
            "cat_l2": 10, "max_cat_threshold": 64, "verbose": -1, "seed": 0}


def fit(X, y, rounds, fast, Xv=None, yv=None, objective="regression"):
    import lightgbm as lgb
    dtr = lgb.Dataset(X, y, categorical_feature=CAT_FEATS, free_raw_data=True)
    params = lgb_params(fast, objective)
    if Xv is None:
        return lgb.train(params, dtr, rounds, callbacks=[lgb.log_evaluation(0)])
    dv = lgb.Dataset(Xv, yv, reference=dtr)
    return lgb.train(params, dtr, rounds, valid_sets=[dv], valid_names=["valid"],
                     callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])


def fit_masks(dep, mask, fast, max_target):
    """Rows for the normal-taxi regressor and for the stale-block classifier."""
    labelled = mask & dep["stale"].notna() & dep[TARGET].notna()
    if fast:
        labelled &= np.random.default_rng(0).random(len(dep)) < 0.3
    reg = labelled & dep["stale"].eq(0) & (dep[TARGET] > 0) & (dep[TARGET] <= max_target)
    return reg.to_numpy(), labelled.to_numpy()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def blend_parts(normal, ms, cap=None):
    """Clipped normal-taxi prediction and the stale alternative (normal when there is no schedule).
    cap bounds the stale alternative, so one huge takeoff - scheduled gap cannot drag a row far."""
    n = np.clip(normal, 0, None)
    return n, np.where(np.isnan(ms), n, np.clip(np.nan_to_num(ms), 0, cap))


def mix_rows(X, y, mask, tail, max_target):
    """Rows the blend booster fits and stops on. With tail rules, late unmatched rows are priced by
    their airport rates instead; without, targets past max_target are out of reach of a capped blend."""
    if tail:
        return mask & ~late_unmatched(X)
    yv = np.asarray(y, float)
    return mask & (yv > 0) & (yv <= max_target)


def stack_features(X, normal):
    """X plus the normal-taxi prediction and how far the schedule gap is from it. A stale row's gap is
    a normal taxi, a late flight's is not; the blend booster cannot form this difference on its own."""
    n = np.clip(np.asarray(normal, dtype=float), 0, None)
    gap = X["mvt_minus_sched"].to_numpy(float) - n
    return X.assign(n_hat=n, ms_minus_n=gap, abs_ms_minus_n=np.abs(gap))


def late_unmatched(X):
    return (X["unmatched"].to_numpy(float) == 1) & (X["mvt_minus_sched"].to_numpy(float) > LATE_UNMATCHED_SEC)


def late_unmatched_rates(X, y, stale, mask):
    """Per airport, among late unmatched rows: (share logged stale, share a day early)."""
    rows = np.asarray(mask) & late_unmatched(X)
    apt = X["airport"].astype(object).to_numpy()
    yv, st = np.asarray(y, float), np.asarray(stale, float) == 1
    rates = {}
    for a in sorted(set(apt[rows])):
        r = rows & (apt == a)
        rates[a] = (float(st[r].mean()), float((~st[r] & (yv[r] > DAY_SEC / 2)).mean()))
    return rates


def predict_mix(models, X):
    """Expected taxi-out: P(stale) * (takeoff - scheduled) + (1 - P(stale)) * normal taxi."""
    normal = np.asarray(models["reg"].predict(X), dtype=float)
    if "mix" in models:
        z = (models["stale"].predict(X, raw_score=True)
             + models["mix"].predict(stack_features(X, normal), raw_score=True))
        p = sigmoid(np.asarray(z, dtype=float))
    else:
        p = np.asarray(models["stale"].predict(X), dtype=float)
    n, m = blend_parts(normal, X["mvt_minus_sched"].to_numpy(float), models.get("cap"))
    pred = p * m + (1 - p) * n
    rates = models.get("late_rates")
    if rates:
        # Late unmatched rows use their airport's observed rates instead of the classifier.
        apt = X["airport"].astype(object).to_numpy()
        for i in np.flatnonzero(late_unmatched(X)):
            if apt[i] in rates:
                stl, day = rates[apt[i]]
                pred[i] = stl * m[i] + day * (DAY_SEC + n[i]) + (1 - stl - day) * n[i]
    return np.clip(pred, 0, None), p, n


def fit_both(X, y, stale, reg_m, stale_m, rounds, fast, reg_v=None, stale_v=None):
    ev = reg_v is not None and stale_v is not None
    reg = fit(X.loc[reg_m], y.loc[reg_m], rounds["reg"], fast,
              X.loc[reg_v] if ev else None, y.loc[reg_v] if ev else None)
    stl = fit(X.loc[stale_m], stale.loc[stale_m], rounds["stale"], fast,
              X.loc[stale_v] if ev else None, stale.loc[stale_v] if ev else None, "binary")
    return {"reg": reg, "stale": stl}


def oof_parts(X, y, stale, reg_m, stale_m, month, rounds, fast):
    """Normal-taxi predictions and stale logits for the classifier rows, each from models trained
    on the other month parity, so the mix booster learns from errors like those on unseen months."""
    n, z = np.full(len(X), np.nan), np.full(len(X), np.nan)
    odd = (month % 2 == 1).to_numpy()
    for fold in (odd, ~odd):
        m = fit_both(X, y, stale, reg_m & ~fold, stale_m & ~fold, rounds, fast)
        rows = stale_m & fold
        n[rows] = m["reg"].predict(X.loc[rows])
        z[rows] = m["stale"].predict(X.loc[rows], raw_score=True)
    return n, z


MIX_SCALE = 1e4  # keeps gradients of squared seconds in a range LightGBM handles well
MIX_MIN_HESS = 1e-3


def fit_mix(X, y, ms, n, z, rounds, fast, valid=None, cap=None):
    """Boosts the stale logit so the blend itself, not the stale label, has the lowest squared error.
    RMSE is driven by rows where takeoff - scheduled is far from a normal taxi, and only this loss
    knows that misjudging those rows costs far more than misjudging the rest.
    valid is (Xv, yv, msv, nv, zv) for early stopping."""
    import lightgbm as lgb

    class BlendDataset(lgb.Dataset):
        parts: tuple[np.ndarray, np.ndarray]

    def objective(score, ds):
        nn, mm = ds.parts
        p = sigmoid(score)
        d = (mm - nn) * p * (1 - p)
        r = nn + p * (mm - nn) - ds.get_label()
        return 2 * r * d / MIX_SCALE, np.maximum(2 * d * d, MIX_MIN_HESS) / MIX_SCALE

    def blend_rmse(score, ds):
        nn, mm = ds.parts
        p = sigmoid(score)
        return "blend_rmse", rmse(nn + p * (mm - nn), ds.get_label()), False

    dtr = BlendDataset(X, y, init_score=z, categorical_feature=CAT_FEATS, free_raw_data=True)
    dtr.parts = blend_parts(n, ms, cap)
    params = dict(lgb_params(fast), objective=objective, metric="None", learning_rate=0.02,
                  num_leaves=31, min_data_in_leaf=500, min_sum_hessian_in_leaf=0)
    if valid is None:
        return lgb.train(params, dtr, rounds)
    Xv, yv, msv, nv, zv = valid
    dv = BlendDataset(Xv, yv, init_score=zv, reference=dtr)
    dv.parts = blend_parts(nv, msv, cap)
    return lgb.train(params, dtr, rounds, valid_sets=[dv], valid_names=["valid"], feval=blend_rmse,
                     callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)])


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def cmd_validate(args):
    dep, X = prepare(args)
    _, art, _ = paths(args)
    is_train = dep["split"].eq("train")
    hold = dep["month"].isin([1, 7])
    reg_m, stale_m = fit_masks(dep, is_train & ~hold, args.fast, args.max_target)
    reg_v, stale_v = fit_masks(dep, is_train & hold, False, args.max_target)
    vm = (is_train & hold & dep[TARGET].notna()).to_numpy()
    if not vm.any():
        sys.exit("No Jan/Jul 2025 departures with a target to validate on.")
    X = encode_cats(X, stale_m)
    y = pd.Series(dep[TARGET], dtype=float)
    stale = pd.Series(dep["stale"], dtype=float)
    print(f"regressor rows {reg_m.sum():,}, classifier rows {stale_m.sum():,}, "
          f"validation rows {vm.sum():,}, features {X.shape[1]}")

    rounds = 1500 if args.fast else 5000
    models = fit_both(X, y, stale, reg_m, stale_m, {"reg": rounds, "stale": rounds}, args.fast,
                      reg_v, stale_v)
    best = {k: m.best_iteration for k, m in models.items()}
    if args.tail:
        rates_m = (is_train & ~hold & dep["stale"].notna() & dep[TARGET].notna()).to_numpy()
        models["late_rates"] = late_unmatched_rates(X, y, stale, rates_m)
        print("Late unmatched (> %ds) rates (stale, day early): %s" % (LATE_UNMATCHED_SEC, ", ".join(
            f"{a} ({s:.2f}, {d:.2f})" for a, (s, d) in models["late_rates"].items())))
    else:
        models["cap"] = args.max_target
    classifier_only, _, _ = predict_mix(models, X.loc[vm])
    print("\nOut-of-fold predictions for the blend booster")
    n_oof, z_oof = oof_parts(X, y, stale, reg_m, stale_m, dep["month"], best, args.fast)
    mix_m, vl = (mix_rows(X, y, m, args.tail, args.max_target) for m in (stale_m, vm))
    Xl = X.loc[vl]
    nl = models["reg"].predict(Xl)
    valid = (stack_features(Xl, nl), y.loc[vl], Xl["mvt_minus_sched"].to_numpy(float), nl,
             models["stale"].predict(Xl, raw_score=True))
    print("Blend booster")
    models["mix"] = fit_mix(stack_features(X.loc[mix_m], n_oof[mix_m]), y.loc[mix_m],
                            X.loc[mix_m, "mvt_minus_sched"].to_numpy(float),
                            n_oof[mix_m], z_oof[mix_m], 3000, args.fast, valid, models.get("cap"))
    Xv = X.loc[vm]
    pred, p, normal = predict_mix(models, Xv)
    yv = y.loc[vm].to_numpy(float)
    c = (yv > 0) & (yv <= args.max_target)

    fit_y = y.loc[stale_m]
    apt_mean = fit_y.groupby(dep.loc[stale_m, "airport"]).mean()
    base = dep.loc[vm, "airport"].map(apt_mean).fillna(fit_y.mean()).to_numpy(float)
    print(f"\nRMSE all rows ({vm.sum():,}): model {rmse(pred, yv):.2f}  |  classifier blend "
          f"{rmse(classifier_only, yv):.2f}  |  normal-taxi model only {rmse(normal, yv):.2f}  |  "
          f"airport-mean baseline {rmse(base, yv):.2f}")
    print(f"RMSE clean (0 < y <= {args.max_target:.0f}s, {c.sum():,} rows): "
          f"model {rmse(pred[c], yv[c]):.2f}  |  classifier blend {rmse(classifier_only[c], yv[c]):.2f}  |  "
          f"airport-mean baseline {rmse(base[c], yv[c]):.2f}")
    sv = stale.loc[vm].to_numpy(float)
    print(f"Stale-block share {np.nanmean(sv):.3f}; mean P(stale) when stale {p[sv == 1].mean():.3f}, "
          f"when not {p[sv == 0].mean():.3f}")
    se, lu = (pred - yv) ** 2, late_unmatched(Xv)
    print("Squared error by row group (rows, share of total, RMS error):")
    for name, g in [("clean", c), ("late unmatched", ~c & lu), ("stale, not clean", ~c & ~lu & (sv == 1)),
                    ("other, not clean", ~c & ~lu & (sv != 1))]:
        if g.any():
            print(f"  {name:<17}{g.sum():>9,}{se[g].sum() / se.sum():>8.1%}{np.sqrt(se[g].mean()):>9.0f}s")

    per = pd.DataFrame({"airport": dep.loc[vm, "airport"].to_numpy(), "y": yv, "p": pred, "b": base,
                        "clean": c})
    per["se"], per["se_b"] = (per.p - per.y) ** 2, (per.b - per.y) ** 2
    tab = per.groupby("airport").agg(n=("y", "size"), rmse_all=("se", "mean"),
                                     baseline_all=("se_b", "mean"))
    tab["rmse_clean"] = per[per.clean].groupby("airport")["se"].mean()
    tab[["rmse_all", "baseline_all", "rmse_clean"]] = np.sqrt(tab[["rmse_all", "baseline_all", "rmse_clean"]])
    print("\nPer airport:\n" + tab.round(1).to_string())

    for name, key, k in [("normal-taxi regressor", "reg", 20), ("stale-block classifier", "stale", 10)]:
        m = models[key]
        imp = pd.Series(m.feature_importance("gain"), index=m.feature_name())
        print(f"\nTop features, {name} (gain share):")
        print((imp / imp.sum()).sort_values(ascending=False).head(k).round(4).to_string())

    art.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: dep.loc[vm, ID].to_numpy(), "airport": dep.loc[vm, "airport"].to_numpy(),
                  "y": yv, "pred": pred, "classifier": classifier_only, "p_stale": p, "normal": normal,
                  "stale": sv, "mvt_minus_sched": Xv["mvt_minus_sched"].to_numpy(float),
                  "unmatched": Xv["unmatched"].to_numpy(float)}).to_parquet(art / "val_predictions.parquet",
                                                                          index=False)
    best["mix"] = models["mix"].best_iteration
    (art / "best_iter.json").write_text(json.dumps(best))
    print(f"\nSaved {art / 'best_iter.json'}: {best}")


def cmd_submit(args):
    data, art, subs = paths(args)
    template = pd.read_parquet(single_file(data, "submitting"))
    dep, X = prepare(args)
    reg_m, stale_m = fit_masks(dep, dep["split"].eq("train"), args.fast, args.max_target)
    rm = dep["split"].eq("rank").to_numpy()
    X = encode_cats(X, stale_m)

    best = art / "best_iter.json"
    rounds = {"reg": 600 if args.fast else 2000, "stale": 300 if args.fast else 1000, "mix": 100}
    if args.rounds:
        rounds.update(reg=args.rounds, stale=args.rounds)
    elif best.exists():
        # More training rows than in validation, so train ~10% longer.
        rounds.update({k: max(int(v * 1.1), 50) for k, v in json.loads(best.read_text()).items()})
    print(f"regressor rows {reg_m.sum():,}, classifier rows {stale_m.sum():,}, "
          f"ranking departures {rm.sum():,}, rounds {rounds}")
    y = pd.Series(dep[TARGET], dtype=float)
    stale = pd.Series(dep["stale"], dtype=float)
    models = fit_both(X, y, stale, reg_m, stale_m, rounds, args.fast)
    if args.tail:
        rates_m = (dep["split"].eq("train") & dep["stale"].notna() & dep[TARGET].notna()).to_numpy()
        models["late_rates"] = late_unmatched_rates(X, y, stale, rates_m)
        print("Late unmatched rates (stale, day early):", models["late_rates"])
    else:
        models["cap"] = args.max_target
    n_oof, z_oof = oof_parts(X, y, stale, reg_m, stale_m, dep["month"], rounds, args.fast)
    mix_m = mix_rows(X, y, stale_m, args.tail, args.max_target)
    models["mix"] = fit_mix(stack_features(X.loc[mix_m], n_oof[mix_m]), y.loc[mix_m],
                            X.loc[mix_m, "mvt_minus_sched"].to_numpy(float),
                            n_oof[mix_m], z_oof[mix_m], rounds["mix"], args.fast, cap=models.get("cap"))
    pred = pd.Series(predict_mix(models, X.loc[rm])[0], index=dep.loc[rm, ID].astype(str).to_numpy())

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
    for c in ["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]:
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
                   help="normal-taxi regressor ignores taxi-out above this many seconds")
    s.add_argument("--fast", action="store_true")
    s.add_argument("--use-actual-obt", action="store_true")
    s.add_argument("--tail", action=argparse.BooleanOptionalAction, default=True,
                   help="price huge late-unmatched targets with per-airport rates and leave the stale "
                        "term uncapped, since the leaderboard scores every row; --no-tail caps "
                        "predictions at max-target")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("submit")
    s.add_argument("--max-target", type=float, default=3600,
                   help="normal-taxi regressor ignores taxi-out above this many seconds")
    s.add_argument("--name", required=True)
    s.add_argument("--version", type=int, default=1)
    s.add_argument("--rounds", type=int)
    s.add_argument("--fast", action="store_true")
    s.add_argument("--use-actual-obt", action="store_true")
    s.add_argument("--tail", action=argparse.BooleanOptionalAction, default=True,
                   help="price huge late-unmatched targets with per-airport rates and leave the stale "
                        "term uncapped, since the leaderboard scores every row; --no-tail caps "
                        "predictions at max-target")
    s.set_defaults(func=cmd_submit)

    s = sub.add_parser("synth", help="write small synthetic data for testing")
    s.add_argument("--rows", type=int, default=4000)
    s.set_defaults(func=cmd_synth)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
