#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weather AI — Train with Data + Wind + optional ONI (2014–2023), predict by DATE only.
Patched: Add NumPy 2.x pickle compat shim + metrics by N on 2023 test set.

Reads
- Class:  Data2014.xlsx ... Data2023.xlsx  (temp/humidity/rain-class)
- Wind :  Wind 2014.xlsx ... Wind 2023.xlsx (wind speed/dir + barometer)
- ONI  :  INO.xlsx or ONI.xlsx (Year + January..December), optional

CLI
  python Ai.py fit --data_dir "C:/.../Class" --wind_dir "C:/.../Wind" --models_dir ./models
  python Ai.py predict --models_dir ./models --date 2025-09-16 [--thr 0.65]
  python Ai.py forecast --models_dir ./models --out ./forecast_two_years.csv [--thr 0.65]
"""
from __future__ import annotations
import argparse, json, re, sys, types
from pathlib import Path
from datetime import date
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    f1_score,
    confusion_matrix,
)
import joblib

# ---- NumPy 2.x pickle compatibility shim ----
def _enable_numpy2_pickle_compat():
    """Allow loading models that were pickled under NumPy >=2.0 on NumPy 1.24.x."""
    if "numpy._core" not in sys.modules:
        core_mod = types.ModuleType("numpy._core")
        core_mod.__dict__.update(np.core.__dict__)
        sys.modules["numpy._core"] = core_mod
        try:
            sys.modules["numpy._core.multiarray"] = np.core.multiarray
        except Exception:
            pass
        try:
            sys.modules["numpy._core.numeric"] = np.core.numeric
        except Exception:
            pass
        try:
            sys.modules["numpy._core.fromnumeric"] = np.core.fromnumeric
        except Exception:
            pass
        if hasattr(np.core, "_exceptions"):
            sys.modules["numpy._core._exceptions"] = np.core._exceptions

# ---------- Parsers for the given Excel layout ----------
DIR2DEG = {
    "N": 0,
    "NNE": 22.5,
    "NE": 45,
    "ENE": 67.5,
    "E": 90,
    "ESE": 112.5,
    "SE": 135,
    "SSE": 157.5,
    "S": 180,
    "SSW": 202.5,
    "SW": 225,
    "WSW": 247.5,
    "W": 270,
    "WNW": 292.5,
    "NW": 315,
    "NNW": 337.5,
}


def _temp_pair(s):
    if not isinstance(s, str):
        return (np.nan, np.nan)
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", s)
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)


def _is_rain(s):
    return isinstance(s, str) and ("rain" in s.lower())


def _dir_deg(x):
    if not isinstance(x, str):
        return np.nan
    return DIR2DEG.get(x.strip().upper(), np.nan)


def _to_float(x, unit=None):
    if isinstance(x, (int, float)) and np.isfinite(x):
        return float(x)
    if not isinstance(x, str):
        return np.nan
    s = x.lower()
    if unit:
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*" + re.escape(unit), s)
        if m:
            return float(m.group(1))
    try:
        return float(x)
    except Exception:
        return np.nan


# ---------- File discovery ----------
def _find_xlsx(root: Path, prefix: str):
    if not root:
        return []
    out = []
    for y in range(2014, 2024):  # fixed to 2014..2023
        pats = [f"{prefix}{y}.xlsx", f"{prefix} {y}.xlsx", f"{prefix}_{y}.xlsx"]
        for name in pats:
            p = root / name
            if p.exists():
                out.append(p)
                break
    return out


# ---------- Build daily Data ----------
def read_daily_data(files) -> pd.DataFrame:
    rows = []
    for f in files:
        df = pd.read_excel(f)
        df2 = df.iloc[2:].copy()
        df2["Date"] = pd.to_datetime(df2["Date"], errors="coerce")
        df2 = df2.dropna(subset=["Date"])
        temp_cols = ["Day Time", "Unnamed: 4", "Night Time", "Unnamed: 10"]
        weather_cols = ["Unnamed: 2", "Unnamed: 5", "Unnamed: 8", "Unnamed: 11"]
        hum_cols = ["Unnamed: 3", "Unnamed: 6", "Unnamed: 9", "Unnamed: 12"]
        for _, r in df2.iterrows():
            his, los, hums = [], [], []
            rainy = False
            for tcol, wcol, hcol in zip(temp_cols, weather_cols, hum_cols):
                hi, lo = _temp_pair(r.get(tcol, np.nan))
                his.append(hi)
                los.append(lo)
                hv = r.get(hcol, np.nan)
                if pd.isna(hv):
                    hums.append(np.nan)
                else:
                    try:
                        hvf = float(hv)
                        if 0 <= hvf <= 1.2:
                            hvf *= 100.0
                        hums.append(hvf)
                    except Exception:
                        hums.append(np.nan)
                rainy = rainy or _is_rain(r.get(wcol, None))
            tmax = np.nanmax(his)
            tmin = np.nanmin(los)
            tavg = np.nanmean(
                [(h + l) / 2 for h, l in zip(his, los) if np.isfinite(h) and np.isfinite(l)]
            )
            hum = np.nanmean(hums)
            rows.append(
                dict(
                    date=r["Date"].date(),
                    t_max_c=tmax,
                    t_min_c=tmin,
                    t_avg_c=tavg,
                    humidity_avg_pct=hum,
                    rain=int(rainy),
                )
            )
    d = (
        pd.DataFrame(rows)
        .dropna(subset=["date"])
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    keep = ~d[["t_max_c", "t_min_c", "t_avg_c", "humidity_avg_pct"]].isna().all(axis=1)
    return d[keep].reset_index(drop=True)


def read_daily_wind(files) -> pd.DataFrame:
    rows = []
    for f in files:
        df = pd.read_excel(f)
        df2 = df.iloc[2:].copy()
        df2["Date"] = pd.to_datetime(df2["Date"], errors="coerce")
        df2 = df2.dropna(subset=["Date"])
        speed_cols = ["Day Time", "Unnamed: 4", "Night Time", "Unnamed: 10"]
        dir_cols = ["Unnamed: 2", "Unnamed: 5", "Unnamed: 8", "Unnamed: 11"]
        baro_cols = ["Unnamed: 3", "Unnamed: 6", "Unnamed: 9", "Unnamed: 12"]
        for _, r in df2.iterrows():
            spd = [_to_float(r.get(c, np.nan), "km/h") for c in speed_cols]
            bar = [_to_float(r.get(c, np.nan), "mbar") for c in baro_cols]
            deg = [_dir_deg(r.get(c, np.nan)) for c in dir_cols]
            sin = [np.sin(np.deg2rad(d)) for d in deg if np.isfinite(d)]
            cos = [np.cos(np.deg2rad(d)) for d in deg if np.isfinite(d)]
            rows.append(
                dict(
                    date=r["Date"].date(),
                    wind_speed_mean_kmh=np.nanmean(spd),
                    barometer_mean_mbar=np.nanmean(bar),
                    wind_dir_sin=np.nanmean(sin) if sin else np.nan,
                    wind_dir_cos=np.nanmean(cos) if cos else np.nan,
                )
            )
    return (
        pd.DataFrame(rows)
        .dropna(subset=["date"])
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )


# ---------- Feature maker ----------
def add_date_feats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["year"] = out["date"].dt.year.astype(int)
    out["month"] = out["date"].dt.month.astype(int)
    out["doy"] = out["date"].dt.dayofyear.astype(int)
    out["doy_sin"] = np.sin(2 * np.pi * out["doy"] / 366.0)
    out["doy_cos"] = np.cos(2 * np.pi * out["doy"] / 366.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    out["is_rainy_season"] = out["month"].isin([5, 6, 7, 8, 9, 10]).astype(int)
    out["year_min"] = out["year"].min()
    out["year_max"] = out["year"].max()
    out["year_norm"] = (out["year"] - out["year_min"]) / (
        (out["year_max"] - out["year_min"]) + 1e-9
    )
    return out


def make_reg():
    return GradientBoostingRegressor(random_state=42)


def make_clf():
    return GradientBoostingClassifier(random_state=42)


# ---------- ONI Excel ----------
MONTH_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def _find_oni_xlsx(data_dir: Path, wind_dir: Path) -> Path | None:
    cands = [
        Path.cwd() / "INO.xlsx",
        Path.cwd() / "ONI.xlsx",
        data_dir / "INO.xlsx",
        data_dir / "ONI.xlsx",
        wind_dir / "INO.xlsx",
        wind_dir / "ONI.xlsx",
    ]
    for p in cands:
        if p.exists():
            return p
    return None


def read_oni_xlsx(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df_long = df.melt(id_vars=["Year"], var_name="month_name", value_name="oni")
    df_long["month"] = df_long["month_name"].map(MONTH_MAP).astype(int)
    df_long["date"] = pd.to_datetime(
        dict(year=df_long["Year"], month=df_long["month"], day=1)
    )
    out = df_long[["date", "oni"]].sort_values("date").reset_index(drop=True)
    out["oni"] = pd.to_numeric(out["oni"], errors="coerce").fillna(0.0)
    return out


def add_oni_feats(daily: pd.DataFrame, oni_monthly: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["_m"] = d["date"].dt.to_period("M")

    m = oni_monthly.copy()
    m["_m"] = m["date"].dt.to_period("M")

    d = d.merge(m[["_m", "oni"]], on="_m", how="left").drop(columns=["_m"])
    d = d.sort_values("date").reset_index(drop=True)

    # lag 1–3 เดือนด้วย period
    month_oni = dict(zip(m["_m"], m["oni"]))

    def get_oni(ts, k=0):
        mm = (pd.Timestamp(ts).to_period("M") - k)
        return float(month_oni.get(mm, 0.0))

    d["oni"] = d["date"].apply(lambda x: get_oni(x, 0))
    d["oni_lag1"] = d["date"].apply(lambda x: get_oni(x, 1))
    d["oni_lag2"] = d["date"].apply(lambda x: get_oni(x, 2))
    d["oni_lag3"] = d["date"].apply(lambda x: get_oni(x, 3))
    return d


def _oni_from_meta(dt: date, meta: dict) -> dict:
    out = {"oni": 0.0, "oni_lag1": 0.0, "oni_lag2": 0.0, "oni_lag3": 0.0}
    if not meta.get("has_oni"):
        return out
    month_map = meta.get("oni_monthly") or {}

    def get(mdt, k=0):
        y_m = (pd.Timestamp(mdt).to_period("M") - k).strftime("%Y-%m")
        return float(month_map.get(y_m, 0.0))

    return {
        "oni": get(dt, 0),
        "oni_lag1": get(dt, 1),
        "oni_lag2": get(dt, 2),
        "oni_lag3": get(dt, 3),
    }


# ---------- Fit / Save ----------
def fit_and_save(data_dir: Path, wind_dir: Path, models_dir: Path):
    models_dir.mkdir(parents=True, exist_ok=True)
    data_files = _find_xlsx(Path(data_dir), "Data")
    wind_files = _find_xlsx(Path(wind_dir), "Wind")
    if not data_files:
        raise FileNotFoundError(f"No Data20xx.xlsx (2014–2023) in {data_dir}")
    if not wind_files:
        raise FileNotFoundError(f"No Wind 20xx.xlsx (2014–2023) in {wind_dir}")
    daily = add_date_feats(
        pd.merge(
            read_daily_data(data_files),
            read_daily_wind(wind_files),
            on="date",
            how="left",
        )
    )

    # Optional ONI
    oni_path = _find_oni_xlsx(Path(data_dir), Path(wind_dir))
    oni_feats = []
    oni_monthly_dict = None
    has_oni = False
    if oni_path is not None:
        try:
            oni_tbl = read_oni_xlsx(oni_path)
            daily = add_oni_feats(daily, oni_tbl)
            oni_feats = ["oni", "oni_lag1", "oni_lag2", "oni_lag3"]
            oni_monthly_dict = {
                pd.to_datetime(r["date"]).strftime("%Y-%m"): float(r["oni"])
                for _, r in oni_tbl.iterrows()
            }
            has_oni = True
            print(f"[INFO] ONI features loaded from {oni_path}")
        except Exception as e:
            print(f"[WARN] Cannot read ONI Excel: {e}")

    # splits
    train = daily[daily["year"] <= 2021].copy()
    val = daily[daily["year"] == 2022].copy()
    test = daily[daily["year"] == 2023].copy()

    date_feats = [
        "doy_sin",
        "doy_cos",
        "year_norm",
        "month",
        "month_sin",
        "month_cos",
        "is_rainy_season",
    ]
    wind_targets = [
        "wind_speed_mean_kmh",
        "barometer_mean_mbar",
        "wind_dir_sin",
        "wind_dir_cos",
    ]

    # Stage A: learn wind from date
    wind_models = {}
    for t in wind_targets:
        m = make_reg()
        m.fit(train[date_feats], train[t])
        wind_models[t] = m

    def add_wpred(df):
        X = df[date_feats].copy()
        for t in wind_targets:
            df[f"wp_{t}"] = wind_models[t].predict(X)
        return df

    train = add_wpred(train)
    val = add_wpred(val)
    test = add_wpred(test)

    reg_targets = ["t_min_c", "t_max_c", "t_avg_c", "humidity_avg_pct"]
    final_feats = date_feats + [f"wp_{t}" for t in wind_targets] + oni_feats

    # regressors
    reg_models = {}
    for t in reg_targets:
        m = make_reg()
        m.fit(train[final_feats], train[t])
        reg_models[t] = m

    # classifier + tuning
    Xtr, ytr = train[final_feats], train["rain"]
    Xv, yv = val[final_feats], val["rain"]
    cls_counts = ytr.value_counts().to_dict()
    wpos = 1.0 / max(cls_counts.get(1, 1), 1)
    wneg = 1.0 / max(cls_counts.get(0, 1), 1)
    sw = np.where(ytr.values == 1, wpos, wneg)
    grid = GridSearchCV(
        make_clf(),
        dict(n_estimators=[150, 250], learning_rate=[0.05, 0.1], max_depth=[2, 3]),
        scoring="f1",
        cv=TimeSeriesSplit(n_splits=3),
        refit=True,
    )
    grid.fit(Xtr, ytr, **{"sample_weight": sw})
    clf = grid.best_estimator_

    vp = clf.predict_proba(Xv)[:, 1]
    best_thr = 0.5
    best_f1 = -1
    for thr in np.linspace(0.3, 0.7, 41):
        f = f1_score(yv, (vp >= thr).astype(int), zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thr = float(thr)
    rain_threshold = best_thr

    # ====== Loss curve (training deviance) ของ classifier ======
    try:
        if hasattr(clf, "train_score_"):
            iters = np.arange(1, len(clf.train_score_) + 1)
            loss_df = pd.DataFrame(
                {"iter": iters, "train_deviance": clf.train_score_}
            )
            loss_csv = models_dir / "clf_rain_loss_curve.csv"
            loss_df.to_csv(loss_csv, index=False)

            plt.figure(figsize=(8, 5))
            plt.plot(iters, clf.train_score_)
            plt.title("Rain Classifier — Training Loss Curve")
            plt.xlabel("Iteration")
            plt.ylabel("Deviance (Loss)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(models_dir / "clf_rain_loss_curve.png", dpi=150)
            plt.close()

            print(
                "[INFO] Saved loss curve CSV and PNG:",
                loss_csv,
                "and clf_rain_loss_curve.png",
            )
    except Exception as e:
        print("[WARN] Cannot compute/plot loss curve:", e)

    # ====== F1 curve บน validation set ตามจำนวนต้นไม้ ======
    try:
        if hasattr(clf, "staged_predict_proba"):
            iter_list = []
            f1_list = []
            thr_curve = rain_threshold
            for i, proba_i in enumerate(clf.staged_predict_proba(Xv), start=1):
                p_i = proba_i[:, 1]
                y_pred_i = (p_i >= thr_curve).astype(int)
                f1_i = f1_score(yv, y_pred_i, zero_division=0)
                iter_list.append(i)
                f1_list.append(f1_i)

            if iter_list:
                f1_df = pd.DataFrame({"iter": iter_list, "F1_val": f1_list})
                f1_csv = models_dir / "clf_rain_f1_curve.csv"
                f1_df.to_csv(f1_csv, index=False)

                plt.figure(figsize=(8, 5))
                plt.plot(iter_list, f1_list)
                plt.title("Rain Classifier — Validation F1 Curve")
                plt.xlabel("Iteration")
                plt.ylabel("F1 (val)")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(models_dir / "clf_rain_f1_curve.png", dpi=150)
                plt.close()

                print(
                    "[INFO] Saved F1 curve CSV and PNG:",
                    f1_csv,
                    "and clf_rain_f1_curve.png",
                )
    except Exception as e:
        print("[WARN] Cannot compute/plot F1 curve:", e)

    # ====== evaluate on 2023 (regression + overall cls) ======
    rows = []

    def add_reg(name, pred, true):
        mae = mean_absolute_error(true, pred)
        rmse = float(np.sqrt(mean_squared_error(true, pred)))
        rows.append(dict(target=name, MAE=mae, RMSE=rmse))

    for t in reg_targets:
        add_reg(t, reg_models[t].predict(test[final_feats]), test[t])

    proba = clf.predict_proba(test[final_feats])[:, 1]
    pred = (proba >= rain_threshold).astype(int)
    acc = accuracy_score(test["rain"], pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        test["rain"], pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(test["rain"], proba)
    except Exception:
        auc = float("nan")
    rows.append(
        dict(
            target="rain (cls)",
            MAE=np.nan,
            RMSE=np.nan,
            Accuracy=acc,
            Precision=prec,
            Recall=rec,
            F1=f1,
            AUC=auc,
            Threshold=rain_threshold,
        )
    )
    pd.DataFrame(rows).to_csv(models_dir / "model_metrics_2023.csv", index=False)

    # ====== classification metrics by N (N=30, 365, 100) บน test 2023 ======
    metrics_by_N = []
    Ns = [30, 365, 100]
    test_sorted = test.sort_values("date")  # เผื่อไม่เรียง

    for N in Ns:
        if len(test_sorted) < N:
            print(f"[WARN] Test size {len(test_sorted)} < N={N}, ข้ามไป")
            continue

        sub = test_sorted.tail(N)
        proba_N = clf.predict_proba(sub[final_feats])[:, 1]
        y_pred_N = (proba_N >= rain_threshold).astype(int)
        y_true_N = sub["rain"].values.astype(int)

        acc_N = accuracy_score(y_true_N, y_pred_N)
        prec_N, rec_N, f1_N, _ = precision_recall_fscore_support(
            y_true_N, y_pred_N, average="binary", zero_division=0
        )
        mse_N = mean_squared_error(y_true_N, y_pred_N)
        rmse_N = float(np.sqrt(mse_N))

        cm = confusion_matrix(y_true_N, y_pred_N, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # กรณีมีคลาสเดียวในชุด N (กันไว้เฉย ๆ)
            tn = fp = fn = tp = 0
            if 0 in np.unique(y_true_N):
                tn = int((y_true_N == 0).sum())
            if 1 in np.unique(y_true_N):
                tp = int((y_true_N == 1).sum())

        metrics_by_N.append(
            dict(
                N=N,
                TP=int(tp),
                TN=int(tn),
                FP=int(fp),
                FN=int(fn),
                Accuracy=acc_N,
                Precision=prec_N,
                Recall=rec_N,
                F1=f1_N,
                MSE=mse_N,
                RMSE=rmse_N,
            )
        )

    if metrics_by_N:
        df_N = pd.DataFrame(metrics_by_N)
        out_N = models_dir / "rain_metrics_by_N_2023.csv"
        df_N.to_csv(out_N, index=False)
        print("\n[INFO] Rain metrics by N on 2023 test set (saved to", out_N, "):")
        print(df_N)

    # save models + meta
    for t, m in wind_models.items():
        joblib.dump(m, models_dir / f"wind_{t}.pkl")
    for t, m in reg_models.items():
        joblib.dump(m, models_dir / f"reg_{t}.pkl")
    joblib.dump(clf, models_dir / "clf_rain.pkl")
    meta = dict(
        date_feats=date_feats,
        wind_targets=wind_targets,
        reg_targets=reg_targets,
        final_feats=final_feats,
        year_min=int(daily["year"].min()),
        year_max=int(daily["year"].max()),
        rain_threshold=rain_threshold,
        has_oni=has_oni,
        oni_feats=oni_feats,
        oni_monthly=oni_monthly_dict,
    )
    (models_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    daily.to_csv(models_dir / "daily_merged_dataset.csv", index=False)
    print("Saved models to:", models_dir)
    print(pd.DataFrame(rows))


# ---------- Load / Predict ----------
def _load(models_dir: Path):
    _enable_numpy2_pickle_compat()
    meta = json.loads((models_dir / "meta.json").read_text(encoding="utf-8"))
    wind = {t: joblib.load(models_dir / f"wind_{t}.pkl") for t in meta["wind_targets"]}
    regs = {t: joblib.load(models_dir / f"reg_{t}.pkl") for t in meta["reg_targets"]}
    clf = joblib.load(models_dir / "clf_rain.pkl")
    return meta, wind, regs, clf


def _date_feats(dt: date, y0: int, y1: int) -> pd.DataFrame:
    ts = pd.Timestamp(dt)
    doy = int(ts.timetuple().tm_yday)
    m = int(ts.month)
    return pd.DataFrame(
        [
            {
                "doy_sin": np.sin(2 * np.pi * doy / 366.0),
                "doy_cos": np.cos(2 * np.pi * doy / 366.0),
                "year_norm": (ts.year - y0) / ((y1 - y0) + 1e-9),
                "month": m,
                "month_sin": np.sin(2 * np.pi * m / 12.0),
                "month_cos": np.cos(2 * np.pi * m / 12.0),
                "is_rainy_season": 1 if m in [5, 6, 7, 8, 9, 10] else 0,
            }
        ]
    )


def predict_one(models_dir: Path, date_str: str, thr: float | None = None) -> dict:
    meta, wind, regs, clf = _load(models_dir)
    dt = pd.to_datetime(date_str).date()
    X = _date_feats(dt, meta["year_min"], meta["year_max"])
    for t in meta["wind_targets"]:
        X[f"wp_{t}"] = wind[t].predict(X[meta["date_feats"]])

    # ONI features if available (from meta)
    if meta.get("has_oni"):
        oni_vals = _oni_from_meta(dt, meta)
        for k, v in oni_vals.items():
            X[k] = v
        for k in meta.get("oni_feats", []):
            if k not in X:
                X[k] = 0.0

    preds = {
        t: float(regs[t].predict(X[meta["final_feats"]])[0])
        for t in meta["reg_targets"]
    }
    p = float(clf.predict_proba(X[meta["final_feats"]])[:, 1][0])
    thr = float(meta.get("rain_threshold", 0.5)) if thr is None else float(thr)
    return dict(
        date=dt.isoformat(),
        t_min_c_pred=preds["t_min_c"],
        t_max_c_pred=preds["t_max_c"],
        t_avg_c_pred=preds["t_avg_c"],
        humidity_avg_pct_pred=preds["humidity_avg_pct"],
        rain_probability=p,
        rain_predicted=("Rain" if p >= thr else "Sunny"),
    )


def forecast_two_years(models_dir: Path, out_csv: Path, thr: float | None = None):
    meta, _, _, _ = _load(models_dir)
    last = meta["year_max"]
    dates = pd.date_range(f"{last+1}-01-01", f"{last+2}-12-31", freq="D")
    rows = [predict_one(models_dir, d.date().isoformat(), thr) for d in dates]
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved:", out_csv)


# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    f = sub.add_parser("fit")
    f.add_argument("--data_dir", required=True)
    f.add_argument("--wind_dir", required=True)
    f.add_argument("--models_dir", default="./models")
    q = sub.add_parser("predict")
    q.add_argument("--models_dir", default="./models")
    q.add_argument("--date", required=True)
    q.add_argument("--thr", type=float, default=None)
    r = sub.add_parser("forecast")
    r.add_argument("--models_dir", default="./models")
    r.add_argument("--out", default="./forecast_two_years.csv")
    r.add_argument("--thr", type=float, default=None)
    a = p.parse_args()
    if a.cmd is None:
        prog = Path(sys.argv[0]).name
        print("\nExamples:")
        print(
            f'  python {prog} fit --data_dir "C:/Users/Champ/Desktop/V6/Class" --wind_dir "C:/Users/Champ/Desktop/V6/Wind" --models_dir ./models'
        )
        print(
            f"  python {prog} predict --models_dir ./models --date 2025-09-16 --thr 0.65"
        )
        print(
            f"  python {prog} forecast --models_dir ./models --out ./forecast_two_years.csv --thr 0.65"
        )
        return
    if a.cmd == "fit":
        fit_and_save(Path(a.data_dir), Path(a.wind_dir), Path(a.models_dir))
    elif a.cmd == "predict":
        print(
            json.dumps(
                predict_one(Path(a.models_dir), a.date, thr=a.thr),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif a.cmd == "forecast":
        forecast_two_years(Path(a.models_dir), Path(a.out), thr=a.thr)


if __name__ == "__main__":
    main()
