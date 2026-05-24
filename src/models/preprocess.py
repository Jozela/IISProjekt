import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NesreceWeatherPreprocessor(BaseEstimator, TransformerMixin):
    """
    Merges accident data with weather data and builds a
    per-municipality per-hour feature table with a binary target:
      1 = at least one accident occurred in that (obcina, hour) slot
      0 = no accident
    """
    def __init__(self, time_freq="h"):
        self.time_freq = time_freq

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        nesrece_df, vreme_df = X

        # ── 1. Parse & floor timestamps ──────────────────────────────────────
        nesrece_df = nesrece_df.copy()
        vreme_df   = vreme_df.copy()

        nesrece_df["nastanekCas"] = pd.to_datetime(
            nesrece_df["nastanekCas"], errors="coerce"
        )
        nesrece_df["hour_slot"] = nesrece_df["nastanekCas"].dt.floor(self.time_freq)

        vreme_df["date"] = pd.to_datetime(vreme_df["date"], errors="coerce")
        # weather is daily → broadcast to every hour of that day
        vreme_df["hour_slot"] = vreme_df["date"].dt.floor("D")

        # ── 2. Build accident labels ─────────────────────────────────────────
        # Count accidents per (obcina, hour_slot), then binarise
        accidents = (
            nesrece_df
            .groupby(["obcinaNaziv", "hour_slot"])
            .size()
            .reset_index(name="accident_count")
        )
        accidents["label"] = (accidents["accident_count"] > 0).astype(int)

        # ── 3. Build full (obcina × hour) grid ──────────────────────────────
        all_obcine   = nesrece_df["obcinaNaziv"].unique()
        min_time     = nesrece_df["hour_slot"].min()
        max_time     = nesrece_df["hour_slot"].max()
        all_hours    = pd.date_range(start=min_time, end=max_time, freq=self.time_freq)

        grid = pd.MultiIndex.from_product(
            [all_obcine, all_hours], names=["obcinaNaziv", "hour_slot"]
        ).to_frame(index=False)

        # Merge accident labels (missing = 0, no accident)
        grid = grid.merge(accidents[["obcinaNaziv", "hour_slot", "label"]],
                          on=["obcinaNaziv", "hour_slot"], how="left")
        grid["label"] = grid["label"].fillna(0).astype(int)

        # ── 4. Merge weather ─────────────────────────────────────────────────
        vreme_df["day_slot"] = vreme_df["date"].dt.floor("D")
        grid["day_slot"]     = grid["hour_slot"].dt.floor("D")

        grid = grid.merge(
            vreme_df.drop(columns=["date", "hour_slot"], errors="ignore"),
            left_on=["obcinaNaziv", "day_slot"],
            right_on=["obcina",     "day_slot"],
            how="left",
        ).drop(columns=["obcina", "day_slot"], errors="ignore")

        # ── 5. Time-based features ───────────────────────────────────────────
        grid["hour_of_day"]  = grid["hour_slot"].dt.hour
        grid["day_of_week"]  = grid["hour_slot"].dt.dayofweek
        grid["month"]        = grid["hour_slot"].dt.month
        grid["is_weekend"]   = (grid["day_of_week"] >= 5).astype(int)

        # ── 6. Encode municipality as integer ────────────────────────────────
        grid["obcina_enc"] = grid["obcinaNaziv"].astype("category").cat.codes

        grid = grid.sort_values(["obcinaNaziv", "hour_slot"]).reset_index(drop=True)
        return grid


class SlidingWindowTransformer(BaseEstimator, TransformerMixin):
    """
    Builds supervised (X, y) pairs from a time-ordered feature matrix
    using a sliding window.
    """
    def __init__(self, window_size):
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self._create_windows(X, self.window_size)

    @staticmethod
    def _create_windows(data, window_size):
        X_list, y_list = [], []
        for i in range(len(data) - window_size):
            X_list.append(data[i : i + window_size])
            y_list.append(data[i + window_size, -1])   # last col = label
        return np.array(X_list), np.array(y_list)