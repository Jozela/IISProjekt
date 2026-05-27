import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NesreceWeatherPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, time_freq="D"):
        self.time_freq = time_freq

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        nesrece_df, vreme_df = X

        nesrece_df = nesrece_df.copy()
        vreme_df   = vreme_df.copy()

        nesrece_df["nastanekCas"] = pd.to_datetime(nesrece_df["nastanekCas"], errors="coerce")
        nesrece_df["day_slot"]    = nesrece_df["nastanekCas"].dt.floor("D")
        nesrece_df["hour"]        = nesrece_df["nastanekCas"].dt.hour

        vreme_df["date"]     = pd.to_datetime(vreme_df["date"], errors="coerce")
        vreme_df["day_slot"] = vreme_df["date"].dt.floor("D")

        # Accidents per (obcina, day)
        accidents = (
            nesrece_df
            .groupby(["obcinaNaziv", "day_slot"])
            .size()
            .reset_index(name="accident_count")
        )
        accidents["label"] = (accidents["accident_count"] > 0).astype(int)

        # Average accident hour per (obcina, day)
        hour_stats = (
            nesrece_df
            .groupby(["obcinaNaziv", "day_slot"])["hour"]
            .mean()
            .reset_index(name="avg_accident_hour")
        )

        # Full (obcina x day) grid
        all_obcine = nesrece_df["obcinaNaziv"].unique()
        all_days   = pd.date_range(
            start=nesrece_df["day_slot"].min(),
            end=nesrece_df["day_slot"].max(),
            freq="D",
        )
        grid = pd.MultiIndex.from_product(
            [all_obcine, all_days], names=["obcinaNaziv", "day_slot"]
        ).to_frame(index=False)

        grid = grid.merge(
            accidents[["obcinaNaziv", "day_slot", "label"]],
            on=["obcinaNaziv", "day_slot"], how="left"
        )
        grid["label"] = grid["label"].fillna(0).astype(int)

        grid = grid.merge(
            hour_stats, on=["obcinaNaziv", "day_slot"], how="left"
        )
        grid["avg_accident_hour"] = grid["avg_accident_hour"].fillna(-1)

        # Merge weather
        grid = grid.merge(
            vreme_df.drop(columns=["date"], errors="ignore"),
            left_on=["obcinaNaziv", "day_slot"],
            right_on=["obcina", "day_slot"],
            how="left",
        ).drop(columns=["obcina"], errors="ignore")

        # Time features
        grid["day_of_week"] = grid["day_slot"].dt.dayofweek
        grid["month"]       = grid["day_slot"].dt.month
        grid["is_weekend"]  = (grid["day_of_week"] >= 5).astype(int)
        grid["obcina_enc"]  = grid["obcinaNaziv"].astype("category").cat.codes
        grid["hour_of_day"] = grid["day_slot"].dt.hour  # will be 0-23 for hourly, 0 for daily
        grid = grid.sort_values(["obcinaNaziv", "day_slot"]).reset_index(drop=True)
        return grid


class SlidingWindowTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, window_size):
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self._create_windows(X, self.window_size)

    @staticmethod
    def _create_windows(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i : i + window_size])
            y.append(data[i + window_size, -1])
        return np.array(X), np.array(y)