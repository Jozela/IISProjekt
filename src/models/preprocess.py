import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NesreceWeatherPreprocessor(BaseEstimator, TransformerMixin):
    """
    Builds an hourly (or configurable frequency) grid per obcina and merges weather,
    producing a row per (obcinaNaziv, time_slot) with label=1 if any accident happened in that slot.
    """
    def __init__(self, time_freq="h"):
        self.time_freq = time_freq

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        nesrece_df, vreme_df = X

        nesrece_df = nesrece_df.copy()
        vreme_df = vreme_df.copy()

        nesrece_df["nastanekCas"] = pd.to_datetime(nesrece_df["nastanekCas"], errors="coerce")
        nesrece_df = nesrece_df.dropna(subset=["nastanekCas", "obcinaNaziv"])

        # slot to given frequency (hourly by default)
        nesrece_df["time_slot"] = nesrece_df["nastanekCas"].dt.floor(self.time_freq)

        vreme_df["date"] = pd.to_datetime(vreme_df["date"], errors="coerce")
        vreme_df = vreme_df.dropna(subset=["date"])
        vreme_df["time_slot"] = vreme_df["date"].dt.floor(self.time_freq)

        # Accidents per (obcina, time_slot) => binary label
        accidents = (
            nesrece_df
            .groupby(["obcinaNaziv", "time_slot"])
            .size()
            .reset_index(name="accident_count")
        )
        accidents["label"] = (accidents["accident_count"] > 0).astype(int)

        # Full grid (obcina x all slots)
        all_obcine = nesrece_df["obcinaNaziv"].unique()

        start = min(nesrece_df["time_slot"].min(), vreme_df["time_slot"].min())
        end   = max(nesrece_df["time_slot"].max(), vreme_df["time_slot"].max())

        all_slots = pd.date_range(
            start=start,
            end=end,
            freq=self.time_freq,
        )

        grid = (
            pd.MultiIndex.from_product([all_obcine, all_slots], names=["obcinaNaziv", "time_slot"])
            .to_frame(index=False)
        )

        grid = grid.merge(
            accidents[["obcinaNaziv", "time_slot", "label"]],
            on=["obcinaNaziv", "time_slot"],
            how="left",
        )
        grid["label"] = grid["label"].fillna(0).astype(int)

        # Merge weather (expects weather column "obcina" matching obcinaNaziv)
        grid = grid.merge(
            vreme_df.drop(columns=["date"], errors="ignore"),
            left_on=["obcinaNaziv", "time_slot"],
            right_on=["obcina", "time_slot"],
            how="left",
        ).drop(columns=["obcina"], errors="ignore")

        # Time features
        grid["day_of_week"] = grid["time_slot"].dt.dayofweek
        grid["month"] = grid["time_slot"].dt.month
        grid["is_weekend"] = (grid["day_of_week"] >= 5).astype(int)

        # Hour-of-day only makes sense when freq is hourly; still safe to compute:
        grid["hour_of_day"] = grid["time_slot"].dt.hour

        grid["obcina_enc"] = grid["obcinaNaziv"].astype("category").cat.codes

        grid = grid.sort_values(["obcinaNaziv", "time_slot"]).reset_index(drop=True)
        return grid