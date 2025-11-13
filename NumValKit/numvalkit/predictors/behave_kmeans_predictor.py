import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from collections import OrderedDict
from typing import Any, List, Tuple
from tqdm import tqdm
from datetime import datetime, timedelta

from numvalkit.core import BasePredictor
from numvalkit.data_loader import ForegroundLoader
from numvalkit.utils import get_category_from_package


class BehaveKmeansPredictor(BasePredictor):
    """
    KMeans-based predictor for future active app set.
    Inherits from numvalkit.core.BasePredictor.
    """

    def __init__(
        self,
        data_dir: str,
        user_list: List[str],
        cluster_n: int = 3,
        period_min: int = 5,
        window_periods: int = 24,
    ):
        super().__init__()
        self.loader = ForegroundLoader(data_dir=data_dir)
        self.cluster_n = cluster_n
        self.period_min = period_min
        self.user_list = user_list
        self.window_periods = window_periods
        self.user_models = {}  # user -> model dict

    @staticmethod
    def _extract_daily_feature(records):
        df = pd.DataFrame(
            [(r[0].date(), r[2]) for r in records], columns=["day", "pkg"]
        )
        return df.groupby(["day", "pkg"]).size().unstack(fill_value=0)

    @staticmethod
    def _extract_daily_feature_by_category(records):
        """
        records: list of (start: datetime, end: datetime, pkg: str)
        returns: DataFrame indexed by day, columns=category, values=total seconds
        that category was in foreground on that day.
        """
        rows = []
        for start, end, pkg in records:
            cat = get_category_from_package(pkg)
            cur = start
            while cur.date() < end.date():
                midnight = datetime.combine(
                    cur.date(), datetime.min.time()
                ) + timedelta(days=1)
                rows.append((cur.date(), cat, (midnight - cur).total_seconds()))
                cur = midnight
            rows.append((cur.date(), cat, (end - cur).total_seconds()))

        df = pd.DataFrame(rows, columns=["day", "category", "duration"])
        feats = df.groupby(["day", "category"])["duration"].sum().unstack(fill_value=0)
        return feats

    @staticmethod
    def _build_period_feature(records, period_min):
        df = pd.DataFrame(
            [(r[0], r[2], r[1]) for r in records], columns=["start", "pkg", "end"]
        )
        df["day"] = df["start"].dt.date
        period_cnt = (24 * 60) // period_min
        all_days = sorted(df["day"].unique())
        full_index = pd.MultiIndex.from_product(
            [all_days, range(period_cnt)], names=["day", "period_id"]
        )
        pkgs = sorted(df["pkg"].unique())
        period_df = pd.DataFrame(0, index=full_index, columns=pkgs)
        for start, end, pkg in records:
            day = start.date()
            s = start.hour * 60 + start.minute
            e = end.hour * 60 + end.minute if end.date() == day else 24 * 60 - 1
            s_pid = s // period_min
            e_pid = e // period_min
            for pid in range(s_pid, e_pid + 1):
                period_df.at[(day, pid), pkg] = 1
        return period_df

    @staticmethod
    def _build_numpy_tensor(period_df):
        apps = list(period_df.columns)
        app2idx = {a: i for i, a in enumerate(apps)}
        days = sorted({d for d, _ in period_df.index})
        pids = sorted({p for _, p in period_df.index})
        day2idx = {d: i for i, d in enumerate(days)}
        pid2idx = {p: i for i, p in enumerate(pids)}
        arr = np.zeros((len(days), len(pids), len(apps)), dtype=int)
        for (d, p), row in period_df.iterrows():
            di, pi = day2idx[d], pid2idx[p]
            for pkg, val in row.items():
                arr[di, pi, app2idx[pkg]] = val
        return arr, app2idx, day2idx, pid2idx

    @staticmethod
    def _fast_window_pkg_counts(cur_pid, same_idxs, window_periods, arr, pid2idx):
        counts = np.zeros(arr.shape[2], dtype=int)
        for di in same_idxs:
            for w in range(window_periods):
                pid = cur_pid + w
                if pid in pid2idx:
                    counts += arr[di, pid2idx[pid]]
        return counts

    def fit(self, users: List[str] = None) -> None:
        """
        Load foreground records per user and fit clustering in parallel.
        `users` is list of user IDs to train on.
        """
        users = users or self.user_list

        def _fit_one(user):
            recs = self.loader.load(user, True) or []  # Remove off screen
            feats = self._extract_daily_feature(recs)
            # ==== Use Categories For Clustering ====
            # feats = self._extract_daily_feature_by_category(recs)
            if feats.shape[0] < self.cluster_n + 1:
                return None
            kmeans = KMeans(n_clusters=self.cluster_n, random_state=0, n_init=10)
            kmeans.fit(feats.values)
            labels = pd.Series(kmeans.labels_, index=feats.index)
            period_df = self._build_period_feature(recs, self.period_min)
            arr, a2i, d2i, p2i = self._build_numpy_tensor(period_df)
            return user, {
                "feats": feats,
                "kmeans": kmeans,
                "labels": labels,
                "period_df": period_df,
                "arr": arr,
                "app2idx": a2i,
                "day2idx": d2i,
                "pid2idx": p2i,
            }

        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(_fit_one)(u) for u in tqdm(users, desc="Extract users")
        )
        for item in results:
            if item:
                user, model = item
                self.user_models[user] = model

    def predict(
        self, user: str, session: List[Tuple[datetime, datetime, str]], top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Predict top-K future active apps based on a recent session of foreground records.

        Args:
            user: user identifier
            session: list of (start, end, pkg) tuples representing recent foreground apps
            top_k: number of apps to return

        Returns:
            List of (app, probability) tuples for the top_k predicted apps
        """
        model = self.user_models.get(user)
        if not model or not session:
            return []
        # Use the last event timestamp to determine current day and period
        last_ts = session[-1][1]
        day = last_ts.date()
        pid = (last_ts.hour * 60 + last_ts.minute) // self.period_min

        feats = model["feats"]
        labels = model["labels"]
        # If we have no label for this day, cannot predict
        if day not in labels.index:  # Topk user global topk app
            freq = feats.sum(axis=0)
            top_apps = freq.sort_values(ascending=False).index[:top_k]
            total_freq = freq.sum()
            return [(app, float(freq[app] / total_freq)) for app in top_apps]
        label = labels[day]
        # Gather other days with same label
        same_days = [d for d in labels.index if labels[d] == label and d != day]
        same_idxs = [model["day2idx"][d] for d in same_days if d in model["day2idx"]]

        # Compute future usage counts over window_periods
        counts = self._fast_window_pkg_counts(
            pid, same_idxs, self.window_periods, model["arr"], model["pid2idx"]
        )
        if counts.sum() == 0:  # Topk user global topk app
            freq = feats.sum(axis=0)
            top_apps = freq.sort_values(ascending=False).index[:top_k]
            total_freq = freq.sum()
            return [(app, float(freq[app] / total_freq)) for app in top_apps]
        probs = counts / counts.sum()
        # Take top_k highest-probability apps
        idxs = np.argsort(-probs)[:top_k]
        idx2app = {i: a for a, i in model["app2idx"].items()}
        return [(idx2app[i], float(probs[i])) for i in idxs if probs[i] > 0]
