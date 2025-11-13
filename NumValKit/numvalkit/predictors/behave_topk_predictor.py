import os
import pickle
import math
from typing import List, Tuple, Dict
from datetime import datetime
from collections import defaultdict, Counter
from joblib import Parallel, delayed
from tqdm import tqdm

from numvalkit.core import BasePredictor
from numvalkit.data_loader import ForegroundLoader


class BehaveTopKPredictor(BasePredictor):
    """
    A session‐aware Top-K predictor: for each (user, hour) slot it learns the
    history of “next app” frequencies and at predict time falls back:
      1) to that user-hour’s top-K,
      2) to that user’s global top-K,
      3) to the global top-K.
    """

    def __init__(self, data_dir: str, user_list: List[str], gap: int = 60 * 60):
        """
        :param data_dir:   root folder with per‐user foreground.csv
        :param user_list:  list of user IDs
        :param gap:        session gap in seconds
        """
        self.loader = ForegroundLoader(data_dir)
        self.user_list = user_list
        self.gap = gap

        # to be populated in fit() or load()
        self.user_time_counts: Dict[Tuple[str, int], Counter] = {}
        self.global_counter: Counter = Counter()
        self.user_time_topk: Dict[Tuple[str, int], List[Tuple[str, float]]] = {}
        self.global_topk: List[Tuple[str, float]] = []

    def fit(self, users: List[str] = None) -> None:
        """
        Parallelized fit: build per‐user, per‐hour and global next‐app counters,
        then precompute Top-K lists for fast lookup.
        """
        users = users or self.user_list

        # 1. Extract (user, hour, next_pkg) for every session
        def extract_meta(u):
            recs = self.loader.load(u, True) or []
            sessions = []
            if recs:
                cur = [recs[0]]
                for r in recs[1:]:
                    if (r[0] - cur[-1][1]).total_seconds() <= self.gap:
                        cur.append(r)
                    else:
                        sessions.append(cur)
                        cur = [r]
                sessions.append(cur)
            out = []
            for sess in sessions:
                if len(sess) < 2:
                    continue
                hour = sess[0][0].hour
                next_pkg = sess[-1][2]
                out.append((u, hour, next_pkg))
            return out

        all_meta = Parallel(n_jobs=-1, backend="loky")(
            delayed(extract_meta)(u) for u in tqdm(users, desc="Extracting sessions")
        )
        session_meta = [x for sub in all_meta for x in sub]

        # 2. Count in chunks to use all cores
        def process_chunk(chunk):
            local_counts = defaultdict(Counter)
            local_glob = Counter()
            for u, hour, pkg in chunk:
                local_counts[(u, hour)][pkg] += 1
                local_glob[pkg] += 1
            return local_counts, local_glob

        num_cores = os.cpu_count() or 1
        chunk_size = math.ceil(len(session_meta) / num_cores)
        chunks = [
            session_meta[i : i + chunk_size]
            for i in range(0, len(session_meta), chunk_size)
        ]

        parts = Parallel(n_jobs=-1, backend="loky")(
            delayed(process_chunk)(ch)
            for ch in tqdm(chunks, desc="Counting distributions")
        )

        # 3. Merge partial results
        self.user_time_counts = defaultdict(Counter)
        self.global_counter = Counter()
        for local_counts, local_glob in parts:
            for key, ctr in local_counts.items():
                self.user_time_counts[key].update(ctr)
            self.global_counter.update(local_glob)

        # 4. Precompute Top-K lists
        def make_topk(ctr: Counter):
            total = sum(ctr.values())
            return [(pkg, cnt / total) for pkg, cnt in ctr.most_common()]

        self.global_topk = make_topk(self.global_counter)
        self.user_time_topk = {
            key: make_topk(ctr) for key, ctr in self.user_time_counts.items()
        }

    def predict(
        self, user: str, session: List[Tuple[datetime, datetime, str]], top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        1) Use hour of the first event in session to pick user-hour Top-K;
        2) If missing, fall back to user-global Top-K;
        3) If still missing, fall back to global Top-K.
        """
        if not session:
            return self.global_topk[:top_k]

        pred_hour = session[-1][1].hour  # End of last app
        lst = self.user_time_topk.get((user, pred_hour))
        if lst:
            return lst[:top_k]

        # user-global
        user_ctr = Counter()
        for (u, h), ctr in self.user_time_counts.items():
            if u == user:
                user_ctr.update(ctr)
        if user_ctr:
            total = sum(user_ctr.values())
            user_topk = [(pkg, cnt / total) for pkg, cnt in user_ctr.most_common()]
            return user_topk[:top_k]

        # global
        return self.global_topk[:top_k]

    def save(self, path: str) -> None:
        """
        Persist counters and Top-K lists to disk via pickle.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "user_time_counts": dict(self.user_time_counts),
                    "global_counter": self.global_counter,
                    "user_time_topk": self.user_time_topk,
                    "global_topk": self.global_topk,
                },
                f,
            )
        print(f"[BehaveTopKPredictor] saved to {path}")

    def load(self, path: str) -> None:
        """
        Load counters and Top-K lists from disk.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.user_time_counts = defaultdict(Counter, data["user_time_counts"])
        self.global_counter = data["global_counter"]
        self.user_time_topk = data["user_time_topk"]
        self.global_topk = data["global_topk"]
        print(f"[BehaveTopKPredictor] loaded from {path}")
