import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from typing import List, Dict
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from joblib import Parallel, delayed
from sklearn.metrics import f1_score

from numvalkit.core import BasePredictor
from numvalkit.data_loader import ThermalSessionGenerator


class ThermalLGBMPredictor(BasePredictor):
    def __init__(self, data_dir: str, user_list: List[str]):
        self.data_dir = data_dir
        self.user_list = user_list
        self.generator = ThermalSessionGenerator(data_dir)
        self.model = None
        self.features = []

    @staticmethod
    def _prepare_features(sessions: List[dict]) -> pd.DataFrame:
        records = []
        for sess in sessions:
            row = {
                "user": sess["user"],
                "start_hour": sess["start_hour"],
                "duration_min": sess["duration_min"],
                "temp_raise": sess["temp_raise"],
                "battery_discharge": sess["battery_discharge"],
                "start_temp": sess["start_temp"],
                "is_charging": sess["is_charging"],
                "high_temp": sess["high_temp"],  # label
            }
            row.update(sess["vector"])
            records.append(row)
        return pd.DataFrame(records)

    def fit(self, users: List[str] = None, verbose: bool = False) -> None:
        user_list = users if users else self.user_list

        def load_user_data(user):
            try:
                return self.generator.generate(user) or []
            except Exception as e:
                print(f"[Error] loading {user}: {e}")
                return []

        all_sessions = Parallel(n_jobs=-1)(
            delayed(load_user_data)(user)
            for user in tqdm(user_list, desc="Load training set")
        )
        flat_sessions = [s for user_sessions in all_sessions for s in user_sessions]
        if not flat_sessions:
            raise ValueError("No usable sessions found.")

        df = self._prepare_features(flat_sessions)
        df.fillna(0, inplace=True)
        self.features = [col for col in df.columns if col not in ["user", "high_temp"]]

        X = df[self.features]
        y = df["high_temp"]

        self.model = LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
            learning_rate=0.05,
            num_leaves=31,
            n_estimators=200,
            n_jobs=-1,
            verbose=-1,
        )

        self.model.fit(X, y)
        if verbose:
            print(classification_report(y, self.model.predict(X)))

        # Find best threshold
        y_prob = self.model.predict_proba(X)[:, 1]
        best_f1 = -1
        best_threshold = 0.5
        for t in np.linspace(0.1, 0.9, 81):
            y_pred = (y_prob > t).astype(int)
            f1 = f1_score(y, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        self.threshold = best_threshold
        if verbose:
            print(
                f"[Threshold] Best threshold found: {self.threshold:.3f} (F1={best_f1:.4f})"
            )

    def predict(self, user: str, session: dict) -> float:
        df = self._prepare_features([session])
        df.fillna(0, inplace=True)
        df = df.reindex(columns=self.features, fill_value=0)
        return float(self.model.predict_proba(df)[0, 1])

    def predict_batch(self, sessions: List[dict]) -> np.ndarray:
        if not sessions:
            return np.array([])
        df = self._prepare_features(sessions)
        df.fillna(0, inplace=True)
        df = df.reindex(columns=self.features, fill_value=0)
        return self.model.predict_proba(df)[:, 1]

    def save(self, path: str) -> None:
        joblib.dump((self.model, self.features), path)

    def load(self, path: str) -> None:
        self.model, self.features = joblib.load(path)
