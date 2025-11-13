import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from numvalkit.utils import dict_add
from numvalkit.core import BasePredictor


class BatteryKmeansPredictorPlusHuber(BasePredictor):
    """
    在原有KMeans流程上，新增可开关的特征工程模块：
    (A) 行归一化(占比) + 标准化
    (B) PCA降维（保留指定方差比例）
    (D) 工作日/周末标签

    并在此基础上加入“自适应 a、b”（无偏置）：
    y(next) ≈ a * last_hist + b * hist_avg，按 0-9、9-15、15-24 三个时段桶分别学习。
    """

    def __init__(
        self,
        n_clusters,
        alpha,
        beta,
        standardize: bool = False,
        use_pca: bool = False,
        pca_var_ratio: float = 0.95,
        weekend_tag: bool = False,
        time_triangle: bool = True,
        time_tri_anchors: tuple = (0, 4, 8, 12, 16, 20),
        time_tri_width: float = 6.0,
    ):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta

        # feature toggles
        self.standardize = standardize
        self.use_pca = use_pca
        self.pca_var_ratio = pca_var_ratio
        self.weekend_tag = weekend_tag

        # 时间三角特征
        self.use_time_triangle = time_triangle
        self.time_tri_anchors = tuple(time_tri_anchors)
        self.time_tri_width = float(time_tri_width)

        # 为不同时窗保留特征变换器
        self._scalers = {}
        self._pcas = {}

        # 三个窗口的中心时刻（仅用于时间三角特征编码）
        self._window_centers = {0: 4.5, 1: 12.0, 2: 19.5}

        # ---- 自适应 a、b（按时段桶）----
        # {bucket(0/1/2): (a, b)}
        self.adaptive_ab_by_bucket = {}

    # --- 时间三角核工具 ---
    @staticmethod
    def _circ_dist_hour(h1: float, h2: float, period: float = 24.0) -> float:
        d = abs(h1 - h2) % period
        return min(d, period - d)

    def _triangle_bump(self, hour: float, anchor: float, width: float) -> float:
        d = self._circ_dist_hour(hour, anchor, 24.0)
        if d >= width:
            return 0.0
        return 1.0 - (d / width)

    def _build_features(self, df: pd.DataFrame, window_id: int) -> pd.DataFrame:
        X = df.copy()

        # (D) 工作日/周末标签
        if self.weekend_tag:
            idx_dt = pd.to_datetime(X.index)
            is_weekend = (idx_dt.weekday >= 5).astype(float)
            X = X.assign(__is_weekend__=is_weekend)

        # 时间三角特征（窗口级编码）
        if self.use_time_triangle:
            center_hour = self._window_centers.get(window_id, 12.0)
            tri_cols = {}
            for a in self.time_tri_anchors:
                col = f"__ttri_{int(a)}__"
                tri_cols[col] = self._triangle_bump(center_hour, float(a), self.time_tri_width)
            for col, val in tri_cols.items():
                X[col] = float(val)

        # (A) 行归一化 + 标准化
        if self.standardize:
            row_sum = X.sum(axis=1).replace(0, np.finfo(float).eps)
            X = (X.T / row_sum).T
            scaler = StandardScaler()
            X_values = scaler.fit_transform(X.values)
            self._scalers[window_id] = scaler
        else:
            X_values = X.values
            self._scalers[window_id] = None

        # (B) PCA
        if self.use_pca:
            pca = PCA(n_components=self.pca_var_ratio, svd_solver="full", random_state=42)
            X_pca = pca.fit_transform(X_values)
            self._pcas[window_id] = pca
            return X_pca, X.index
        else:
            self._pcas[window_id] = None
            return X_values, X.index

    # ---- 自适应 ab 所需的小工具 ----
    @staticmethod
    def _time_bucket(t: datetime.time) -> int:
        if t < datetime.time(9, 0, 0):
            return 0
        if t < datetime.time(15, 0, 0):
            return 1
        return 2

    def _get_hist_avg(self, day, time_obj):
        """读取当日对应时段的历史均值曲线；缺失则回退到 alldays，再回退到离屏耗电近似。"""
        per_bucket_avgs, alldays_avg = self.average_per_granularity_discharge_dict[day]
        if time_obj < datetime.time(9, 0, 0):
            series = alldays_avg  # 与原实现保持一致：0-9 用 alldays
        elif time_obj < datetime.time(15, 0, 0):
            series = per_bucket_avgs[0]
        else:
            series = per_bucket_avgs[1]
        if time_obj in series:
            return float(series[time_obj])
        if time_obj in alldays_avg:
            return float(alldays_avg[time_obj])
        return float(self.vector_granularity * 1.0)

    def fit(self, user_vector_sequence, vector_granularity) -> None:
        self.vector_granularity = vector_granularity
        # 0-9 / 9-15 / 15-24 向量聚合
        daily_vectors = [
            defaultdict(lambda: defaultdict(int)),
            defaultdict(lambda: defaultdict(int)),
            defaultdict(lambda: defaultdict(int)),
        ]
        for key, entry in user_vector_sequence.items():
            time_of_entry = key.time()
            date_of_entry = key.date()
            if time_of_entry < datetime.time(9, 0, 0):
                daily_vectors[0][date_of_entry] = dict_add(daily_vectors[0][date_of_entry], entry[0])
            elif datetime.time(9, 0, 0) <= time_of_entry < datetime.time(15, 0, 0):
                daily_vectors[1][date_of_entry] = dict_add(daily_vectors[1][date_of_entry], entry[0])
            elif datetime.time(15, 0, 0) <= time_of_entry:
                daily_vectors[2][date_of_entry] = dict_add(daily_vectors[2][date_of_entry], entry[0])
            else:
                raise ValueError("时间无效")

        daily_df = []
        all_categories = set()
        for time_vector in daily_vectors:
            all_categories.update(
                cat
                for d in time_vector.values()
                for cat in d
                if (cat != "Unknown" and cat != "off")
            )
            df = pd.DataFrame([{"date": date, **time_vector[date]} for date in sorted(time_vector)])
            if "date" in df.columns:
                df.set_index("date", inplace=True)
                df = df.fillna(0)
            daily_df.append(df)

        all_categories = sorted(all_categories)
        for i, df in enumerate(daily_df):
            for cat in all_categories:
                if cat not in df.columns:
                    df[cat] = 0
            daily_df[i] = df[all_categories].sort_index()

        behavior_list = []
        for window_id, df in enumerate(daily_df):
            df_filtered = df.drop(columns=["off"], errors="ignore").fillna(0)
            X_for_kmeans, idx = self._build_features(df_filtered, window_id)
            kmeans = KMeans(
                n_clusters=min(self.n_clusters, len(df_filtered)),
                random_state=42,
                n_init="auto"
            )
            labels = kmeans.fit_predict(X_for_kmeans)
            df_out = df_filtered.copy()
            df_out["cluster"] = pd.Series(labels, index=idx)
            df_sorted = df_out.sort_index()
            df_sorted.index = pd.to_datetime(df_sorted.index)
            behavior_list.append(df_sorted)

        discharge_df = pd.DataFrame(
            [(k.date(), k.time(), v[1], v[2]) for k, v in user_vector_sequence.items()],
            columns=["date", "time", "discharge", "is_charging"],
        )
        unique_dates = sorted({ts.date() for ts in user_vector_sequence.keys()})
        self.average_per_granularity_discharge_dict = {}
        for day in unique_dates:
            avg_discharge_by_time_list = []
            for behavior_df in behavior_list:
                key_str = pd.Timestamp(day).strftime("%Y-%m-%d")
                cluster_id = behavior_df.loc[key_str, "cluster"] if key_str in behavior_df.index else 0
                cluster_dates = behavior_df[behavior_df["cluster"] == cluster_id].index
                valid_train_dates = [d.date() for d in cluster_dates if d != day]
                df_train = discharge_df[discharge_df["date"].isin(valid_train_dates)]
                avg_discharge_by_time_list.append(
                    (df_train[df_train["is_charging"] == False].groupby("time")["discharge"].mean())
                )

            alldays_avg_discharge_by_time = (
                discharge_df[discharge_df["is_charging"] == False].groupby("time")["discharge"].mean()
            )
            self.average_per_granularity_discharge_dict[day] = (
                avg_discharge_by_time_list,
                alldays_avg_discharge_by_time,
            )

        # ---- 自适应 a、b 学习（无偏置）----
        # 训练样本：对于每个时间片 ts，预测下一片 ts+Δ
        if True:
            X_bucket = {0: [], 1: [], 2: []}  # [ [last_hist, hist_avg], ... ]
            y_bucket = {0: [], 1: [], 2: []}  # 下一片真实放电

            delta = pd.Timedelta(minutes=int(self.vector_granularity))
            keys = sorted(user_vector_sequence.keys())
            for ts in keys:
                nxt = ts + delta
                if nxt not in user_vector_sequence:
                    continue
                # 过滤充电：当前/下一片任一在充电则跳过（与评测集一致）
                if user_vector_sequence[ts][2] or user_vector_sequence[nxt][2]:
                    continue

                day = ts.date()
                tm = ts.time()
                bucket = self._time_bucket(tm)

                last_hist = float(user_vector_sequence[ts][1])  # 当前片真实放电（作为“上一片”能量）
                hist_avg = float(self._get_hist_avg(day, tm))   # 历史均值曲线
                y_next = float(user_vector_sequence[nxt][1])    # 下一片真实放电

                # 异常值保护
                if not np.isfinite(last_hist) or not np.isfinite(hist_avg) or not np.isfinite(y_next):
                    continue

                X_bucket[bucket].append([last_hist, hist_avg])
                y_bucket[bucket].append(y_next)

            # 对每个时段桶最小二乘拟合（样本不足回退到原 alpha/beta）
            self.adaptive_ab_by_bucket = {}
            for b in (0, 1, 2):
                Xb = np.asarray(X_bucket[b], dtype=float)
                yb = np.asarray(y_bucket[b], dtype=float)
                if Xb.shape[0] >= 10:  # 至少10个样本再学参数
                    # 最小二乘： y ~ [last_hist, hist_avg] @ [a, b]
                    try:
                        ab, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
                        a_learn, b_learn = float(ab[0]), float(ab[1])
                        # 合理性小约束（可视需要加更严格的裁剪）
                        if not np.isfinite(a_learn) or not np.isfinite(b_learn):
                            a_learn, b_learn = self.alpha, self.beta
                    except Exception:
                        a_learn, b_learn = self.alpha, self.beta
                else:
                    a_learn, b_learn = self.alpha, self.beta
                self.adaptive_ab_by_bucket[b] = (a_learn, b_learn)

    def predict(self, start_time, last_hist_discharge, opcode, input, ratio=1) -> float:
        # opcode 0 -> predict_battery_life, input = capacity
        # opcode 1 -> predict_discharge, input =  target_remaining_mins
        day = start_time.date()
        alldays_avg_discharge_by_time = self.average_per_granularity_discharge_dict[day][1]
        if start_time.time() < datetime.time(9, 0, 0):
            avg_discharge_by_time = self.average_per_granularity_discharge_dict[day][1]
        elif datetime.time(9, 0, 0) <= start_time.time() < datetime.time(15, 0, 0):
            avg_discharge_by_time = self.average_per_granularity_discharge_dict[day][0][0]
        elif datetime.time(15, 0, 0) <= start_time.time():
            avg_discharge_by_time = self.average_per_granularity_discharge_dict[day][0][1]
        else:
            raise ValueError("时间无效")

        current_time = start_time
        predicted_discharge = []
        discharge = 0.0
        battery_life_mins = 0.0

        while input > 0:
            off_screen_discharge = self.vector_granularity * 1.0
            if current_time.time() in avg_discharge_by_time:
                hist_avg = float(avg_discharge_by_time[current_time.time()])
            elif current_time.time() in alldays_avg_discharge_by_time:
                hist_avg = float(alldays_avg_discharge_by_time[current_time.time()])
            else:
                hist_avg = off_screen_discharge

            # 使用自适应 a、b（按时段桶）；若无则回退到 alpha、beta
            bucket = self._time_bucket(current_time.time())
            a, b = self.adaptive_ab_by_bucket.get(bucket, (self.alpha, self.beta))

            if len(predicted_discharge) > 0:
                last_hist_discharge = predicted_discharge[-1]

            next_pred = a * float(last_hist_discharge) + b * float(hist_avg)
            predicted_discharge.append(next_pred)

            if opcode == 1:  # predict_battery_life
                if input - predicted_discharge[-1] >= 0:
                    battery_life_mins += self.vector_granularity
                else:
                    battery_life_mins += (self.vector_granularity * input / predicted_discharge[-1])
                off_screen_discharge = self.vector_granularity * 1.0
                if predicted_discharge[-1] < off_screen_discharge:
                    input -= predicted_discharge[-1]
                else:
                    input -= (predicted_discharge[-1] - off_screen_discharge) * ratio + off_screen_discharge
            elif opcode == 2:  # predict_discharge
                if input - self.vector_granularity >= 0:
                    discharge += predicted_discharge[-1]
                else:
                    discharge += predicted_discharge[-1] * input / self.vector_granularity
                input -= self.vector_granularity
            else:
                assert 0
            current_time += pd.Timedelta(minutes=self.vector_granularity)

        if opcode == 1:
            return battery_life_mins
        elif opcode == 2:
            return discharge

    def predict_battery_life(self, start_time, last_hist_discharge, capacity, ratio) -> float:
        return self.predict(start_time, last_hist_discharge, 1, capacity, ratio)

    def predict_discharge(self, start_time, last_hist_discharge, target_remaining_mins) -> float:
        return self.predict(start_time, last_hist_discharge, 2, target_remaining_mins)
