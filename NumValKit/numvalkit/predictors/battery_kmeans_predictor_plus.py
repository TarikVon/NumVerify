import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from numvalkit.utils import dict_add
from numvalkit.core import BasePredictor


class BatteryKmeansPredictorPlus(BasePredictor):
    """
    在原有KMeans流程上，新增可开关的特征工程模块：
    (A) 行归一化(占比) + 标准化
    (B) PCA降维（保留指定方差比例）
    (D) 工作日/周末标签

    通过 __init__ 的开关控制是否启用
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
        time_tri_anchors: tuple = (0, 4, 8, 12, 16, 20),  # 锚点（小时），可改：每4小时一个
        time_tri_width: float = 6.0,                      # 半宽（小时），越大越平滑
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

        # 三个窗口的中心时刻
        self._window_centers = {0: 4.5, 1: 12.0, 2: 19.5}

    # --- 时间三角核工具 ---
    @staticmethod
    def _circ_dist_hour(h1: float, h2: float, period: float = 24.0) -> float:
        """24h 周期环形距离（小时）。"""
        d = abs(h1 - h2) % period
        return min(d, period - d)

    def _triangle_bump(self, hour: float, anchor: float, width: float) -> float:
        """
        三角基函数：距 anchor 小于 width 线性递减，超过则为0。
        最高点=1，线性下降到0。
        """
        d = self._circ_dist_hour(hour, anchor, 24.0)
        if d >= width:
            return 0.0
        # 线性形状：距锚点越远权重越小
        return 1.0 - (d / width)

    def _build_features(self, df: pd.DataFrame, window_id: int) -> pd.DataFrame:
        """
        输入：df（索引为date，列为类别使用秒数，已drop 'off' 且 fillna(0)）
        输出：可用于KMeans的二维特征矩阵（与索引对齐）
        步骤：
          (D) 加入weekend哑变量
          (A) 行归一化(把每行转为占比) + StandardScaler
          (B) PCA保留指定方差比例
        返回：和 index 对齐的 numpy 数组
        """
        X = df.copy()

        # (D) 工作日/周末标签（按索引date）
        # 先把时间索引转为DatetimeIndex
        if self.weekend_tag:
            idx_dt = pd.to_datetime(X.index)
            is_weekend = (idx_dt.weekday >= 5).astype(float)  # 周六日=1，否则=0
            X = X.assign(__is_weekend__=is_weekend)

        # (+) 时间三角特征：对“窗口中心时刻”做三角基展开
        if self.use_time_triangle:
            center_hour = self._window_centers.get(window_id, 12.0)
            tri_cols = {}
            for a in self.time_tri_anchors:
                col = f"__ttri_{int(a)}__"
                tri_cols[col] = self._triangle_bump(center_hour, float(a), self.time_tri_width)
            # 所有日期共享当前窗口的同一时间编码（因为此窗口即代表该时段）
            for col, val in tri_cols.items():
                X[col] = float(val)

        # (A) 行归一化 + 标准化
        if self.standardize:
            # 行归一化（占比），避免不同类别量纲影响距离
            row_sum = X.sum(axis=1).replace(0, np.finfo(float).eps)
            X = (X.T / row_sum).T  # L1到1，得到各类别占比

            # StandardScaler（零均值单位方差），有助于KMeans在各维上权重均衡
            scaler = StandardScaler()
            X_values = scaler.fit_transform(X.values)
            self._scalers[window_id] = scaler
        else:
            X_values = X.values
            self._scalers[window_id] = None

        # (B) PCA
        if self.use_pca:
            # 保留指定的方差比例（如0.95）
            pca = PCA(n_components=self.pca_var_ratio, svd_solver="full", random_state=42)
            X_pca = pca.fit_transform(X_values)
            self._pcas[window_id] = pca
            return X_pca, X.index
        else:
            self._pcas[window_id] = None
            return X_values, X.index

        # 

    def fit(self, user_vector_sequence, vector_granularity) -> None:
        self.vector_granularity = vector_granularity
        # 算出用户的0-9，9-15,15-24向量
        daily_vectors = [
            defaultdict(lambda: defaultdict(int)),
            defaultdict(lambda: defaultdict(int)),
            defaultdict(lambda: defaultdict(int)),
        ]
        for key, entry in user_vector_sequence.items():
            time_of_entry = key.time()
            date_of_entry = key.date()

            if time_of_entry < datetime.time(9, 0, 0):
                daily_vectors[0][date_of_entry] = dict_add(
                    daily_vectors[0][date_of_entry], entry[0]
                )
            elif datetime.time(9, 0, 0) <= time_of_entry < datetime.time(15, 0, 0):
                daily_vectors[1][date_of_entry] = dict_add(
                    daily_vectors[1][date_of_entry], entry[0]
                )
            elif datetime.time(15, 0, 0) <= time_of_entry:
                daily_vectors[2][date_of_entry] = dict_add(
                    daily_vectors[2][date_of_entry], entry[0]
                )
            else:
                print("时间无效")
                exit(1)

        daily_df = []
        # 构建 DataFrame
        all_categories = set()
        for time_vector in daily_vectors:
            all_categories.update(
                cat
                for d in time_vector.values()
                for cat in d
                if (cat != "Unknown" and cat != "off")
            )
            df = pd.DataFrame(
                [{"date": date, **time_vector[date]} for date in sorted(time_vector)]
            )
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
        # === 仅在这里插入特征工程与KMeans，尽量不动外部逻辑 ===
        for window_id, df in enumerate(daily_df):
            df_filtered = df.drop(columns=["off"], errors="ignore").fillna(0)

            # 构建特征
            X_for_kmeans, idx = self._build_features(df_filtered, window_id)

            # KMeans
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
                cluster_id = (
                    behavior_df.loc[key_str, "cluster"]
                    if key_str in behavior_df.index
                    else 0
                )
                cluster_dates = behavior_df[behavior_df["cluster"] == cluster_id].index
                valid_train_dates = [d.date() for d in cluster_dates if d != day]
                df_train = discharge_df[discharge_df["date"].isin(valid_train_dates)]
                avg_discharge_by_time_list.append(
                    (
                        df_train[df_train["is_charging"] == False]
                        .groupby("time")["discharge"]
                        .mean()
                    )
                )

            valid_train_dates = [d for d in unique_dates if d != day]
            df_train = discharge_df[discharge_df["date"].isin(valid_train_dates)]
            alldays_avg_discharge_by_time = (
                discharge_df[discharge_df["is_charging"] == False]
                .groupby("time")["discharge"]
                .mean()
            )
            self.average_per_granularity_discharge_dict[day] = (
                avg_discharge_by_time_list,
                alldays_avg_discharge_by_time,
            )

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
            print("时间无效")
            exit(1)

        current_time = start_time
        predicted_discharge = []
        discharge = 0
        battery_life_mins = 0
        while input > 0:
            off_screen_discharge = self.vector_granularity * 1
            if current_time.time() in avg_discharge_by_time:
                hist_avg = avg_discharge_by_time[current_time.time()]
            elif current_time.time() in alldays_avg_discharge_by_time:
                hist_avg = alldays_avg_discharge_by_time[current_time.time()]
            else:
                hist_avg = off_screen_discharge

            if len(predicted_discharge) > 0:
                last_hist_discharge = predicted_discharge[-1]
            predicted_discharge.append(self.alpha * last_hist_discharge + self.beta * hist_avg)

            if opcode == 1:
                if input - predicted_discharge[-1] >= 0:
                    battery_life_mins += self.vector_granularity
                else:
                    battery_life_mins += (self.vector_granularity * input / predicted_discharge[-1])
                off_screen_discharge = self.vector_granularity * 1
                if predicted_discharge[-1] < off_screen_discharge:
                    input -= predicted_discharge[-1]
                else:
                    input -= (predicted_discharge[-1] - off_screen_discharge) * ratio + off_screen_discharge
            elif opcode == 2:
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

def _build_discharge_df(user_vector_sequence):
    """构造 (date, time, discharge, is_charging) 表"""
    return pd.DataFrame(
        [(k.date(), k.time(), v[1], v[2]) for k, v in user_vector_sequence.items()],
        columns=["date", "time", "discharge", "is_charging"],
    )

class OracleTimeMedianPredictor(BasePredictor):
    """
    理论上界（不分聚类）：每个 time slot 使用“所有天（含当天）未充电片段”的放电中位数。
    预测阶段不做 alpha/beta 平滑，直接用中位数逐步累加——对 MAE 最优。
    """
    def __init__(self, use_weekday_weekend: bool = False):
        self.use_weekday_weekend = use_weekday_weekend

    def fit(self, user_vector_sequence, vector_granularity) -> None:
        self.vector_granularity = vector_granularity
        df = _build_discharge_df(user_vector_sequence)
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["is_charging"] == False]

        if self.use_weekday_weekend:
            df["is_weekend"] = df["date"].dt.weekday >= 5
            self.medians = {}
            for flag, sub in df.groupby("is_weekend"):
                self.medians[flag] = sub.groupby("time")["discharge"].median()
        else:
            self.medians = df.groupby("time")["discharge"].median()

    def _lookup_slot(self, current_time: pd.Timestamp) -> float:
        off_baseline = self.vector_granularity * 1.0
        if self.use_weekday_weekend:
            flag = (current_time.weekday() >= 5)
            series = self.medians.get(flag, None)
            return float(series.get(current_time.time(), off_baseline)) if series is not None else off_baseline
        return float(self.medians.get(current_time.time(), off_baseline))

    def predict(self, start_time, last_hist_discharge, opcode, input, ratio=1) -> float:
        current_time = start_time
        discharge = 0.0
        life = 0.0
        while input > 0:
            step = self._lookup_slot(current_time)
            off_baseline = self.vector_granularity * 1.0
            if opcode == 1:  # battery life
                if input - step >= 0:
                    life += self.vector_granularity
                else:
                    life += self.vector_granularity * input / max(step, 1e-9)
                if step < off_baseline:
                    input -= step
                else:
                    input -= (step - off_baseline) * ratio + off_baseline
            elif opcode == 2:  # discharge in remaining minutes
                step_minutes = min(self.vector_granularity, input)
                discharge += step * step_minutes / self.vector_granularity
                input -= step_minutes
            else:
                assert 0
            current_time += pd.Timedelta(minutes=self.vector_granularity)
        return life if opcode == 1 else discharge

    def predict_battery_life(self, start_time, last_hist_discharge, capacity, ratio) -> float:
        return self.predict(start_time, last_hist_discharge, 1, capacity, ratio)

    def predict_discharge(self, start_time, last_hist_discharge, target_remaining_mins) -> float:
        return self.predict(start_time, last_hist_discharge, 2, target_remaining_mins)


class OracleKMeansMedianPredictor(BasePredictor):
    """
    理论上界（按KMeans分桶）：复用原 0–9 / 9–15 / 15–24 聚类流程，
    但簇内“逐时刻”用中位数（含当天，信息泄漏），预测时直接累加中位数（不做 alpha/beta）。
    """
    def __init__(self, n_clusters: int = 10):
        self.n_clusters = n_clusters

    def fit(self, user_vector_sequence, vector_granularity) -> None:
        import datetime
        from collections import defaultdict
        from numvalkit.utils import dict_add
        from sklearn.cluster import KMeans

        self.vector_granularity = vector_granularity

        # 1) 与现有流程一致：把行为聚合到三段（不涉及特征工程）
        daily_vectors = [defaultdict(lambda: defaultdict(int)),
                         defaultdict(lambda: defaultdict(int)),
                         defaultdict(lambda: defaultdict(int))]
        for ts, entry in user_vector_sequence.items():
            t, d = ts.time(), ts.date()
            if t < datetime.time(9, 0, 0):
                daily_vectors[0][d] = dict_add(daily_vectors[0][d], entry[0])
            elif datetime.time(9, 0, 0) <= t < datetime.time(15, 0, 0):
                daily_vectors[1][d] = dict_add(daily_vectors[1][d], entry[0])
            else:
                daily_vectors[2][d] = dict_add(daily_vectors[2][d], entry[0])

        daily_df, all_categories = [], set()
        for tv in daily_vectors:
            all_categories.update(cat for d in tv.values() for cat in d if (cat != "Unknown" and cat != "off"))
            df = pd.DataFrame([{"date": date, **tv[date]} for date in sorted(tv)])
            if "date" in df.columns:
                df.set_index("date", inplace=True)
                df = df.fillna(0)
            daily_df.append(df)

        all_categories = sorted(all_categories)
        for i, df in enumerate(daily_df):
            for c in all_categories:
                if c not in df.columns:
                    df[c] = 0
            daily_df[i] = df[all_categories].sort_index()

        # 2) KMeans 打标签（与原版一致）
        self.behavior_list = []
        for df in daily_df:
            df_f = df.drop(columns=["off"], errors="ignore").fillna(0)
            n_clusters = min(self.n_clusters, len(df_f)) if len(df_f) > 0 else 1
            labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(df_f) if len(df_f) > 0 else np.array([])
            df_f["cluster"] = labels
            df_sorted = df_f.sort_index()
            df_sorted.index = pd.to_datetime(df_sorted.index)
            self.behavior_list.append(df_sorted)

        # 3) 构造 overall 与 “按簇”逐时刻中位数（含当天 → 理想上界）
        discharge_df = _build_discharge_df(user_vector_sequence)
        discharge_df = discharge_df[discharge_df["is_charging"] == False]
        self.overall_median = discharge_df.groupby("time")["discharge"].median()

        self.cluster_time_medians = []  # 对应 9–15 和 15–24
        for df_clusters in self.behavior_list[1:3]:
            cmap = {}
            for cid, days in df_clusters.groupby("cluster"):
                day_index = set(days.index.date)
                sub = discharge_df[discharge_df["date"].isin(day_index)]
                cmap[cid] = sub.groupby("time")["discharge"].median()
            self.cluster_time_medians.append(cmap)

    def _series_for(self, current_time: pd.Timestamp):
        import datetime
        dstr = pd.Timestamp(current_time.date()).strftime("%Y-%m-%d")
        t = current_time.time()
        if t < datetime.time(9, 0, 0):
            return self.overall_median
        elif datetime.time(9, 0, 0) <= t < datetime.time(15, 0, 0):
            df = self.behavior_list[1]
            if dstr in df.index:
                cid = df.loc[dstr, "cluster"]
                return self.cluster_time_medians[0].get(cid, self.overall_median)
            return self.overall_median
        else:
            df = self.behavior_list[2]
            if dstr in df.index:
                cid = df.loc[dstr, "cluster"]
                return self.cluster_time_medians[1].get(cid, self.overall_median)
            return self.overall_median

    def predict(self, start_time, last_hist_discharge, opcode, input, ratio=1) -> float:
        current_time = start_time
        discharge = 0.0
        life = 0.0
        while input > 0:
            series = self._series_for(current_time)
            step = float(series.get(current_time.time(), self.vector_granularity * 1.0))
            off_baseline = self.vector_granularity * 1.0
            if opcode == 1:
                if input - step >= 0:
                    life += self.vector_granularity
                else:
                    life += self.vector_granularity * input / max(step, 1e-9)
                if step < off_baseline:
                    input -= step
                else:
                    input -= (step - off_baseline) * ratio + off_baseline
            elif opcode == 2:
                step_minutes = min(self.vector_granularity, input)
                discharge += step * step_minutes / self.vector_granularity
                input -= step_minutes
            else:
                assert 0
            current_time += pd.Timedelta(minutes=self.vector_granularity)
        return life if opcode == 1 else discharge

    def predict_battery_life(self, start_time, last_hist_discharge, capacity, ratio) -> float:
        return self.predict(start_time, last_hist_discharge, 1, capacity, ratio)

    def predict_discharge(self, start_time, last_hist_discharge, target_remaining_mins) -> float:
        return self.predict(start_time, last_hist_discharge, 2, target_remaining_mins)
