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
    在原有KMeans聚类流程的基础上，增加了可配置的特征工程模块：
    1. 特征行归一化（转换为占比）和标准化处理。
    2. PCA降维处理，保留指定比例的方差信息。
    3. 加入工作日/周末的日期标签。
    4. 加入时间三角函数特征。

    这些功能通过 __init__ 方法中的参数进行控制和启用。
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
        time_tri_anchors: tuple = (0, 4, 8, 12, 16, 20),  # 时间锚点（以小时计），默认每4小时设置一个，可自定义
        time_tri_width: float = 6.0,  # 三角基函数的半宽（以小时计），值越大曲线越平滑
    ):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta

        # 特征开关控制
        self.standardize = standardize
        self.use_pca = use_pca
        self.pca_var_ratio = pca_var_ratio
        self.weekend_tag = weekend_tag

        # 时间三角函数特征参数
        self.use_time_triangle = time_triangle
        self.time_tri_anchors = tuple(time_tri_anchors)
        self.time_tri_width = float(time_tri_width)

        # 存储不同时间窗口的特征缩放器（StandardScaler）
        self._scalers = {}
        # 存储不同时间窗口的降维模型（PCA）
        self._pcas = {}

        # 三个时间窗口的中心小时时刻
        self._window_centers = {0: 4.5, 1: 12.0, 2: 19.5}

    # --- 时间三角核函数工具方法 ---
    @staticmethod
    def _circ_dist_hour(h1: float, h2: float, period: float = 24.0) -> float:
        """计算24小时周期内的环形距离（以小时计）。"""
        d = abs(h1 - h2) % period
        return min(d, period - d)

    def _triangle_bump(self, hour: float, anchor: float, width: float) -> float:
        """
        三角基函数实现：当时间点与锚点距离小于半宽时，权重线性递减，超出半宽则权重为零。
        最高点（在锚点处）的权重为1，并向两侧线性下降至0。
        """
        d = self._circ_dist_hour(hour, anchor, 24.0)
        if d >= width:
            return 0.0
        return 1.0 - (d / width)

    def _build_features(self, df: pd.DataFrame, window_id: int) -> pd.DataFrame:
        """
        输入参数：DataFrame（索引为日期，列为各类别的总使用秒数，已去除 'off' 类别并填充空值0）。
        输出结果：可用于KMeans的二维特征矩阵，其索引与输入DataFrame的索引对齐。
        主要处理步骤：
          1. 根据索引日期加入工作日/周末的哑变量。
          2. 对特征进行行归一化（转为占比）和标准差归一化处理。
          3. 使用PCA保留指定比例的方差进行降维。
        返回结果：与原索引对齐的Numpy数组和索引。
        """
        X = df.copy()

        # 工作日/周末标签（基于索引日期）
        # 首先将日期索引转换为DatetimeIndex类型
        if self.weekend_tag:
            idx_dt = pd.to_datetime(X.index)
            is_weekend = (idx_dt.weekday >= 5).astype(float)  # 周六日为1，否则为0
            X = X.assign(__is_weekend__=is_weekend)

        # 时间三角特征：对当前“窗口中心时刻”进行三角基函数展开
        if self.use_time_triangle:
            center_hour = self._window_centers.get(window_id, 12.0)
            tri_cols = {}
            for a in self.time_tri_anchors:
                col = f"__ttri_{int(a)}__"
                tri_cols[col] = self._triangle_bump(center_hour, float(a), self.time_tri_width)
            # 当前时间窗口内的所有日期共享相同的三角时间编码值
            for col, val in tri_cols.items():
                X[col] = float(val)

        # 行归一化与标准化处理
        if self.standardize:
            # 行归一化（转换为占比），目的是消除不同类别数据量纲对距离计算的影响
            row_sum = X.sum(axis=1).replace(0, np.finfo(float).eps)
            X = (X.T / row_sum).T  # 特征向量L1范数归一化，得到各类别的占比

            # 标准化处理（零均值、单位方差），确保KMeans在各维度上的权重相对均衡
            scaler = StandardScaler()
            X_values = scaler.fit_transform(X.values)
            self._scalers[window_id] = scaler
        else:
            X_values = X.values
            self._scalers[window_id] = None

        # PCA 降维处理
        if self.use_pca:
            # PCA 保留指定的方差比例（例如0.95）
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
        # 计算用户在三个时间段（0-9点、9-15点、15-24点）内的行为向量
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
                print("时间无效")
                exit(1)

        daily_df = []
        # 构建时间段行为的 DataFrame 列表
        all_categories = set()
        for time_vector in daily_vectors:
            all_categories.update(cat for d in time_vector.values() for cat in d if (cat != "Unknown" and cat != "off"))
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
        # 特征工程与KMeans聚类流程主要在此处执行
        for window_id, df in enumerate(daily_df):
            df_filtered = df.drop(columns=["off"], errors="ignore").fillna(0)

            # 构建特征
            X_for_kmeans, idx = self._build_features(df_filtered, window_id)

            # KMeans 聚类
            kmeans = KMeans(n_clusters=min(self.n_clusters, len(df_filtered)), random_state=42, n_init="auto")
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
                avg_discharge_by_time_list.append((df_train[df_train["is_charging"] == False].groupby("time")["discharge"].mean()))

            valid_train_dates = [d for d in unique_dates if d != day]
            df_train = discharge_df[discharge_df["date"].isin(valid_train_dates)]
            alldays_avg_discharge_by_time = discharge_df[discharge_df["is_charging"] == False].groupby("time")["discharge"].mean()
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
                    battery_life_mins += self.vector_granularity * input / predicted_discharge[-1]
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
    """根据用户向量序列构造包含日期、时间、放电量和是否充电状态的DataFrame"""
    return pd.DataFrame(
        [(k.date(), k.time(), v[1], v[2]) for k, v in user_vector_sequence.items()],
        columns=["date", "time", "discharge", "is_charging"],
    )


class OracleTimeMedianPredictor(BasePredictor):
    """
    作为不进行行为聚类的理论性能上界预测器：
    对于每个时间槽，它使用所有天（包括预测当天）未充电时间段的放电量中位数进行预测。
    预测过程中不进行 alpha/beta 平滑处理，而是直接利用中位数进行逐步累加，这在评估平均绝对误差（MAE）时通常表现最优。
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
            flag = current_time.weekday() >= 5
            series = self.medians.get(flag, None)
            return float(series.get(current_time.time(), off_baseline)) if series is not None else off_baseline
        return float(self.medians.get(current_time.time(), off_baseline))

    def predict(self, start_time, last_hist_discharge, opcode, input, ratio=1) -> float:
        # opcode 1 -> battery life, opcode 2 -> discharge
        current_time = start_time
        discharge = 0.0
        life = 0.0
        while input > 0:
            step = self._lookup_slot(current_time)
            off_baseline = self.vector_granularity * 1.0
            if opcode == 1:  # 预测电池剩余寿命
                if input - step >= 0:
                    life += self.vector_granularity
                else:
                    life += self.vector_granularity * input / max(step, 1e-9)
                if step < off_baseline:
                    input -= step
                else:
                    input -= (step - off_baseline) * ratio + off_baseline
            elif opcode == 2:  # 预测剩余分钟数内的放电量
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
    作为基于KMeans聚类的理论性能上界预测器：
    它沿用原版的 0–9、9–15 和 15–24 三个时间段的聚类划分流程。
    但不同之处在于，在簇内计算放电量时，它使用“逐时刻”的中位数（包含预测当天的数据，这构成了一种理想化/信息泄漏的上界）。
    预测时同样不进行 alpha/beta 平滑，直接累加中位数。
    """

    def __init__(self, n_clusters: int = 10):
        self.n_clusters = n_clusters

    def fit(self, user_vector_sequence, vector_granularity) -> None:
        import datetime
        from collections import defaultdict
        from numvalkit.utils import dict_add
        from sklearn.cluster import KMeans

        self.vector_granularity = vector_granularity

        daily_vectors = [defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))]
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

        self.behavior_list = []
        for df in daily_df:
            df_f = df.drop(columns=["off"], errors="ignore").fillna(0)
            n_clusters = min(self.n_clusters, len(df_f)) if len(df_f) > 0 else 1
            labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(df_f) if len(df_f) > 0 else np.array([])
            df_f["cluster"] = labels
            df_sorted = df_f.sort_index()
            df_sorted.index = pd.to_datetime(df_sorted.index)
            self.behavior_list.append(df_sorted)

        discharge_df = _build_discharge_df(user_vector_sequence)
        discharge_df = discharge_df[discharge_df["is_charging"] == False]
        self.overall_median = discharge_df.groupby("time")["discharge"].median()

        self.cluster_time_medians = []
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
        # 0-9点使用总体中位数
        if t < datetime.time(9, 0, 0):
            return self.overall_median
        # 9-15点使用对应聚类簇的中位数
        elif datetime.time(9, 0, 0) <= t < datetime.time(15, 0, 0):
            df = self.behavior_list[1]
            if dstr in df.index:
                cid = df.loc[dstr, "cluster"]
                return self.cluster_time_medians[0].get(cid, self.overall_median)
            return self.overall_median
        # 15-24点使用对应聚类簇的中位数
        else:
            df = self.behavior_list[2]
            if dstr in df.index:
                cid = df.loc[dstr, "cluster"]
                return self.cluster_time_medians[1].get(cid, self.overall_median)
            return self.overall_median

    def predict(self, start_time, last_hist_discharge, opcode, input, ratio=1) -> float:
        # opcode 1 -> battery life, opcode 2 -> discharge
        current_time = start_time
        discharge = 0.0
        life = 0.0
        while input > 0:
            series = self._series_for(current_time)
            step = float(series.get(current_time.time(), self.vector_granularity * 1.0))
            off_baseline = self.vector_granularity * 1.0
            if opcode == 1:  # 预测电池剩余寿命
                if input - step >= 0:
                    life += self.vector_granularity
                else:
                    life += self.vector_granularity * input / max(step, 1e-9)
                if step < off_baseline:
                    input -= step
                else:
                    input -= (step - off_baseline) * ratio + off_baseline
            elif opcode == 2:  # 预测剩余分钟数内的放电量
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
