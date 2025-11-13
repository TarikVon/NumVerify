import math
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from numvalkit.core import BasePredictor

class BatteryDTPredictor(BasePredictor):
    """
    决策树逐片回归 + 自回归滚动（按日留一评测）
    - 特征： [上一片放电量, 同一时刻(时间槽)历史均值(排除被评当天)]
    - 目标： 当前片(=vector_granularity 分钟)放电量
    - 预测： 按片滚动，合计得到目标区间总放电量
    """

    def __init__(self, max_depth=None, min_samples_leaf=5, random_state=42):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.vector_granularity = None

        # 每个 day 的模型与统计缓存
        self.model_per_day = {}              # day -> DecisionTreeRegressor
        self.slot_mean_excl_day = {}         # day -> {slot: mean_discharge}
        self.global_mean_excl_day = {}       # day -> float (训练集整体均值，做兜底)

    # ---- 工具 ----
    def _day_of(self, ts: pd.Timestamp):
        # 返回日期对象（不带时分秒）
        return ts.normalize().date()

    def _slot_of(self, ts: pd.Timestamp):
        # 把一天切成 1440 / granularity 个时间槽（与评测粒度一致）
        tmin = ts.hour * 60 + ts.minute
        return int(tmin // self.vector_granularity)

    # ---- 训练（构造 per-day 留一模型与统计）----
    def fit(self, user_vector_sequence, vector_granularity) -> None:
        """
        user_vector_sequence: {Timestamp -> [usage_dict, discharge(float), is_charging(bool)]}
        """
        self.vector_granularity = int(vector_granularity)

        # 1) 整理为样本表（仅非充电片，且上一片也需是非充电）
        records = []
        keys_sorted = sorted(user_vector_sequence.keys())
        for ts in keys_sorted:
            cur = user_vector_sequence[ts]
            if cur[2]:  # 当前片在充电 -> 不参与
                continue
            prev_ts = (ts - pd.Timedelta(minutes=self.vector_granularity)).round("min")
            if prev_ts not in user_vector_sequence:
                continue
            prev = user_vector_sequence[prev_ts]
            if prev[2]:  # 上一片在充电 -> 自回归基准不干净，跳过
                continue

            records.append({
                "day": self._day_of(ts),
                "slot": self._slot_of(ts),
                "last": float(prev[1]),
                "y": float(cur[1]),
            })

        if not records:
            # 无可用训练样本
            self.model_per_day.clear()
            self.slot_mean_excl_day.clear()
            self.global_mean_excl_day.clear()
            return

        df = pd.DataFrame.from_records(records)
        all_days = sorted(df["day"].unique().tolist())

        # 2) 为每个待评日期 day 训练“排除当天”的模型，并计算同一时刻历史均值
        for day in all_days:
            df_train = df[df["day"] != day]
            if len(df_train) < 2:
                # 训练数据太少，记为不可用；预测阶段将返回 -1 跳过
                self.model_per_day[day] = None
                self.slot_mean_excl_day[day] = {}
                self.global_mean_excl_day[day] = 0.0
                continue

            # (a) 计算“同一时刻历史均值”：仅基于 days != day
            slot_mean = df_train.groupby("slot")["y"].mean().to_dict()
            global_mean = float(df_train["y"].mean())
            self.slot_mean_excl_day[day] = slot_mean
            self.global_mean_excl_day[day] = global_mean

            # (b) 构造训练特征与标签
            # 特征 = [last, slot_mean[slot] (若缺则用 global_mean)]
            sm_feat = df_train["slot"].map(lambda s: slot_mean.get(int(s), global_mean)).to_numpy()
            X = np.column_stack([df_train["last"].to_numpy(dtype=float), sm_feat])
            y = df_train["y"].to_numpy(dtype=float)

            # (c) 训练决策树
            model = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
            model.fit(X, y)
            self.model_per_day[day] = model

    # ---- 统一入口：满足 BasePredictor 抽象方法 ----
    def predict(self, start_time, last_hist_discharge, opcode, input, ratio=1) -> float:
        """
        opcode=2 -> 预测放电总量，input=target_remaining_mins
        opcode=1 -> 预测续航分钟数（可选，当前评测未用；这里给出合理实现）
        """
        if self.vector_granularity is None:
            return -1.0

        day = self._day_of(start_time)
        model = self.model_per_day.get(day, None)
        slot_mean = self.slot_mean_excl_day.get(day, {})
        global_mean = self.global_mean_excl_day.get(day, 0.0)

        if model is None:
            # 无可用模型（如只有单日数据）-> 按评测约定跳过该样本
            return -1.0

        cur_time = start_time
        last = float(last_hist_discharge)

        if opcode == 2:
            # 目标区间总放电量
            rem = int(input)
            total = 0.0
            while rem > 0:
                step = min(self.vector_granularity, rem)
                slot = self._slot_of(cur_time)
                sm = float(slot_mean.get(slot, global_mean))

                # 预测整片放电
                pred_full = float(model.predict([[last, sm]])[0])
                pred_full = max(0.0, pred_full)  # 非负约束

                if step == self.vector_granularity:
                    total += pred_full
                    last = pred_full
                else:
                    total += pred_full * (step / self.vector_granularity)

                rem -= step
                cur_time += pd.Timedelta(minutes=step)
            return float(total)

        elif opcode == 1:
            # （备用）续航分钟数估计：把“剩余电量”按片扣减
            cap = float(input)
            life = 0.0
            while cap > 0:
                slot = self._slot_of(cur_time)
                sm = float(slot_mean.get(slot, global_mean))
                pred_full = float(model.predict([[last, sm]])[0])
                pred_full = max(0.0, pred_full)

                # 简单处理：把整片耗电按 ratio 分摊；避免 0 造成死循环
                effective = max(pred_full * float(ratio), 1e-6)

                if cap - effective >= 0:
                    life += self.vector_granularity
                    last = pred_full
                    cap -= effective
                    cur_time += pd.Timedelta(minutes=self.vector_granularity)
                else:
                    life += self.vector_granularity * (cap / effective)
                    break
            return float(life)

        else:
            return -1.0

    # ---- 评测使用的包装 ----
    def predict_discharge(self, start_time, last_hist_discharge, target_remaining_mins) -> float:
        return self.predict(start_time, last_hist_discharge, 2, target_remaining_mins)
