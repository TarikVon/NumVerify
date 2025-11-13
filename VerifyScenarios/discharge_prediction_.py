import os
import datetime
import pickle
from collections import defaultdict
import pandas as pd
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from utils import weighted_average_2d, draw_error_abs_cdf, draw_error_matrix_figure, get_productname_from_user
from config import *
from numvalkit.predictors import *
from numvalkit.data_loader import BehaveVectorGenerator

# 开始时间：[0,24]
# 预测时间：现在包括 [12, 9, 6, 3, 1]
# 要求：开始时间之前有5个向量（seq_len），预测时间期间没有充电

start_time_list = range(0, 24)
interval_list = [12, 9, 6, 3, 1]

# 按区间切换时间片长度：
# 12/9/6 小时 → 90min；3/1 小时 → 60min
interval_granularity_map = {
    12: 90,
    9: 90,
    6: 90,
    3: 60,
    1: 60,
}

# 历史窗口长度（向后取 seq_len * granularity 的历史片段）
seq_len = 5

# 进程池
max_workers = 112
verbose = False

result_dir = "./results/"

class DatasetGenerator:
    def __init__(self, data_dir: str = "./data"):
        """
        :param data_dir: Root directory where user data folders are located.
        """
        self.data_dir = data_dir

    # 产生用于训练和验证的向量序列
    def generate_dataset(
        self,
        user,
        user_vector_sequence,
        start_time_list,
        interval_list,          # 允许传入单个或多个 interval
        seq_len,
        vector_granularity,      # 本次数据集所用的时间片长度
        reflush,
    ):
        path = os.path.join(self.data_dir, user)
        if not os.path.exists(path):
            return None

        # ---- 缓存文件名包含 granularity / seq_len / interval 组合，避免覆盖或读错 ----
        interval_tag = "_".join(map(str, interval_list)) + "h"
        dataset_path = f"{path}/dataset_{vector_granularity}_{seq_len}_{interval_tag}"

        if os.path.exists(dataset_path) and (not reflush):
            if verbose:
                print(f"read dataset of granularity {vector_granularity} from {dataset_path}")
            with open(dataset_path, "rb") as f:
                return pickle.load(f)
        else:
            if verbose:
                print(
                    f"compute dataset of granularity {vector_granularity}, seq_len {seq_len} for {user} ({interval_tag})"
                )

        # dict{(start_time, interval): [(day, [vectors], discharge)]}
        val_dict = defaultdict(list)

        unique_dates = sorted({ts.date() for ts in user_vector_sequence.keys()})
        for start_time in start_time_list:
            for interval in interval_list:
                entry_list = []
                for day in unique_dates:
                    start_date_time = pd.Timestamp(
                        datetime.datetime.combine(
                            day, datetime.time(hour=start_time, minute=0)
                        )
                    )
                    # 收集历史 seq_len 个时间片
                    vector_list = []
                    vector_missing = False
                    for delta in range(-vector_granularity * seq_len, 0, vector_granularity):
                        time = (start_date_time + pd.Timedelta(minutes=delta)).round("min")
                        if time in user_vector_sequence:
                            vector_list.append(user_vector_sequence[time])
                        else:
                            vector_missing = True
                            break
                    if vector_missing:
                        continue

                    # 目标预测区间内累计耗电，且必须不充电
                    discharge = 0
                    need_skipping = False
                    for delta in range(0, interval * 60, vector_granularity):
                        time = (start_date_time + pd.Timedelta(minutes=delta)).round("min")
                        if time in user_vector_sequence:
                            discharge += user_vector_sequence[time][1]
                            if user_vector_sequence[time][2]:
                                need_skipping = True
                                break
                        else:
                            need_skipping = True
                            break
                    if need_skipping:
                        continue
                    entry_list.append((day, vector_list, discharge))
                val_dict[(start_time, interval)] = entry_list

        with open(dataset_path, "wb") as f:
            pickle.dump(val_dict, f)
            if verbose:
                print(
                    f"store dataset of granularity {vector_granularity}, seq_len {seq_len} for {user} at {dataset_path}"
                )
        return val_dict


def make_predictor(method_name, user=None):
    """小工厂：根据方法名构造预测器实例"""
    if method_name == "kmeans":
        return BatteryKmeansPredictor(10, 0.2, 0.8)
    elif method_name == "kmeans_plus":
        return BatteryKmeansPredictorPlus(
            10,
            0.2,
            0.8,
            standardize=True,
            use_pca=True,
            weekend_tag=True,
            time_triangle=True
        )
    elif method_name == "kmeans_plus_huber":
        return BatteryKmeansPredictorPlusHuber(n_clusters=10, alpha=0.2, beta=0.8, standardize=True, use_pca=True, weekend_tag=True, time_triangle=True)
    elif method_name == "baseline":
        return BatteryStatisticalPredictor()
    elif method_name == "aosp":
        return BatteryAospPredictor()
    elif method_name == "ae":
        return BatteryAEPredictor(
            n_clusters=10,
            alpha=0.2,
            beta=0.8,
            z_dim=16,
            ae_epochs=100,
            ae_batch_size=256,
            ae_lr=1e-3,
            user_id=user
        )
    elif method_name == "dt":
        return BatteryDTPredictor()
    else:
        raise ValueError(f"Unknown method: {method_name}")


def predictUserDischarge(user, method_name):
    n_intervals = len(interval_list)

    error_mat = np.zeros((n_intervals, 24))
    weight_mat = np.zeros((n_intervals, 24))
    error_abs_list = [[] for _ in range(len(interval_list))]
    # battery_capacity = get_productname_from_user(user)
    battery_capacity = 5500

    behaveVectorGenerator = BehaveVectorGenerator(DATA_DIR)

    # ---- 预生成两套（60/90min）行为向量序列，供不同 interval 复用 ----
    needed_granularities = sorted(set(interval_granularity_map[i] for i in interval_list))
    user_vector_seq_by_g = {}
    for g in needed_granularities:
        # use_appname=False 与原逻辑一致；是否刷新由 config.regenerate_cached_data 控制
        user_vector_seq_by_g[g] = behaveVectorGenerator.generate(
            user, g, False, regenerate_cached_data
        )

    datasetGenerator = DatasetGenerator(DATA_DIR)

    # ---- 为每个 interval 按对应的 granularity 单独生成/读取数据集，并汇总到一个 val_dict ----
    val_dict = {}
    for inter in interval_list:
        granularity = interval_granularity_map[inter]
        uvs = user_vector_seq_by_g[granularity]
        if uvs is None:
            # 没有该粒度的向量，整个 interval 下的数据不可用
            for st in start_time_list:
                val_dict[(st, inter)] = []
            continue
        # 传入单元素 interval 列表，缓存名会标注成 ..._{inter}h
        val_dict_interval = datasetGenerator.generate_dataset(
            user=user,
            user_vector_sequence=uvs,
            start_time_list=start_time_list,
            interval_list=[inter],
            seq_len=seq_len,
            vector_granularity=granularity,
            reflush=regenerate_cached_data,
        )
        if val_dict_interval is not None:
            val_dict.update(val_dict_interval)
        else:
            for st in start_time_list:
                val_dict[(st, inter)] = []

    def safe_fit(predictor, *args):
        try:
            predictor.fit(*args)
            return predictor, None
        except Exception as e:
            detacted_abnormal_user_in_runtime.add(user)
            return None, e

    # ---- 按“粒度”缓存已拟合的模型（同一粒度只 fit 一次）----
    fitted_by_g = {}
    for g in needed_granularities:
        predictor = make_predictor(method_name, user=user)

        if method_name in ["kmeans", "kmeans_plus", "baseline", "ae", "dt", "kmeans_plus_huber"]:
            uvs = user_vector_seq_by_g[g]
            if uvs is None:
                fitted_by_g[g] = None
                continue
            predictor, err = safe_fit(predictor, uvs, g)
        elif method_name == "aosp":
            predictor, err = safe_fit(predictor, f"{DATA_DIR}/{user}/battery_data.csv", g, battery_capacity)
        else:
            raise ValueError(method_name)

        if predictor is None:
            fitted_by_g[g] = None
        else:
            fitted_by_g[g] = predictor

    # ---- 误差计算：按 interval 取对应粒度的模型进行预测 ----
    for start_time in start_time_list:
        for idx, interval in enumerate(interval_list):
            granularity = interval_granularity_map[interval]
            entry_list = val_dict.get((start_time, interval), [])
            batteryPredictor = fitted_by_g.get(granularity, None)

            if batteryPredictor is None:
                error_mat[idx][start_time] = float("nan")
                weight_mat[idx][start_time] = 0
                continue

            predicted_sum = 0
            real_sum = 0
            valid_count = 0

            for entry in entry_list:
                real_discharge = entry[2]
                day = entry[0]
                start_predict_time = pd.Timestamp(day).replace(hour=start_time, minute=0)
                last_hist_discharge = entry[1][-1][1]
                # 如果最后一个历史时间片处于充电状态，跳过（已在生成阶段防过一遍，这里再次稳妥处理）
                if entry[1][-1][2]:
                    continue

                predicted_discharge = batteryPredictor.predict_discharge(
                    start_predict_time, last_hist_discharge, interval * 60
                )
                if predicted_discharge == -1:
                    continue

                predicted_sum += predicted_discharge
                real_sum += real_discharge
                valid_count += 1
                error_abs_list[idx].append(
                    (predicted_discharge - real_discharge) / battery_capacity
                )

            if valid_count > 0:
                if real_sum <= 0:
                    detacted_abnormal_user_in_runtime.add(user)
                    error_mat[idx][start_time] = float("nan")
                    weight_mat[idx][start_time] = 0
                    continue
                error_mat[idx][start_time] = (predicted_sum - real_sum) / real_sum
                weight_mat[idx][start_time] = valid_count
            else:
                error_mat[idx][start_time] = float("nan")
                weight_mat[idx][start_time] = 0

    # 计算“每个 interval 在 24 个起点上的加权平均误差”
    average_err, weights = weighted_average_2d(np.abs(error_mat), weight_mat, 1)

    if verbose:
        for idx in range(len(interval_list)):
            print(
                f"{user} interval {interval_list[idx]}h's average error is {average_err[idx]}, valid count is {weights[idx]}"
            )

    return user, error_mat, average_err, weights, error_abs_list


# eval_methods = ["aosp", "baseline", "kmeans_plus"]
eval_methods = ["kmeans_plus"]

for method in eval_methods:
    print(f"Evaluate {method}")
    user_error_mat = {}
    user_average_err = {}
    user_weights = {}
    user_error_abs_list = {}

    # 使用 tqdm 展示任务进度
    with concurrent.futures.ProcessPoolExecutor(max_workers) as executor:
        futures = [executor.submit(predictUserDischarge, user, method) for user in user_list]

        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures),
                           desc=f"Processing {method}",
                           ncols=100):
            user, error_mat, average_err, weights, error_abs_list = future.result()
            user_error_mat[user] = error_mat
            user_average_err[user] = average_err
            user_weights[user] = weights
            user_error_abs_list[user] = error_abs_list

    # 输出到文件
    df_user_average_err = pd.DataFrame(user_average_err)
    df_user_average_err.to_csv(f"{result_dir + method}.csv", index=False)
    print(df_user_average_err)

    df_user_weights = pd.DataFrame(user_weights)
    # df_user_weights.to_csv(f"{result_dir + method}_valid_samples.csv", index=False)

    # 计算总体加权平均误差（跨用户汇总）
    stacked_error = []
    stacked_weights = []
    for key in user_average_err.keys():
        stacked_error.append(user_average_err[key])
        stacked_weights.append(user_weights[key])
    stacked_error = np.array(stacked_error)
    stacked_weights = np.array(stacked_weights)

    avg_array, weights = weighted_average_2d(stacked_error, stacked_weights, 0)
    with open(f"{result_dir + method}.txt", "w") as f:
        for idx in range(len(interval_list)):
            f.write(
                f"interval {interval_list[idx]}h's average error is {avg_array[idx]}, valid count is {weights[idx]}\n"
            )

    print("detacted_abnormal_user_in_runtime:", detacted_abnormal_user_in_runtime)

    # 输出 MAE / battery_capacity 指标（每用户、分 interval、总体）
    # 每用户整体（所有 interval 合并）的 MAE/capacity
    per_user_overall_mae_cap = {}
    for user, lists_per_interval in user_error_abs_list.items():
        all_errs = []
        for lst in lists_per_interval:
            all_errs.extend(lst)
        per_user_overall_mae_cap[user] = float(np.mean(np.abs(all_errs))) if len(all_errs) > 0 else np.nan

    # 总体（所有用户、所有 interval 样本合并）的 MAE/capacity
    all_users_all_intervals = []
    for lists_per_interval in user_error_abs_list.values():
        for lst in lists_per_interval:
            all_users_all_intervals.extend(lst)
    overall_mae_cap = float(np.mean(np.abs(all_users_all_intervals))) if len(all_users_all_intervals) > 0 else np.nan

    # 分 interval 的 MAE/capacity（按样本平均）
    interval_mae_cap_sample_weighted = []
    for idx in range(len(interval_list)):
        errs = []
        for lists_per_interval in user_error_abs_list.values():
            errs.extend(lists_per_interval[idx])
        interval_mae_cap_sample_weighted.append(
            float(np.mean(np.abs(errs))) if len(errs) > 0 else np.nan
        )

    pd.Series(per_user_overall_mae_cap).to_csv(
        f"{result_dir + method}_mae_cap_per_user.csv", header=["Overall MAE"], index_label=["User"]
    )
    with open(f"{result_dir + method}_mae_cap.txt", "w") as f:
        f.write(f"overall MAE/capacity: {overall_mae_cap}\n")
        for idx, inter in enumerate(interval_list):
            f.write(f"interval {inter}h MAE/capacity: {interval_mae_cap_sample_weighted[idx]}\n")

    # 可选图表（保持和原工程一致，默认注释）
    # draw_error_abs_cdf(user_error_abs_list, interval_list, method, DATA_DIR)
    # draw_error_matrix_figure(user_error_mat, method)
