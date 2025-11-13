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
# 预测时间：[12,6,3,1]
# 要求：开始时间之前有5个向量 （2.5小时），预测时间期间没有充电

start_time_list = range(0, 24)
interval_list = [12, 9, 6, 3, 1]
vector_granularity = 60  # min
seq_len = 5
max_workers = 112
verbose = False


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
        interval_list,
        seq_len,
        vector_granularity,
        reflush,
    ):
        path = os.path.join(self.data_dir, user)
        if not os.path.exists(path):
            return None

        if os.path.exists(f"{path}/dataset_{vector_granularity}_{seq_len}") and (
            not reflush
        ):

            if verbose:
                print(
                    f"read dataset of granularity {vector_granularity} from file {path}s/dataset_{vector_granularity}_{seq_len}"
                )
            with open(f"{path}/dataset_{vector_granularity}_{seq_len}", "rb") as f:
                return pickle.load(f)
        else:
            if verbose:
                print(
                    f"compute dataset of granularity {vector_granularity}, seq_len {seq_len} for {user}"
                )

        # dict{(start_time, interval): [(day, [vectors], discharge)]}
        val_dict = defaultdict(list)

        unique_dates = sorted({ts.date() for ts in user_vector_sequence.keys()})
        # print(unique_dates)
        for start_time in start_time_list:
            for interval in interval_list:
                entry_list = []
                for day in unique_dates:
                    start_date_time = pd.Timestamp(
                        datetime.datetime.combine(
                            day, datetime.time(hour=start_time, minute=0)
                        )
                    )
                    vector_list = []
                    vector_missing = False
                    for delta in range(
                        -vector_granularity * seq_len, 0, vector_granularity
                    ):
                        time = (start_date_time + pd.Timedelta(minutes=delta)).round(
                            "min"
                        )
                        # print(time)
                        if time in user_vector_sequence:
                            vector_list.append(user_vector_sequence[time])
                            # print(user_vector_sequence[time][0])
                        else:
                            vector_missing = True
                            break
                    if vector_missing:
                        continue
                    discharge = 0
                    need_skipping = False
                    for delta in range(0, interval * 60, vector_granularity):
                        # print(delta)
                        time = (start_date_time + pd.Timedelta(minutes=delta)).round(
                            "min"
                        )
                        # print(time)
                        # print(time, user_vector_sequence[time][2])
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

        # for key,value in val_dict.items():
        #     print(key, len(value))

        with open(f"{path}/dataset_{vector_granularity}_{seq_len}", "wb") as f:
            pickle.dump(val_dict, f)
            if verbose:
                print(
                    f"store dataset of granularity {vector_granularity}, seq_len {seq_len} for {user}"
                )
        return val_dict


def predictUserDischarge(user, method_name):
    n_intervals = len(interval_list)

    error_mat = np.zeros((n_intervals, 24))
    weight_mat = np.zeros((n_intervals, 24))
    error_abs_list = [[] for _ in range(len(interval_list))]
    # battery_capacity = get_productname_from_user(user)
    battery_capacity = 5500

    behaveVectorGenerator = BehaveVectorGenerator(DATA_DIR)
    user_vector_sequence = behaveVectorGenerator.generate(
        user, vector_granularity, False, regenerate_cached_data
    )
    datasetGenerator = DatasetGenerator(DATA_DIR)
    val_dict = datasetGenerator.generate_dataset(
        user,
        user_vector_sequence,
        start_time_list,
        interval_list,
        seq_len,
        vector_granularity,
        regenerate_cached_data,
    )

    def safe_fit(predictor, *args):
        try:
            predictor.fit(*args)
            return predictor, None
        except Exception as e:
            detacted_abnormal_user_in_runtime.add(user)
            return None,e

    if method_name == "kmeans":
        batteryPredictor = BatteryKmeansPredictor(10, 0.2, 0.8)
        # batteryPredictor.fit(user_vector_sequence, vector_granularity)
        batteryPredictor, err = safe_fit(batteryPredictor, user_vector_sequence, vector_granularity)
    elif method_name == "kmeans_plus":
        batteryPredictor = BatteryKmeansPredictorPlus(
            10,
            0.2,
            0.8, 
            standardize=True,
            use_pca=True,
            weekend_tag=True,
            time_triangle=True
        )
        # batteryPredictor.fit(user_vector_sequence, vector_granularity)
        batteryPredictor, err = safe_fit(batteryPredictor, user_vector_sequence, vector_granularity)
    elif method_name == "baseline":
        batteryPredictor = BatteryStatisticalPredictor()
        # batteryPredictor.fit(user_vector_sequence, vector_granularity)
        batteryPredictor, err = safe_fit(batteryPredictor, user_vector_sequence, vector_granularity)
    elif method_name == "aosp":
        batteryPredictor = BatteryAospPredictor()
        # batteryPredictor.fit(f"{DATA_DIR}/{user}/battery_data.csv", vector_granularity, battery_capacity)
        batteryPredictor, err = safe_fit(batteryPredictor, f"{DATA_DIR}/{user}/battery_data.csv", vector_granularity, battery_capacity)
    elif method_name == "ae":
        batteryPredictor = BatteryAEPredictor(
            n_clusters=10,
            alpha=0.2,
            beta=0.8,
            z_dim=16,
            ae_epochs=200,
            ae_batch_size=256,
            ae_lr=1e-3,
            user_id=user
        )
        # batteryPredictor.fit(f"{DATA_DIR}/{user}/battery_data.csv", vector_granularity, battery_capacity)
        batteryPredictor, err = safe_fit(batteryPredictor, user_vector_sequence, vector_granularity)
    elif method_name == "dt":
        batteryPredictor = BatteryDTPredictor()
        # batteryPredictor.fit(f"{DATA_DIR}/{user}/battery_data.csv", vector_granularity, battery_capacity)
        batteryPredictor, err = safe_fit(batteryPredictor, user_vector_sequence, vector_granularity)
    elif method_name == "oracle_time_median":
        batteryPredictor = OracleTimeMedianPredictor()
        batteryPredictor, err = safe_fit(batteryPredictor, user_vector_sequence, vector_granularity)
    elif method_name == "oracle_kmeans_median":
        batteryPredictor = OracleKMeansMedianPredictor(n_clusters=10)
        batteryPredictor, err = safe_fit(batteryPredictor, user_vector_sequence, vector_granularity)
    else:
        exit(-1)

    if batteryPredictor is None:
        nan_err = np.full((n_intervals,24),np.nan)
        zero_w = np.zeros((n_intervals,), dtype=float)
        empty_abs = [[] for _ in range(n_intervals)]
        return user, nan_err, np.full((n_intervals,), np.nan), zero_w, empty_abs

    for start_time in start_time_list:
        for idx, interval in enumerate(interval_list):
            entry_list = val_dict[(start_time, interval)]
            predicted_sum = 0
            real_sum = 0
            valid_count = 0
            for entry in entry_list:
                real_discharge = entry[2]
                day = entry[0]
                start_predict_time = pd.Timestamp(day).replace(
                    hour=start_time, minute=0
                )
                last_hist_discharge = entry[1][-1][1]
                if entry[1][-1][2]:  # if in the last time slice the battery is charging
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
                # print(predicted_sum, real_sum, error_mat[idx][start_time])
            else:
                error_mat[idx][start_time] = float("nan")
                weight_mat[idx][start_time] = 0

    # print(user)
    # print(error_mat)
    average_err, weights = weighted_average_2d(np.abs(error_mat), weight_mat, 1)
    # np.nanmean(np.abs(error_mat), axis=1)

    if verbose:
        for idx in range(len(interval_list)):
            print(
                f"{user} interval {interval_list[idx]}h's average error is {average_err[idx]}, valid count is {weights[idx]}"
            )

    return user, error_mat, average_err, weights, error_abs_list

# eval_methods = ["aosp", "baseline", "kmeans_plus"]
eval_methods = ["kmeans_plus", "oracle_time_median", "oracle_kmeans_median"]
for method in eval_methods:
    print(f"Evaluate {method}")
    user_error_mat = {}
    user_average_err = {}
    user_weights = {}
    user_error_abs_list = {}

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
    # print(user_average_err)
    df_user_average_err = pd.DataFrame(user_average_err)
    df_user_average_err.to_csv(f"{method}.csv", index=False)
    print(df_user_average_err)
    df_user_weights = pd.DataFrame(user_weights)
    # df_user_weights.to_csv(f"{method}_valid_samples.csv", index=False)

    # 计算平均误差
    stacked_error = []
    stacked_weights = []
    for key in user_average_err.keys():
        stacked_error.append(user_average_err[key])
        stacked_weights.append(user_weights[key])
    stacked_error = np.array(stacked_error)
    stacked_weights = np.array(stacked_weights)

    # stacked_error = np.stack(list(user_average_err.values()))
    # stacked_weights = np.stack(list(user_weights.values()))
    avg_array, weights = weighted_average_2d(stacked_error, stacked_weights, 0)
    with open(f"{method}.txt", "w") as f:
        for idx in range(len(interval_list)):
            f.write(
                f"interval {interval_list[idx]}h's average error is {avg_array[idx]}, valid count is {weights[idx]}\n"
            )
    # print(avg_array)

    print("detacted_abnormal_user_in_runtime:", detacted_abnormal_user_in_runtime)

    # 输出 MAE / battery_capacity 指标（每用户、分 interval、总体）
    # 每用户整体（四个 interval 合并）的 MAE/capacity
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

    # 分 interval 的 MAE/capacity
    interval_mae_cap_sample_weighted = []
    for idx in range(len(interval_list)):
        errs = []
        for lists_per_interval in user_error_abs_list.values():
            errs.extend(lists_per_interval[idx])
        interval_mae_cap_sample_weighted.append(
            float(np.mean(np.abs(errs))) if len(errs) > 0 else np.nan
        )

    pd.Series(per_user_overall_mae_cap).to_csv(f"{method}_mae_cap_per_user.csv", header=["Overall MAE"], index_label=["User"])
    with open(f"{method}_mae_cap.txt", "w") as f:
        f.write(f"overall MAE/capacity: {overall_mae_cap}\n")
        for idx, inter in enumerate(interval_list):
            f.write(f"interval {inter}h MAE/capacity: {interval_mae_cap_sample_weighted[idx]}\n")

    # draw_error_abs_cdf(user_error_abs_list, interval_list, method_name)
    # draw_error_matrix_figure(user_error_mat, method_name)
