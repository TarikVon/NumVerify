import datetime
from collections import defaultdict
import pandas as pd
from sklearn.cluster import KMeans
from numvalkit.utils import dict_add
from numvalkit.core import BasePredictor

class BatteryKmeansPredictor(BasePredictor):

    def __init__(self, n_clusters, alpha, beta):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta

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
                # print("在 0-9 点之间")
                daily_vectors[0][date_of_entry] = dict_add(
                    daily_vectors[0][date_of_entry], entry[0]
                )
            elif datetime.time(9, 0, 0) <= time_of_entry < datetime.time(15, 0, 0):
                # print("在 9-15 点之间")
                daily_vectors[1][date_of_entry] = dict_add(
                    daily_vectors[1][date_of_entry], entry[0]
                )
            elif (
                datetime.time(15, 0, 0) <= time_of_entry
            ):  # time(24, 0, 0) 实际非法，可省略
                # print("在 15-24 点之间")
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
                # print(f'NORMAL')
            # else:
            # print(f'ERROR')
            daily_df.append(df)
            # print(df)
        all_categories = sorted(all_categories)
        for i, df in enumerate(daily_df):
            for cat in all_categories:
                if cat not in df.columns:
                    df[cat] = 0
            daily_df[i] = df[all_categories].sort_index()

        behavior_list = []
        for df in daily_df:
            # print(df)
            df_filtered = df.drop(columns=["off"], errors="ignore").fillna(0)
            # print(df_filtered)
            kmeans = KMeans(
                n_clusters=min(self.n_clusters, len(df)),
                random_state=42,
            )

            labels = kmeans.fit_predict(df_filtered)

            df_filtered["cluster"] = labels
            df_sorted = df_filtered.sort_index()
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
                cluster_id = (
                    behavior_df.loc[pd.Timestamp(day).strftime("%Y-%m-%d"), "cluster"]
                    if pd.Timestamp(day).strftime("%Y-%m-%d") in behavior_df.index
                    else 0
                )
                cluster_dates = behavior_df[behavior_df["cluster"] == cluster_id].index
                valid_train_dates = [d.date() for d in cluster_dates if d != day]
                # print(user_vector_sequence)
                df_train = discharge_df[discharge_df["date"].isin(valid_train_dates)]
                avg_discharge_by_time_list.append(
                    (
                        df_train[df_train["is_charging"] == False]
                        .groupby("time")["discharge"]
                        .mean()
                    )
                )

            valid_train_dates = [d for d in unique_dates if d != day]
            # print(user_vector_sequence)
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
        alldays_avg_discharge_by_time = self.average_per_granularity_discharge_dict[
            day
        ][1]
        if start_time.time() < datetime.time(9, 0, 0):
            # print("在 0-9 点之间")
            avg_discharge_by_time = self.average_per_granularity_discharge_dict[day][1]
        elif datetime.time(9, 0, 0) <= start_time.time() < datetime.time(15, 0, 0):
            # print("在 9-15 点之间")
            avg_discharge_by_time = self.average_per_granularity_discharge_dict[day][0][
                0
            ]
        elif (
            datetime.time(15, 0, 0) <= start_time.time()
        ):  # time(24, 0, 0) 实际非法，可省略
            # print("在 15-24 点之间")
            avg_discharge_by_time = self.average_per_granularity_discharge_dict[day][0][
                1
            ]
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
            predicted_discharge.append(
                self.alpha * last_hist_discharge
                + self.beta * hist_avg
            )

            if opcode == 1:
                if input - predicted_discharge[-1] >= 0:
                    battery_life_mins += self.vector_granularity
                else:
                    battery_life_mins += (
                        self.vector_granularity * input / predicted_discharge[-1]
                    )
                off_screen_discharge = self.vector_granularity * 1
                if predicted_discharge[-1] < off_screen_discharge:
                    input -= predicted_discharge[-1]
                else:
                    input -= (
                        predicted_discharge[-1] - off_screen_discharge
                    ) * ratio + off_screen_discharge
            elif opcode == 2:
                if input - self.vector_granularity >= 0:
                    discharge += predicted_discharge[-1]
                else:
                    discharge += (
                        predicted_discharge[-1] * input / self.vector_granularity
                    )
                input -= self.vector_granularity
            else:
                assert(0)
            current_time += pd.Timedelta(minutes=self.vector_granularity)

        if opcode == 1:
            return battery_life_mins
        elif opcode == 2:
            return discharge

    def predict_battery_life(
        self, start_time, last_hist_discharge, capacity, ratio
    ) -> float:
        return self.predict(start_time, last_hist_discharge, 1, capacity, ratio)

    def predict_discharge(self, start_time, last_hist_discharge, target_remaining_mins) -> float:
        return self.predict(start_time, last_hist_discharge, 2, target_remaining_mins)
