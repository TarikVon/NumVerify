import datetime
import csv
import pandas as pd
from numvalkit.core import BasePredictor

class BatteryAospPredictor(BasePredictor):

    def fit(self, battery_data_path, vector_granularity, battery_capacity) -> None:
        self.vector_granularity = int(vector_granularity)
        self.battery_capacity = battery_capacity
        # 读取discharge
        data = []
        # 读取数据
        with open(battery_data_path, mode="r", newline="") as file:
            reader = csv.reader(file)
            next(reader)  # 跳过表头
            for row in reader:
                timestamp = pd.Timestamp(
                    datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                )
                battery = float(row[1])
                charging = int(float(row[3])) == 1
                data.append((timestamp, battery, charging))
        # print(data)
        discharge_time_per_level = []
        current_level_start_timestamp = data[0][0]
        current_level_start_capacity = data[0][1]
        is_charging = data[0][2]
        per_level_capacity = battery_capacity / 100

        for idx in range(1, len(data)):
            current_entry = data[idx]
            last_entry = data[idx - 1]
            seconds_from_last_entry = (current_entry[0] - last_entry[0]).total_seconds()
            if seconds_from_last_entry > 3600 * 6 or current_entry[2] != is_charging:
                current_level_start_timestamp = current_entry[0]
                current_level_start_capacity = current_entry[1]
                is_charging = current_entry[2]
                continue
            if is_charging == True:
                continue

            capacity_threshold = current_level_start_capacity - per_level_capacity
            while current_entry[1] <= capacity_threshold:
                discharge_from_last_entry = last_entry[1] - current_entry[1]
                assert discharge_from_last_entry > 0
                seconds_from_last_entry_to_touch_the_threshold = (
                    seconds_from_last_entry
                    * (last_entry[1] - capacity_threshold)
                    / discharge_from_last_entry
                )
                timestamp_of_touching_threshold = last_entry[0] + pd.Timedelta(
                    seconds=seconds_from_last_entry_to_touch_the_threshold
                )
                seconds_to_discharge_the_level = (
                    timestamp_of_touching_threshold - current_level_start_timestamp
                ).total_seconds()
                discharge_time_per_level.append(
                    (
                        current_level_start_timestamp,
                        timestamp_of_touching_threshold,
                        seconds_to_discharge_the_level,
                    )
                )

                current_level_start_timestamp = timestamp_of_touching_threshold
                current_level_start_capacity = capacity_threshold
                capacity_threshold = current_level_start_capacity - per_level_capacity

        self.per_level_discharge_time_list = discharge_time_per_level

    def predict(self, start_time, last_hist_discharge, opcode, input, ratio=1) -> float:
        # opcode 0 -> predict_battery_life, input = capacity
        # opcode 1 -> predict_discharge, input =  target_remaining_mins

        # 1. 过滤出 Timestamp 小于 x 的数据
        filtered_data = [
            item for item in self.per_level_discharge_time_list if item[1] < start_time
        ]
        # 2. 获取最后 200 项（如果数据少于 200，直接取所有）
        last_200_items = filtered_data[-200:]
        # 3. 计算第 2 维（即第二个元素）的平均值
        average_discharge_seconds_per_level = (
            sum(item[2] for item in last_200_items) / len(last_200_items)
            if last_200_items
            else 0
        )

        if len(last_200_items) < 5:
            return -1

        if opcode == 1:
            discharge_per_second = (
                self.battery_capacity / 100 / average_discharge_seconds_per_level
            )
            discharge_per_graularity = self.vector_granularity * 60 * discharge_per_second
            off_screen_discharge = self.vector_granularity * 1
            if discharge_per_graularity < off_screen_discharge:
                scaled_discharge_per_graularity = discharge_per_graularity
            else:
                scaled_discharge_per_graularity = (
                    discharge_per_graularity - off_screen_discharge
                ) * ratio + off_screen_discharge

            battery_life_mins = (
                input / scaled_discharge_per_graularity * self.vector_granularity
            )
            return battery_life_mins
        elif opcode == 2:
            predicted_discharge = []
            discharge = 0
            while input > 0:
                predicted_discharge.append(
                    self.vector_granularity
                    * 60
                    / average_discharge_seconds_per_level
                    * self.battery_capacity
                    / 100
                )
                if input - self.vector_granularity >= 0:
                    discharge += predicted_discharge[-1]
                else:
                    discharge += (
                        predicted_discharge[-1] * input / self.vector_granularity
                    )
                input -= self.vector_granularity
            return discharge

    def predict_battery_life(
        self, start_time, last_hist_discharge, capacity, ratio
    ) -> float:
        return self.predict(start_time, last_hist_discharge, 1, capacity, ratio)

    def predict_discharge(self, start_time, last_hist_discharge, target_remaining_mins) -> float:
        return self.predict(start_time, last_hist_discharge, 2, target_remaining_mins)
