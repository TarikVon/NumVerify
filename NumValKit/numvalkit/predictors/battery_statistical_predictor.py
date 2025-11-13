import pandas as pd
from numvalkit.core import BasePredictor

class BatteryStatisticalPredictor(BasePredictor):
    n_clusters = 7
    alpha = 0.3
    beta = 0.7

    def fit(self, user_vector_sequence, vector_granularity) -> None:
        self.vector_granularity = vector_granularity

        discharge_df = pd.DataFrame(
            [(k.date(), k.time(), v[1], v[2]) for k, v in user_vector_sequence.items()],
            columns=["date", "time", "discharge", "is_charging"],
        )
        unique_dates = sorted({ts.date() for ts in user_vector_sequence.keys()})
        self.average_per_granularity_discharge_dict = {}
        for day in unique_dates:
            valid_train_dates = [d for d in unique_dates if d != day]
            # print(user_vector_sequenc
            df_train = discharge_df[discharge_df["date"].isin(valid_train_dates)]
            alldays_avg_discharge_by_time = (
                discharge_df[discharge_df["is_charging"] == False]
                .groupby("time")["discharge"]
                .mean()
            )
            self.average_per_granularity_discharge_dict[day] = (
                alldays_avg_discharge_by_time
            )
    
    def predict(self, start_time, last_hist_discharge, opcode, input, ratio=1) -> float:
        # opcode 0 -> predict_battery_life, input = capacity
        # opcode 1 -> predict_discharge, input =  target_remaining_mins
        day = start_time.date()
        alldays_avg_discharge_by_time = self.average_per_granularity_discharge_dict[day]
        current_time = start_time
        predicted_discharge = []
        discharge = 0
        battery_life_mins = 0
        while input > 0:
            if current_time.time() in alldays_avg_discharge_by_time:
                hist_avg = alldays_avg_discharge_by_time[current_time.time()]
            else:
                exit(0)
            if len(predicted_discharge) == 0:
                predicted_discharge.append(
                    BatteryStatisticalPredictor.alpha * last_hist_discharge
                    + BatteryStatisticalPredictor.beta * hist_avg
                )
            else:
                predicted_discharge.append(
                    BatteryStatisticalPredictor.alpha * predicted_discharge[-1]
                    + BatteryStatisticalPredictor.beta * hist_avg
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
