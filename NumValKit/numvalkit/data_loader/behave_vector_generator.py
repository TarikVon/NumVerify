import os
import pickle
from collections import defaultdict
from datetime import datetime
from typing import Tuple, Optional, Dict
import pandas as pd

from numvalkit.core import BaseGenerator
from numvalkit.data_loader import BatteryLoader, ForegroundLoader
from numvalkit.utils import get_category_from_package, get_overlap_seconds


class BehaveVectorGenerator(BaseGenerator):
    """
    Calls BatteryLoader and ForegroundLoader, generate a dict: {time_interval -> (behavior vector, discharge, is_charging)}
    """

    def __init__(self, data_dir: str = "./data"):
        """
        :param data_dir: Root directory where user data folders are located.
        """
        self.data_dir = data_dir
        self.batteryLoader = BatteryLoader(data_dir)
        self.foregroundLoader = ForegroundLoader(data_dir)

    def generate(self, user: str, vector_granularity, use_appname, reflush=False) -> Optional[Dict]:
        """
        Load battery data and foreground data from their data loaders for the specified user.
        Generate a dict: Dict[pd.Timestamp, Tuple[Dict[str, float], float, bool]]

        :param user: Identifier for the user (e.g., folder name under data_dir)
        :return: Dict[pd.Timestamp, Tuple[Dict[str, float], float, bool]] or None if file is missing.
        """
        path = os.path.join(self.data_dir, user)
        if not os.path.exists(path):
            return None
        vector_granularity = int(vector_granularity)
        if os.path.exists(f"{path}/vector_{vector_granularity}_{use_appname}") and (not reflush):
            with open(f"{path}/vector_{vector_granularity}_{use_appname}", "rb") as f:
                return pickle.load(f)
        else:
            print(f"compute vector of granularity {vector_granularity} {use_appname} for {user}")

        # read foreground data
        data = self.foregroundLoader.load(user)

        user_vector_sequence = {}
        usage_vector = defaultdict(int)

        if len(data) == 0:
            return {}

        vector_start_time = pd.Timestamp(data[0][0]).ceil(f"{vector_granularity}min")
        vector_end_time = vector_start_time + pd.Timedelta(minutes=vector_granularity)

        def insert_vector():
            nonlocal user_vector_sequence, usage_vector, vector_start_time, vector_end_time
            sum = 0
            for key, value in usage_vector.items():
                sum += value
            if sum >= vector_granularity * 60 - 1:
                user_vector_sequence[vector_start_time] = [[usage_vector, 0, False], 0]
            usage_vector = defaultdict(int)
            vector_start_time = vector_end_time
            vector_end_time += pd.Timedelta(minutes=vector_granularity)

        for entry in data:
            if use_appname:
                category = entry[2]
            else:
                category = get_category_from_package(entry[2])
            while 1:

                overlap_seconds = get_overlap_seconds(
                    vector_start_time,
                    vector_end_time,
                    pd.Timestamp(entry[0]),
                    pd.Timestamp(entry[1]),
                )
                if overlap_seconds > 0:
                    usage_vector[category] += overlap_seconds
                if pd.Timestamp(entry[1]) >= vector_end_time:
                    insert_vector()
                elif pd.Timestamp(entry[1]) < vector_end_time:
                    break

        # read battery data
        data = self.batteryLoader.load(user)
        for entry in data:
            seconds_in_entry = (
                pd.Timestamp(entry[1]) - pd.Timestamp(entry[0])
            ).total_seconds()
            vector_start_time = pd.Timestamp(entry[0]).floor(f"{vector_granularity}min")
            vector_end_time = vector_start_time + pd.Timedelta(
                minutes=vector_granularity
            )
            while 1:
                if vector_start_time > pd.Timestamp(entry[1]):
                    break
                overlap_seconds = get_overlap_seconds(
                    vector_start_time, vector_end_time, entry[0], entry[1]
                )
                if overlap_seconds > 0:
                    if vector_start_time in user_vector_sequence:
                        user_vector_sequence[vector_start_time][1] += overlap_seconds
                        user_vector_sequence[vector_start_time][0][1] += (
                            float(entry[2]) * overlap_seconds / seconds_in_entry
                        )
                        user_vector_sequence[vector_start_time][0][2] |= bool(entry[3])

                vector_start_time = vector_end_time
                vector_end_time += pd.Timedelta(minutes=vector_granularity)

        ret = {}
        for key, entry in user_vector_sequence.items():
            vector_sum = sum(entry[0][0].values())
            if (
                entry[1] < 60 * vector_granularity - 1
                or vector_sum < 60 * vector_granularity - 1
            ):
                continue
            else:
                ret[key] = entry[0]

        with open(f"{path}/vector_{vector_granularity}_{use_appname}", "wb") as f:
            pickle.dump(ret, f)
            print(f"store vector of granularity {vector_granularity} {use_appname} for {user}")

        return ret
