import os
import csv
from datetime import datetime
from typing import List, Tuple, Optional

from numvalkit.core.base_loader import BaseLoader


class BatteryLoader(BaseLoader):
    """
    Loader for battery data. Expects CSV at {data_dir}/{user}/battery_data.csv
    with columns including 'timestamp' and 'shell_frame_temp'.
    """

    def __init__(self, data_dir: str = "./data"):
        """
        :param data_dir: Root directory where user data folders are located.
        """
        self.data_dir = data_dir
        self.abnormal_duration = 3600 * 6

    def load(
        self, user: str, with_thermal: bool = False
    ) -> Optional[List[Tuple[datetime, datetime, float, bool]]]:
        """
        Load battery data for the specified user.

        :param user: Identifier for the user (e.g., folder name under data_dir)
        :return: List of tuples (start, end, discharge, is_charging) or None if file is missing.
        """
        path = os.path.join(self.data_dir, user, "battery_data.csv")
        if not os.path.exists(path):
            return None

        records: List[Tuple[datetime, datetime, float, bool]] = []
        prev_row = None
        with open(path, mode="r", newline="") as file:
            reader = csv.reader(file)
            next(reader)  # 跳过表头
            for row in reader:
                if prev_row is not None:
                    start_time = datetime.strptime(prev_row[0], "%Y-%m-%d %H:%M:%S")
                    end_time = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                    battery_usage = float(prev_row[1]) - float(row[1])
                    charging = (
                        (battery_usage < -10)
                        or int(float(prev_row[3])) == 1
                        or int(float(row[3])) == 1
                    )
                    thermal = float(row[2]) / 10.0
                    if thermal > 100:  # user 1-20, diff format
                        thermal = thermal / 100

                    # Filter for long duration
                    if (end_time - start_time).total_seconds() > self.abnormal_duration:
                        prev_row = row
                        continue
                    if with_thermal:  # Also add thermal in records
                        records.append(
                            (start_time, end_time, battery_usage, thermal, charging)
                        )
                    else:
                        records.append((start_time, end_time, battery_usage, charging))
                prev_row = row
        return records
