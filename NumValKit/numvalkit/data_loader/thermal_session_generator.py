from typing import List, Optional, Dict
from collections import defaultdict

from numvalkit.core import BaseGenerator
from numvalkit.utils import get_category_from_package
from numvalkit.data_loader import BatteryLoader, ForegroundLoader, ProductLoader


class ThermalSessionGenerator(BaseGenerator):
    """
    Loader for thermal session records.
    """

    def __init__(self, data_dir: str = "./data"):
        """
        :param data_dir: Root directory where user data folders are located.
        """
        self.data_dir = data_dir
        self.product_loader = ProductLoader(data_dir)
        self.battery_loader = BatteryLoader(data_dir)
        self.foreground_loader = ForegroundLoader(data_dir)
        self.minimal_duration_in_min = 10  # minimal sesion
        self.thermal_start_threshold = 35
        self.thermal_overheat_threshold = 38

    @staticmethod
    def _generate_screen_on_sessions(
        foreground_data: List[tuple], minimal_duration
    ) -> List[tuple]:
        sessions = []
        current_start = None
        current_end = None
        current_pkgs = []

        for start, end, pkg in sorted(foreground_data):
            if current_start is None:
                if "screen_off" in pkg.lower():  # skip those start with screen_off
                    continue
                current_start = start
                current_end = end
                current_pkgs = [(start, end, pkg)]
                continue

            if "screen_off" in pkg.lower():
                if (
                    current_end - current_start
                ).total_seconds() / 60 > minimal_duration:
                    # print(f"seesion append {(current_end - current_start).total_seconds()}s, start {current_end}")
                    sessions.append((current_start, current_end, current_pkgs))
                current_start = None
                current_end = None
                current_pkgs = []
                continue

            if start <= current_end:
                current_end = max(current_end, end)
                current_pkgs.append((start, end, pkg))
            else:  # Time gap, record last
                if (
                    current_end - current_start
                ).total_seconds() / 60 > minimal_duration:
                    sessions.append((current_start, current_end, current_pkgs))
                current_start = start
                current_end = end
                current_pkgs = [(start, end, pkg)]

        if current_start is not None:
            sessions.append((current_start, current_end, current_pkgs))

        return sessions

    def generate(self, user: str) -> Optional[List[Dict]]:
        """
        ThermalSession, with high temp marks (start temp > thermal_start_threshold)
        and (max_temp > thermal_overheat_threshold)
        """
        battery_data = self.battery_loader.load(user, True)  # With thermal data
        foreground_data = self.foreground_loader.load(user)

        if battery_data is None or foreground_data is None:
            return None

        sessions = self._generate_screen_on_sessions(
            foreground_data, self.minimal_duration_in_min
        )

        results = []

        for start, end, pkg_list in sessions:
            records_in_range = [
                (s, e, b, t, c)
                for (s, e, b, t, c) in battery_data
                if s <= end and e >= start
            ]
            if not records_in_range:
                continue

            timestamps = [s for s, _, _, _, _ in records_in_range] + [
                e for _, e, _, _, _ in records_in_range
            ]
            if min(timestamps) > start or max(timestamps) < end:
                continue

            battery_series, thermal_series, charge_series = [], [], []
            for _, _, b, t, c in records_in_range:
                battery_series.append(b)
                thermal_series.append(t)
                charge_series.append(c)

            if not thermal_series:
                continue

            start_temp = thermal_series[0]
            if start_temp > self.thermal_start_threshold:
                continue

            is_charging = max(charge_series) == 1
            try:
                temp_raise = (
                    thermal_series[1] - thermal_series[0]
                    if len(thermal_series) > 1
                    else 0
                )
            except:
                temp_raise = 0

            duration = (end - start).total_seconds()
            try:
                battery_discharge = (
                    (battery_series[1] - battery_series[0]) / duration
                    if duration > 0
                    else 0
                )
            except:
                battery_discharge = 0

            cat_time = defaultdict(float)
            for s1, e1, pkg in pkg_list:
                usage_time = (e1 - s1).total_seconds()
                cat = get_category_from_package(pkg)
                if cat == "off":
                    print(sessions)
                    exit(-1)
                if cat:
                    cat_time[cat] += usage_time
            total_time = sum(cat_time.values())
            if total_time == 0:
                continue
            vector = {k: v / total_time for k, v in cat_time.items()}
            max_temp = max(thermal_series)
            avg_temp = sum(thermal_series) / len(thermal_series)
            high_temp = int(max_temp > self.thermal_overheat_threshold)
            product = self.product_loader.load(user)

            results.append(
                {
                    "user": user,
                    "start": start,
                    "end": end,
                    "product": product,
                    "start_hour": start.hour,
                    "duration_min": duration / 60,
                    "temp_raise": temp_raise,
                    "battery_discharge": battery_discharge,
                    "start_temp": start_temp,
                    "is_charging": int(is_charging),
                    "max_temp": max_temp,
                    "avg_temp": avg_temp,
                    "high_temp": high_temp,
                    "vector": vector,
                }
            )

        return results
