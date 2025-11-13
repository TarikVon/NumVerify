import os
from datetime import datetime
from typing import List, Tuple, Optional
from numvalkit.core.base_loader import BaseLoader


class ForegroundLoader(BaseLoader):
    """
    Loader for foreground app usage records. Expects CSV at {data_dir}/{user}/foreground.csv
    with columns: start_timestamp, end_timestamp, package_name.
    """

    def __init__(self, data_dir: str = "./data"):
        """
        :param data_dir: Root directory where user data folders are located.
        """
        self.data_dir = data_dir

    def load(
        self, user: str, pure_app: bool = False
    ) -> Optional[List[Tuple[datetime, datetime, str]]]:
        """
        Load foreground usage records for the specified user.

        :param user: Identifier for the user (folder name under data_dir)
        :return: List of tuples (start, end, package_name) or None if file is missing.
        """
        path = os.path.join(self.data_dir, user, "foreground.csv")
        if not os.path.exists(path):
            return None

        records: List[Tuple[datetime, datetime, str]] = []
        with open(path, "r") as f:
            # skip header
            next(f)
            for line in f:
                parts = line.strip().split(",")
                try:
                    start = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
                    end = datetime.strptime(parts[1], "%Y-%m-%d %H:%M:%S")
                    pkg = parts[2]
                    if pure_app and (
                        pkg == "com.ohos.sceneboard" or "screen_off" in pkg
                    ):
                        continue
                    records.append((start, end, pkg))
                except (IndexError, ValueError):
                    # skip malformed lines
                    continue
        return records
