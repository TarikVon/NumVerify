from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from datetime import datetime
import os
import csv


class BaseLoader(ABC):
    """
    Abstract base class defining the interface for all Loaders.
    """

    def __init__(self, data_dir: str = "./data"):
        """
        Initialize the loader.

        :param data_dir: Root directory where user data is stored.
        """
        self.data_dir = data_dir

    @abstractmethod
    def load(self, user: str) -> Optional[List]:
        """
        Load data for the specified user.

        :param user: Identifier for the user whose data will be loaded.
        :return: A list of tuples (start_time, end_time, label) if data exists,
                 otherwise None.
        """
        pass
