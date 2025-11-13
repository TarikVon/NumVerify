from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseGenerator(ABC):
    """
    Abstract base class defining the interface for all Generators.
    """

    def __init__(self, data_dir: str = "./data"):
        """
        Initialize the generator.

        :param data_dir: Root directory where input data and cached outputs are stored.
        """
        self.data_dir = data_dir

    @abstractmethod
    def generate(
        self, user: str, vector_granularity: int, reflush: bool
    ) -> Optional[Dict]:
        """
        Generate time‐indexed feature vectors for a given user.

        :param user: Identifier of the user to process.
        :param vector_granularity: Time step (e.g., in seconds) for vector aggregation.
        :param reflush: If True, ignore any existing cache and regenerate results.
        :return:
            - A dict mapping each timestamp (datetime) to a tuple:
                1. A dict of feature names to their float values.
                2. A summary float (e.g., total usage or score).
                3. A boolean flag (e.g., indicating validity or anomaly).
            - None if no data is available or generation fails.
        """
        pass
