import pandas as pd

from numvalkit.core.base_loader import BaseLoader


class ProductLoader(BaseLoader):
    """
    Loader for user phone type
    """

    @staticmethod
    def _load_user_product_map(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df["user"].notna()]
        user_product_map = dict(zip(df["user"], df["productname"]))
        return user_product_map

    def __init__(self, data_dir: str = "./data"):
        """
        Construct the user to phone map
        """
        self.data_dir = data_dir
        self.user_product_map = self._load_user_product_map(
            f"{data_dir}/phone-type.csv"
        )

    def load(self, user: str) -> str:
        """
        Return the phone type of user
        """
        return self.user_product_map.get(user, float("nan"))
