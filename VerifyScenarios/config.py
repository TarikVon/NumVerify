import os
import csv

path = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = f"{path}/../../UserModel/data"

user_list = [f"user{i}" for i in range(1, 1541)] + ["zhanye"]
filter = ["user1422", "user1236", "user1483", "user1496", "user1295", "user520", "user702", "user1018", "user981", "user1239"]

user_list = [user for user in user_list if user not in filter]

# battery_capacity = 5500

regenerate_cached_data = False

detacted_abnormal_user_in_runtime = set()


def get_user_list(path):
    filter_list = []
    # user_list = ["zhanye"]
    for user in user_list:
        app_set = set()
        csv_file_path = f"{path}/{user}/foreground.csv"

        with open(csv_file_path, mode="r", newline="") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                name = row[-1]
                app_set.add(name)
        if len(app_set) < 50:
            filter_list.append(user)
    print("filter users with fewer than 50 apps: ", filter_list)
    new_filter = filter + filter_list
    return [user for user in user_list if user not in new_filter]


# user_list = get_user_list(DATA_DIR)
