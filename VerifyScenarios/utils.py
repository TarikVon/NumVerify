import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import os

data_path = os.path.dirname(os.path.abspath(__file__))+"/../../UserModel/data/phone-type.csv"

def get_productname_from_user(user):
    return user_product_map.get(user, float('nan'))


def load_user_product_map(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['user'].notna()]
    user_product_map = dict(zip(df['user'], df['productname']))
    return user_product_map


user_product_map = load_user_product_map(data_path)

def weighted_average_2d(value_mat, weight_mat, axis):
    if axis == 1:
        weights = np.sum(weight_mat, axis=1)
        avg_array = np.zeros(value_mat.shape[0])
        for idx in range(len(value_mat)):
            if weights[idx] == 0:
                avg_array[idx] = float("nan")
            else:
                weighted_sum = 0
                for idx2 in range(len(value_mat[idx])):
                    if not np.isnan(value_mat[idx][idx2]):
                        weighted_sum += value_mat[idx][idx2] * weight_mat[idx][idx2]
                avg_array[idx] = weighted_sum / weights[idx]
        return avg_array, weights
    elif axis == 0:
        weights = np.sum(weight_mat, axis=0)
        avg_array = np.zeros(value_mat.shape[1])
        for idx in range(value_mat.shape[1]):
            if weights[idx] == 0:
                avg_array[idx] = float("nan")
            else:
                weighted_sum = 0
                for idx2 in range(value_mat.shape[0]):
                    if not np.isnan(value_mat[idx2][idx]):
                        weighted_sum += value_mat[idx2][idx] * weight_mat[idx2][idx]
                avg_array[idx] = weighted_sum / weights[idx]
        return avg_array, weights
    else:
        exit(1)


def draw_error_matrix_figure(user_error_mat_dict, method_name):
    user_list = user_error_mat_dict.keys()
    num_users = len(user_list)
    cols = 2
    rows = int(np.ceil(num_users / cols))

    fig, axes = plt.subplots(
        rows, cols, figsize=(18, rows * 3), sharex=True, sharey=True
    )
    axes = axes.flatten()

    for i, user in enumerate(user_list):
        ax = axes[i]
        # difference = (opt_mat[user] - baseline_mat[user]) / baseline_mat[user]
        difference = user_error_mat_dict[user]
        # 创建自定义颜色映射
        colors = ["green", "lightgreen", "yellow", "orange", "red"]
        cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)
        # 绘制差异矩阵
        im = ax.imshow(difference, cmap=cmap, interpolation="nearest")

        # 添加颜色条
        cbar = plt.colorbar(im)
        cbar.set_label("Difference Magnitude", rotation=270, labelpad=20)
        ax.set_ylabel("start time")
        ax.set_xlabel("interval")
        ax.set_title(f"{user}")

        y_ticks = [0, 1, 2, 3]  # Assuming these correspond to 12,6,3,1 in your data
        y_tick_labels = ["12", "6", "3", "1"]
        x_ticks = range(0, 24)  # 0 to 23
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks)

        # 在每个单元格中添加数值标签
        for i in range(difference.shape[0]):
            for j in range(difference.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{difference[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                )

    # 删除多余子图
    for j in range(num_users, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Daily MAE per User", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{method_name}.png")


def draw_error_abs_cdf(user_error_abs_list, interval_list, method_name, data_dir):
    data_list = []
    for idx in range(len(interval_list)):
        data = []
        for user in user_error_abs_list:
            data.extend(user_error_abs_list[user][idx])
        data_list.append(data)

    titles = interval_list

    # 创建 2x2 的子图网格
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # 绘制每个列表的 CDF
    for i, data in enumerate(data_list):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        sorted_data = np.sort(np.abs(data))
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        ax.plot(sorted_data, cdf, marker=".", linestyle="none")
        ax.set_title(titles[i])
        ax.set_xlabel("Ratio of battery capacity")
        ax.set_ylabel("CDF")
        ax.grid(True)

    plt.tight_layout()  # 自动调整子图间距
    plt.savefig(f"{data_dir}/{method_name}_error_abs_cdf.png")
    plt.close()
    plt.clf()

    for user, data_list in user_error_abs_list.items():
        # 创建 2x2 的子图网格
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # 绘制每个列表的 CDF
        for i, data in enumerate(data_list):
            row = i // 2
            col = i % 2
            ax = axes[row, col]

            sorted_data = np.sort(np.abs(data))
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

            ax.plot(sorted_data, cdf, marker=".", linestyle="none")
            ax.set_title(titles[i])
            ax.set_xlabel("Ratio of battery capacity")
            ax.set_ylabel("CDF")
            ax.grid(True)

        plt.tight_layout()  # 自动调整子图间距
        plt.savefig(f"{data_dir}/data/{user}/{method_name}_error_abs_cdf.png")
        plt.close()
        plt.clf()
