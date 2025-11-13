import datetime
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
import os

from numvalkit.utils import dict_add
from numvalkit.core import BasePredictor


# ====== 简单自编码器，用于日级行为画像的低维表征 ======
class AE(nn.Module):
    def __init__(self, in_dim: int, hid_dims: List[int] = [128, 64], z_dim: int = 16):
        super().__init__()
        enc = []
        last = in_dim
        for h in hid_dims:
            enc += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            last = h
        enc += [nn.Linear(last, z_dim)]
        self.encoder = nn.Sequential(*enc)

        dec = []
        last = z_dim
        for h in reversed(hid_dims):
            dec += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            last = h
        dec += [nn.Linear(last, in_dim)]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def _safe_write_csv(df: pd.DataFrame, final_path: str):
    import os, uuid

    tmp_path = final_path + f".tmp.{uuid.uuid4().hex}"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, final_path)  # 原子替换


def _train_autoencoder(X: np.ndarray, out_dir: str, seg_name: str, epochs: int = 40, batch_size: int = 64, lr: float = 1e-3, z_dim: int = 16, valid_ratio: float = 0.2, device: str = None) -> Tuple[AE, np.ndarray]:
    """
    训练一个小 AE，并记录每轮 train/valid loss，保存 CSV
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    n, d = X.shape
    if n == 0:
        return None, np.zeros((0, z_dim), dtype=np.float32)

    # 标准化
    mu = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    Xn = (X - mu) / std

    # 划分训练集 / 验证集
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * (1 - valid_ratio))
    train_idx, valid_idx = idx[:split], idx[split:]
    X_train, X_valid = Xn[train_idx], Xn[valid_idx]

    model = AE(in_dim=d, z_dim=z_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.MSELoss()

    def to_loader(X):
        tensor = torch.from_numpy(X.astype(np.float32)).to(device)
        ds = torch.utils.data.TensorDataset(tensor)
        return torch.utils.data.DataLoader(ds, batch_size=min(batch_size, len(X)), shuffle=True, drop_last=False)

    train_dl = to_loader(X_train)
    valid_dl = to_loader(X_valid)

    logs = []

    model.train()
    for ep in range(epochs):
        train_loss = 0
        for (xb,) in train_dl:
            x_hat, _ = model(xb)
            loss = crit(x_hat, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train)

        with torch.no_grad():
            valid_loss = 0
            for (xb,) in valid_dl:
                x_hat, _ = model(xb)
                valid_loss += crit(x_hat, xb).item() * len(xb)
            valid_loss /= len(X_valid)

        logs.append({"epoch": ep + 1, "train_loss": train_loss, "valid_loss": valid_loss})
        # print(f"[{seg_name}] Epoch {ep+1}/{epochs}: train={train_loss:.6f}, valid={valid_loss:.6f}")

    # === 保存日志 ===
    os.makedirs(out_dir, exist_ok=True)
    log_df = pd.DataFrame(logs)
    final_csv = os.path.join(out_dir, f"{seg_name}_train_log.csv")
    _safe_write_csv(log_df, final_csv)

    # === 提取嵌入 ===
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(Xn.astype(np.float32)).to(device)
        _, Z = model(tensor)
        Z = Z.cpu().numpy()

    return model, Z


class BatteryAEPredictor(BasePredictor):
    """
    与 BatteryKmeansPredictor 接口一致：
      - fit(user_vector_sequence, vector_granularity)
      - predict_discharge(start_time, last_hist_discharge, target_remaining_mins)
    """

    def __init__(self, n_clusters: int = 10, alpha: float = 0.2, beta: float = 0.8, z_dim: int = 16, ae_epochs: int = 40, ae_batch_size: int = 64, ae_lr: float = 1e-3, log_dir: str = "./ae_train_logs", user_id: str = "unknown"):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta
        self.z_dim = z_dim
        self.ae_epochs = ae_epochs
        self.ae_batch_size = ae_batch_size
        self.ae_lr = ae_lr
        self.log_dir = log_dir
        self.user_id = user_id

    def fit(self, user_vector_sequence: Dict[pd.Timestamp, list], vector_granularity: int) -> None:
        self.vector_granularity = vector_granularity

        daily_vectors = [
            defaultdict(lambda: defaultdict(int)),  # 00-09
            defaultdict(lambda: defaultdict(int)),  # 09-15
            defaultdict(lambda: defaultdict(int)),  # 15-24
        ]
        for ts, entry in user_vector_sequence.items():
            t = ts.time()
            d = ts.date()
            if t < datetime.time(9, 0, 0):
                daily_vectors[0][d] = dict_add(daily_vectors[0][d], entry[0])
            elif datetime.time(9, 0, 0) <= t < datetime.time(15, 0, 0):
                daily_vectors[1][d] = dict_add(daily_vectors[1][d], entry[0])
            else:
                daily_vectors[2][d] = dict_add(daily_vectors[2][d], entry[0])

        all_categories = set()
        for time_vector in daily_vectors:
            all_categories.update(cat for d in time_vector.values() for cat in d if (cat != "Unknown" and cat != "off"))
        all_categories = sorted(all_categories)

        daily_df = []
        for time_vector in daily_vectors:
            df = pd.DataFrame([{"date": date, **time_vector[date]} for date in sorted(time_vector)])
            if "date" in df.columns and len(df) > 0:
                df = df.set_index("date").fillna(0.0)
            else:
                df = pd.DataFrame(columns=["date"]).set_index("date")
            for cat in all_categories:
                if cat not in df.columns:
                    df[cat] = 0.0
            daily_df.append(df[all_categories].sort_index())

        self.segment_models = []
        self.segment_kmeans = []
        self.segment_labels = []

        user_log_dir = os.path.join(self.log_dir, self.user_id)

        for idx, df in enumerate(daily_df):
            seg_name = f"segment{idx}"
            if len(df) == 0:
                labels = pd.Series([0] * 0, index=df.index)
                self.segment_models.append(None)
                self.segment_kmeans.append(None)
                self.segment_labels.append(labels)
                continue

            X = df.values.astype(np.float32)
            _, Z = _train_autoencoder(
                X,
                out_dir=user_log_dir,
                seg_name=seg_name,
                epochs=self.ae_epochs,
                batch_size=self.ae_batch_size,
                lr=self.ae_lr,
                z_dim=self.z_dim,
            )

            kmeans = KMeans(n_clusters=min(self.n_clusters, len(df)), random_state=42)
            labels_np = kmeans.fit_predict(Z)
            labels = pd.Series(labels_np, index=df.index)

            self.segment_models.append(None)
            self.segment_kmeans.append(kmeans)
            self.segment_labels.append(labels)

        discharge_df = pd.DataFrame(
            [(k.date(), k.time(), v[1], v[2]) for k, v in user_vector_sequence.items()],
            columns=["date", "time", "discharge", "is_charging"],
        )
        unique_dates = sorted({ts.date() for ts in user_vector_sequence.keys()})
        self.average_per_granularity_discharge_dict = {}

        for day in unique_dates:
            avg_discharge_by_time_list = []
            for seg_idx, labels in enumerate(self.segment_labels):
                if day in labels.index:
                    cluster_id = labels.loc[day]
                    cluster_dates = labels[labels == cluster_id].index
                else:
                    cluster_dates = labels.index
                valid_train_dates = [d for d in cluster_dates if d != day]
                df_train = discharge_df[discharge_df["date"].isin(valid_train_dates)]
                seg_avg = df_train[df_train["is_charging"] == False].groupby("time")["discharge"].mean()
                avg_discharge_by_time_list.append(seg_avg)

            alldays_avg_discharge_by_time = discharge_df[discharge_df["is_charging"] == False].groupby("time")["discharge"].mean()

            self.average_per_granularity_discharge_dict[day] = (
                avg_discharge_by_time_list,
                alldays_avg_discharge_by_time,
            )

    # === BasePredictor接口 ===
    def _predict(self, start_time: pd.Timestamp, last_hist_discharge: float, opcode: int, input_val: float, ratio: float = 1.0):
        day = start_time.date()
        all_avg = self.average_per_granularity_discharge_dict[day][1]
        if start_time.time() < datetime.time(9, 0, 0):
            avg_by_time = self.average_per_granularity_discharge_dict[day][1]
        elif datetime.time(9, 0, 0) <= start_time.time() < datetime.time(15, 0, 0):
            avg_by_time = self.average_per_granularity_discharge_dict[day][0][0]
        else:
            avg_by_time = self.average_per_granularity_discharge_dict[day][0][1]

        current_time = start_time
        predicted_discharge = []
        discharge = 0.0
        battery_life_mins = 0.0
        remain = float(input_val)

        while remain > 0:
            off_screen = self.vector_granularity * 1.0
            tkey = current_time.time()
            if tkey in avg_by_time:
                hist_avg = float(avg_by_time[tkey])
            elif tkey in all_avg:
                hist_avg = float(all_avg[tkey])
            else:
                hist_avg = off_screen

            if len(predicted_discharge) > 0:
                last_hist_discharge = predicted_discharge[-1]
            pred_t = self.alpha * float(last_hist_discharge) + self.beta * float(hist_avg)
            predicted_discharge.append(pred_t)

            if opcode == 2:
                step = self.vector_granularity
                discharge += pred_t * min(1.0, remain / step)
                remain -= step
            elif opcode == 1:
                step = self.vector_granularity
                if remain - pred_t >= 0:
                    battery_life_mins += step
                else:
                    battery_life_mins += step * (remain / max(pred_t, 1e-6))
                remain -= (pred_t - off_screen) * ratio + off_screen
            current_time += pd.Timedelta(minutes=self.vector_granularity)

        return battery_life_mins if opcode == 1 else discharge

    def predict(self, start_time, last_hist_discharge, opcode, input, ratio=1.0):
        return self._predict(start_time, last_hist_discharge, opcode, input, ratio)

    def predict_battery_life(self, start_time, last_hist_discharge, capacity, ratio=1.0):
        return self._predict(start_time, last_hist_discharge, opcode=1, input_val=capacity, ratio=ratio)

    def predict_discharge(self, start_time, last_hist_discharge, target_remaining_mins):
        return self._predict(start_time, last_hist_discharge, opcode=2, input_val=target_remaining_mins)
