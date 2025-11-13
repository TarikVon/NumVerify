import os
import torch
import torch.nn.functional as F
from datetime import datetime
from typing import Any, List, Tuple
from joblib import Parallel, delayed
from torch.nn import MultiheadAttention
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GatedGraphConv
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm

from numvalkit.core import BasePredictor
from numvalkit.data_loader import ForegroundLoader
from sklearn.model_selection import train_test_split


class TimeAwareGGNN(torch.nn.Module):
    def __init__(self, num_items, embed_dim, hidden_dim, num_classes, ggnn_layers=2):
        super().__init__()
        # 1. 应用 ID 嵌入：将稀疏的 pkg_idx 转为低维稠密向量，捕捉应用语义相似性
        self.embed = torch.nn.Embedding(num_items, embed_dim)
        # 2. 输入投影层：将 [embed_dim + 2] 的原始特征（ID 嵌入 + 时间 + 停留时长）映射到隐藏维度
        self.input_lin = torch.nn.Linear(embed_dim + 2, hidden_dim)
        # 3. GNN 层：使用 GatedGraphConv 在图上做消息传递，聚合邻居信息
        #    hidden_dim 输入输出相同，迭代 ggnn_layers 次来扩展感受野
        self.ggnn = GatedGraphConv(hidden_dim, ggnn_layers)
        # 4. 自注意力层：在同一图内部对节点序列做注意力，强化关键节点的影响
        #    batch_first=True 方便直接处理 (batch_size, seq_len, hidden_dim)
        self.att = MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        # 5. 输出线性层：将最终的图表示映射到类别空间，做下一个应用的分类
        self.lin = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        # a) 从 data.node_idx（形状 [sum_N]）取嵌入，输出 [sum_N, embed_dim]
        x_emb = self.embed(data.node_idx)
        # b) 拼接时间特征和停留时长特征，扩展维度到 embed_dim + 2
        #    data.time_feat 和 data.dwell_feat 都是 [sum_N, 1]
        #    dwell_feat是duration的特征
        x = torch.cat([x_emb, data.time_feat, data.dwell_feat], dim=1)
        # c) 输入投影 + 非线性，将原始特征映射到隐藏空间 [sum_N, hidden_dim]
        x = F.relu(self.input_lin(x))
        # d) 在图结构上做 GNN 消息传递，输出同维度 [sum_N, hidden_dim]
        #    利用 edge_index 聚合前后节点信息，捕捉序列图的局部依赖
        x = self.ggnn(x, data.edge_index)

        # e) 将稀疏节点表示恢复为 batch 内部的密集矩阵
        #    x_batch: [batch_size, max_seq_len, hidden_dim]
        #    mask:    [batch_size, max_seq_len]（True 表示该位置有效）
        x_batch, mask = to_dense_batch(x, data.batch)

        # f) 自注意力：让节点之间能跨步互相影响，强化全局上下文
        #    key_padding_mask 使用 ~mask 来屏蔽填充位置
        att_out, _ = self.att(x_batch, x_batch, x_batch, key_padding_mask=~mask)

        # g) 取每个图（会话）中最后一个实际节点的输出，作为该会话的整体表示
        #    lengths: 每个图的有效节点数 [batch_size]
        lengths = mask.sum(dim=1)
        #    last_idx: 最后一个节点的索引（0-based）
        last_idx = lengths - 1
        #    batch_id: [0,1,…,batch_size-1]
        batch_id = torch.arange(x_batch.size(0), device=lengths.device)
        #    final: [batch_size, hidden_dim]
        final = att_out[batch_id, last_idx]

        # h) 输出层：对 final 做线性变换，得到对每个应用类别的打分 logits
        return self.lin(final)


class BehaveGGNNPredictor(BasePredictor):
    """
    基于 TimeAwareGGNN 的下一个应用预测器
    """

    def __init__(
        self,
        data_dir: str,
        user_list: List[str],
        device: str = "cpu",
        embed_dim: int = 64,
        hidden_dim: int = 64,
        ggnn_layers: int = 2,
        batch_size: int = 32,
        lr: float = 1e-3,
        epochs: int = 10,
    ):
        """
        :param embed_dim:  传给 TimeAwareGGNN 的嵌入维度
        :param hidden_dim: 传给 TimeAwareGGNN 的隐藏维度
        :param ggnn_steps: 传给 TimeAwareGGNN 的迭代层数
        """
        self.data_dir = data_dir
        self.loader = ForegroundLoader(data_dir)
        self.user_list = user_list
        self.device = torch.device(device)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.ggnn_layers = ggnn_layers
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs

        self.pkg2idx = {}
        self.idx2pkg = {}
        self.model = None

    def _build_vocab(self) -> None:
        """
        扫描所有用户并行地提取包名，然后合并成全局词典。
        """

        def _load_pkgs_for_user(u: str):
            recs = self.loader.load(u, True) or []
            return recs

        all_recs = Parallel(n_jobs=-1, backend="loky")(
            delayed(_load_pkgs_for_user)(u)
            for u in tqdm(self.user_list, desc="Building vocab scanning users")
        )
        package_set = set(pkg for recs in all_recs for _, _, pkg in recs)
        self.pkg2idx = {p: i for i, p in enumerate(sorted(package_set))}
        self.idx2pkg = {i: p for p, i in self.pkg2idx.items()}

    def _make_dataset(self, dataset_user_list) -> Tuple[List[Data], List[Data]]:
        """
        Parallelized construction of the session‐graph dataset.
        Returns (train_list, val_list).
        """

        def build_sessions(recs, gap=30 * 60):
            """Split a user's flat record list into sessions by time gap."""
            sessions, cur = [], []
            if not recs:
                return sessions
            cur = [recs[0]]
            for r in recs[1:]:
                if (r[0] - cur[-1][1]).total_seconds() <= gap:
                    cur.append(r)
                else:
                    sessions.append(cur)
                    cur = [r]
            sessions.append(cur)
            return sessions

        def process_user(u):
            """Convert one user's sessions into a list of Data graphs."""
            recs = self.loader.load(u) or []
            sessions = build_sessions(recs)
            user_data = []
            for sess in sessions:
                if len(sess) < 2:
                    continue
                pkg_seq = [self.pkg2idx[p] for _, _, p in sess]
                time_feats = [
                    ((s.hour * 3600 + s.minute * 60 + s.second) / (24 * 3600))
                    for s, _, _ in sess
                ]
                dwell = [(e - s).total_seconds() / 3600 for s, e, _ in sess]

                N = len(pkg_seq) - 1
                src = list(range(N - 1))
                dst = list(range(1, N))
                edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

                node_idx = torch.tensor(pkg_seq[:-1], dtype=torch.long)
                time_feat = torch.tensor(time_feats[:-1], dtype=torch.float).unsqueeze(
                    1
                )
                dwell_feat = torch.tensor(dwell[:-1], dtype=torch.float).unsqueeze(1)
                y = torch.tensor(pkg_seq[-1], dtype=torch.long)

                user_data.append(
                    Data(
                        node_idx=node_idx,
                        time_feat=time_feat,
                        dwell_feat=dwell_feat,
                        edge_index=edge_index,
                        y=y,
                    )
                )
            return user_data

        # parallel map over all users
        results = Parallel(n_jobs=-1, backend="loky", verbose=5)(
            delayed(process_user)(u)
            for u in tqdm(dataset_user_list, desc="Building dataset")
        )

        # flatten list of lists
        data_list = [d for user_list in results for d in user_list]

        # train/val split
        train_data, val_data = train_test_split(
            data_list, test_size=0.2, random_state=42
        )
        return train_data, val_data

    def fit(self, users: List[str] = None) -> None:
        """
        训练模型，将最终模型保存在 self.model
        """
        users = users or self.user_list
        self._build_vocab()
        train_data, val_data = self._make_dataset(users)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size)

        # 初始化模型
        num_items = len(self.pkg2idx)
        self.model = TimeAwareGGNN(
            num_items, self.embed_dim, self.hidden_dim, self.ggnn_layers
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()

        # 训练循环
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                out = self.model(batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs
            avg_loss = total_loss / len(train_loader.dataset)
            # 简单验证
            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    pred = self.model(batch).argmax(dim=1)
                    correct += (pred == batch.y).sum().item()
                    total += batch.num_graphs
            print(f"[Epoch {epoch}] loss={avg_loss:.4f} val_acc={correct/total:.4f}")

    def predict(
        self, user: str, session: List[Tuple[datetime, datetime, str]], top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        给定单个用户当前会话，返回 top_k 预测结果。
        """
        # 把 session 转成 Data 格式
        pkg_seq = [self.pkg2idx.get(p, -1) for _, _, p in session]
        # 如果出现新包，丢弃或映射到 <unk>
        pkg_seq = [i for i in pkg_seq if i >= 0]
        if len(pkg_seq) < 1:
            return []
        time_feats = [
            ((s.hour * 3600 + s.minute * 60 + s.second) / (24 * 3600))
            for s, _, _ in session
        ]
        dwell = [(e - s).total_seconds() / 3600 for s, e, _ in session]
        N = len(pkg_seq)
        src = list(range(N - 1))
        dst = list(range(1, N))
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long).to(
            self.device
        )

        node_idx = torch.tensor(pkg_seq, dtype=torch.long).unsqueeze(0).to(self.device)
        time_feat = (
            torch.tensor(time_feats, dtype=torch.float)
            .unsqueeze(0)
            .unsqueeze(2)
            .to(self.device)
        )
        dwell_feat = (
            torch.tensor(dwell, dtype=torch.float)
            .unsqueeze(0)
            .unsqueeze(2)
            .to(self.device)
        )
        batch = Data(
            node_idx=node_idx.squeeze(0),
            time_feat=time_feat.squeeze(0),
            dwell_feat=dwell_feat.squeeze(0),
            edge_index=edge_index,
            y=None,
        )
        batch.batch = torch.zeros(
            batch.node_idx.size(0), dtype=torch.long, device=self.device
        )

        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                batch.unsqueeze(0) if hasattr(batch, "unsqueeze") else batch
            )
            probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
        top_idx = probs.argsort()[-top_k:][::-1]
        return [(self.idx2pkg[i], float(probs[i])) for i in top_idx]

    def save(self, path: str):
        """
        保存模型参数到指定路径（仅 state_dict）。
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """
        从指定路径加载模型参数。会自动重建 vocab 并初始化模型。
        """
        # 1. 重建映射
        self._build_vocab()
        print(f"Finish build vocab, size {len(self.pkg2idx)}")

        # 2. 初始化模型
        num_items = len(self.pkg2idx)
        self.model = TimeAwareGGNN(
            num_items=num_items,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_classes=num_items,
            ggnn_layers=self.ggnn_layers,
        ).to(self.device)

        # 3. 加载参数
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"Model loaded from {path}")
