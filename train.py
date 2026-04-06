
"""
FCN-LP + 梯度游走图（GWG）训练脚本（伪标签靶子集 + GSNR-ACR 融合掩码 v3）
  - 靶子集来自 pseudo_labels_output_gpt4o.csv，按置信度 top-K 选取
  - 两次反向传播分别获取 g_sup 和 g_anc，用于 EMA 统计
  - GSNR-ACR 融合 soft mask 作用于监督梯度（与原 GSNR 行为一致）
  - anchor 梯度仅用于计算 EMA 指标，不参与实际梯度更新
  - warmup 期不 mask，积累 EMA
用法: python train.py --dataset pheme
"""

import argparse
import logging
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from model import FCN_LP
from mmd import MMDLoss

# ─── 参数 ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='weibo', choices=['pheme', 'twitter', 'weibo'])
parser.add_argument('--seed',         type=int,   default=42)
parser.add_argument('--epochs',       type=int,   default=6000)
parser.add_argument('--lr',           type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--hidden',       type=int,   default=64)
parser.add_argument('--num_classes',  type=int,   default=2)
parser.add_argument('--dropout',      type=float, default=0.85)
parser.add_argument('--lpaiters',     type=int,   default=2)
parser.add_argument('--gcnnum',       type=int,   default=3)
# GWG 超参
parser.add_argument('--gwg_interval', type=int,   default=30,
                    help='每隔多少 epoch 更新一次梯度游走图')
parser.add_argument('--gwg_warmup',   type=int,   default=30,
                    help='前 N epoch 不构建动态边')
parser.add_argument('--gwg_M',        type=int,   default=3,
                    help='每个训练样本连接的靶子数 M')
parser.add_argument('--gwg_batch',    type=int,   default=64,
                    help='梯度计算 batch size')
parser.add_argument('--target_k',     type=int,   default=20,
                    help='靶子集大小')
# GSNR-ACR 融合掩码超参
parser.add_argument('--mask_warmup',  type=int,   default=300,
                    help='掩码 warmup epoch 数（期间不 mask，仅积累 EMA）')
parser.add_argument('--rho',          type=float, default=0.8,
                    help='EMA 衰减率')
parser.add_argument('--alpha',        type=float, default=0.8,
                    help='GSNR 与 ACR 融合权重 (α*GSNR_norm + (1-α)*ACR_norm)')
parser.add_argument('--mask_percentile', type=float, default=0.5,
                    help='soft mask 百分位阈值 q（Q 的第 q 百分位作为 τ）')
parser.add_argument('--mask_beta',    type=float, default=0.5,
                    help='soft mask sigmoid 温度 β')
parser.add_argument('--log_file',     type=str,   default='',
                    help='日志文件路径，为空则使用默认路径')
args = parser.parse_args()

# ─── 随机种子 ──────────────────────────────────────────────────────────────────
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(args.seed)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ─── 日志 ──────────────────────────────────────────────────────────────────────
def setup_logger(log_file):
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode='a')
        fh.setFormatter(logging.Formatter('%(message)s'))
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

_log_path = args.log_file if args.log_file else f'train_log/{args.dataset}_gwg_pseudo_gsnr_ascr_v3_test.log'
logger = setup_logger(_log_path)
logger.info(f'=== GWG-Pseudo GSNR-ACR v3 | Dataset: {args.dataset} | Device: {device} ===')

# ─── 数据加载 ──────────────────────────────────────────────────────────────────
dataset_name = args.dataset
train_data   = pd.read_csv(f'dataset/{dataset_name}/dataforGCN_train.csv')
test_data    = pd.read_csv(f'dataset/{dataset_name}/dataforGCN_test.csv')
tweet_embeds = torch.load(f'dataset/{dataset_name}/TweetEmbeds.pt')
tweet_graph  = torch.load(f'dataset/{dataset_name}/TweetGraph.pt')

label_list_train = train_data["label"].tolist()
event_list_train = train_data["event"].tolist()
label_list_test  = test_data["label"].tolist()

def to_onehot(label_list):
    t = torch.zeros(len(label_list), 2)
    for j, lbl in enumerate(label_list):
        t[j] = torch.FloatTensor([1., 0.]) if lbl == 1 else torch.FloatTensor([0., 1.])
    return t

labels  = torch.cat([to_onehot(label_list_train), to_onehot(label_list_test)], dim=0)
n_train = len(label_list_train)
n_total = len(labels)

base_edge_index = tweet_graph.coalesce().indices()
base_edge_attr  = tweet_graph.coalesce().values().unsqueeze(-1)

data = Data(
    x          = tweet_embeds.float(),
    edge_index = base_edge_index,
    edge_attr  = base_edge_attr,
    train_mask = torch.tensor([True]*n_train + [False]*(n_total-n_train)).bool(),
    test_mask  = torch.tensor([False]*n_train + [True]*(n_total-n_train)).bool(),
    y          = labels
).to(device)

# ─── seen / unseen 划分 ────────────────────────────────────────────────────────
def get_data_splits(label_list, event_list, sel_events, unsel_events):
    event_map = {}
    for i, (lbl, ev) in enumerate(zip(label_list, event_list)):
        event_map.setdefault(ev, [[], []])
        event_map[ev][0 if lbl == 1 else 1].append(i)
    sr, sf, ur, uf = [], [], [], []
    for ev in sel_events:
        sr.extend(event_map.get(ev, [[], []])[0])
        sf.extend(event_map.get(ev, [[], []])[1])
    for ev in unsel_events:
        ur.extend(event_map.get(ev, [[], []])[0])
        uf.extend(event_map.get(ev, [[], []])[1])
    return sr, sf, ur, uf

if dataset_name == 'weibo':
    all_idx    = list(range(n_train))
    unseen_set = set(random.sample(all_idx, n_train // 3))
    seen_set   = set(all_idx) - unseen_set
    seen_real   = [i for i in seen_set   if label_list_train[i] == 1]
    seen_fake   = [i for i in seen_set   if label_list_train[i] == 0]
    unseen_real = [i for i in unseen_set if label_list_train[i] == 1]
    unseen_fake = [i for i in unseen_set if label_list_train[i] == 0]
elif dataset_name == 'twitter':
    sel   = ['boston','columbianChemicals','nepal','pigFish','bringback',
             'sochi','malaysia','sandy','passport','underwater','livr']
    unsel = ['elephant','garissa','eclipse','samurai']
    seen_real, seen_fake, unseen_real, unseen_fake = get_data_splits(
        label_list_train, event_list_train, sel, unsel)
elif dataset_name == 'pheme':
    sel   = ['Ottawa Shooting','sydney siege','Charlie Hebdo','GermanwingsCrash']
    unsel = ['Ferguson']
    seen_real, seen_fake, unseen_real, unseen_fake = get_data_splits(
        label_list_train, event_list_train, sel, unsel)

seen = seen_real + seen_fake
logger.info(f'seen={len(seen)}, unseen_real={len(unseen_real)}, unseen_fake={len(unseen_fake)}')

# ─── 伪标签靶子集构建 ─────────────────────────────────────────────────────────
def build_target_set_from_pseudo(pseudo_csv_path, test_data_csv_path, n_train, K=20):
    """
    从伪标签 CSV 按置信度排序，选 top-K 测试样本作为靶子。
    自动检测 ID 列名（mid / image_id），通过 join 映射到全局节点索引。
    返回:
        target_indices: list[int]  全局节点索引
        target_pseudo_labels: Tensor [K, 2]  伪标签 one-hot
    """
    pseudo_df = pd.read_csv(pseudo_csv_path)
    test_df   = pd.read_csv(test_data_csv_path)

    # 自动检测 ID 列名
    id_col = None
    for candidate in ['mid', 'image_id']:
        if candidate in pseudo_df.columns and candidate in test_df.columns:
            id_col = candidate
            break
    if id_col is None:
        raise ValueError(
            f'Cannot find a common ID column in pseudo CSV and test CSV. '
            f'pseudo cols={pseudo_df.columns.tolist()}, test cols={test_df.columns.tolist()}'
        )

    pseudo_df[id_col] = pseudo_df[id_col].astype(str)
    test_df[id_col]   = test_df[id_col].astype(str)

    pseudo_df = pseudo_df.sort_values('confidence', ascending=False)
    id_to_local = {row[id_col]: i for i, row in test_df.iterrows()}

    target_indices = []
    target_labels  = []
    for _, row in pseudo_df.iterrows():
        uid = str(row[id_col])
        if uid not in id_to_local:
            continue
        local_i  = id_to_local[uid]
        global_i = n_train + local_i
        pseudo_lbl = int(row['pseudo_label'])
        target_indices.append(global_i)
        target_labels.append([1., 0.] if pseudo_lbl == 1 else [0., 1.])
        if len(target_indices) >= K:
            break

    target_pseudo_labels = torch.tensor(target_labels, dtype=torch.float)
    return target_indices, target_pseudo_labels


target_indices, target_pseudo_labels = build_target_set_from_pseudo(
    pseudo_csv_path=f'dataset/{dataset_name}/pseudo_labels_output_gpt4o.csv',
    test_data_csv_path=f'dataset/{dataset_name}/dataforGCN_test.csv',
    n_train=n_train,
    K=args.target_k
)
target_pseudo_labels = target_pseudo_labels.to(device)
logger.info(
    f'Target set: loaded {len(target_indices)} samples from pseudo_labels CSV (top-K by confidence)'
)

# ─── 模型 ──────────────────────────────────────────────────────────────────────
model = FCN_LP(
    tweet_embeds.shape[1], args.hidden, args.num_classes,
    args.dropout, data.num_edges, args.lpaiters, args.gcnnum
).to(device)

optimizer   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion   = nn.CrossEntropyLoss()
mmd_loss_fn = MMDLoss(kernel_type='linear')

# ─── GSNR-ACR 融合掩码器 ─────────────────────────────────────────────────────

class GSNRACRMasker:
    """
    对每个参数元素维护 4 个 EMA 统计量，融合 GSNR 和 ACR 生成 soft mask。
    mask 作用于监督梯度（与原 GSNR 行为一致），anchor 梯度仅用于 EMA 统计。
    """

    def __init__(self, model, rho=0.95, warmup_epochs=300,
                 alpha=0.9, mask_percentile=1.0, beta=0.5):
        self.model = model
        self.rho = rho
        self.warmup = warmup_epochs
        self.alpha = alpha
        self.percentile = mask_percentile
        self.beta = beta

        self.ema_g_anc = {}
        self.ema_g_anc_sq = {}
        self.ema_sign = {}
        self.ema_abs = {}

        for name, p in model.named_parameters():
            if p.requires_grad:
                self.ema_g_anc[name]    = torch.zeros_like(p.data)
                self.ema_g_anc_sq[name] = torch.zeros_like(p.data)
                self.ema_sign[name]     = torch.zeros_like(p.data)
                self.ema_abs[name]      = torch.zeros_like(p.data)

    def compute_and_apply(self, g_sup_dict, g_anc_dict, epoch):
        """
        更新 EMA，计算融合掩码，将 mask * g_sup 写入 p.grad。
        anchor 梯度仅用于 EMA 统计，不参与实际梯度更新。
        返回 warmup 期: (None, None)；正常期: (tau, masked_ratio)
        """
        rho = self.rho
        eps = 1e-8
        param_dict = dict(self.model.named_parameters())

        # ── Step 1: 更新 EMA 统计量 ──────────────────────────────────────────
        for name in g_anc_dict:
            g_a = g_anc_dict[name]
            self.ema_g_anc[name]    = rho * self.ema_g_anc[name]    + (1 - rho) * g_a
            self.ema_g_anc_sq[name] = rho * self.ema_g_anc_sq[name] + (1 - rho) * g_a * g_a

            if name in g_sup_dict:
                g_s = g_sup_dict[name]
                sign_val = torch.sign(g_s * g_a)
                abs_val  = torch.abs(sign_val)
                self.ema_sign[name] = rho * self.ema_sign[name] + (1 - rho) * sign_val
                self.ema_abs[name]  = rho * self.ema_abs[name]  + (1 - rho) * abs_val

        # warmup 期：不 mask，直接用监督梯度（与原 GSNR warmup 一致）
        if epoch < self.warmup:
            for name, p in param_dict.items():
                if not p.requires_grad:
                    continue
                g_s = g_sup_dict.get(name)
                if g_s is not None:
                    p.grad = g_s.clone()
            return None, None

        # ── Step 2: 计算 GSNR 和 ACR ────────────────────────────────────────
        gsnr_log_map = {}
        ascr_map = {}
        all_gsnr_log = []
        all_ascr = []

        for name in g_anc_dict:
            mu = self.ema_g_anc[name]
            sq = self.ema_g_anc_sq[name]
            var = sq - mu * mu
            gsnr = (mu * mu) / (var + eps)
            gsnr_log = torch.log(gsnr + eps)
            gsnr_log_map[name] = gsnr_log

            s_bar = self.ema_sign[name]
            r_bar = self.ema_abs[name]
            ascr = s_bar / (r_bar + eps)
            ascr_map[name] = ascr

            all_gsnr_log.append(gsnr_log.flatten())
            all_ascr.append(ascr.flatten())

        if not all_gsnr_log:
            for name, p in param_dict.items():
                if p.requires_grad and name in g_sup_dict:
                    p.grad = g_sup_dict[name].clone()
            return None, None

        # ── Step 3: min-max 归一化 ────────────────────────────────────────────
        all_gsnr_log_vec = torch.cat(all_gsnr_log)
        all_ascr_vec     = torch.cat(all_ascr)

        gsnr_min, gsnr_max = all_gsnr_log_vec.min(), all_gsnr_log_vec.max()
        ascr_min, ascr_max = all_ascr_vec.min(), all_ascr_vec.max()

        gsnr_range = gsnr_max - gsnr_min + eps
        ascr_range = ascr_max - ascr_min + eps

        # ── Step 4: 融合 Q 并计算阈值 τ ──────────────────────────────────────
        q_map = {}
        all_q = []
        for name in g_anc_dict:
            gsnr_norm = (gsnr_log_map[name] - gsnr_min) / gsnr_range
            ascr_norm = (ascr_map[name] - ascr_min) / ascr_range
            q = self.alpha * gsnr_norm + (1 - self.alpha) * ascr_norm
            q_map[name] = q
            all_q.append(q.flatten())

        all_q_vec = torch.cat(all_q)
        tau = torch.quantile(all_q_vec.float(), self.percentile / 100.0).item()

        # ── Step 5: 生成 soft mask，作用于监督梯度 ────────────────────────────
        total_params = 0
        masked_params = 0

        for name, p in param_dict.items():
            if not p.requires_grad:
                continue
            g_s = g_sup_dict.get(name)
            if g_s is None:
                continue

            if name in q_map:
                q = q_map[name]
                mask = torch.sigmoid((q - tau) / self.beta)
                p.grad = mask * g_s

                total_params  += mask.numel()
                masked_params += (mask < 0.5).sum().item()
            else:
                p.grad = g_s.clone()

        masked_ratio = masked_params / total_params if total_params > 0 else 0.0
        return tau, masked_ratio


masker = GSNRACRMasker(
    model, rho=args.rho, warmup_epochs=args.mask_warmup,
    alpha=args.alpha, mask_percentile=args.mask_percentile, beta=args.mask_beta
)

# ─── 辅助：收集梯度字典 ──────────────────────────────────────────────────────
def _collect_grads(model):
    gd = {}
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            gd[name] = p.grad.detach().clone()
    return gd

# ─── 梯度游走图核心函数 ────────────────────────────────────────────────────────

def _get_clf_params(model):
    params = []
    for p in model.gc[-1].parameters():
        if p.requires_grad:
            params.append(p)
    for lpn_layer in model.lpn:
        for p in lpn_layer.parameters():
            if p.requires_grad:
                params.append(p)
    return params

def _grad_vec(params):
    vecs = []
    for p in params:
        if p.grad is not None:
            vecs.append(p.grad.detach().float().flatten())
        else:
            vecs.append(torch.zeros(p.numel(), device=p.device))
    return torch.cat(vecs)


def build_gwg_edges_pseudo(model, data, seen_indices, target_indices,
                            target_pseudo_labels, criterion, device,
                            M=3, batch_size=64):
    """
    构建梯度游走图动态边，靶子梯度使用伪标签计算。
    返回:
        dyn_src  : LongTensor [E_dyn]
        dyn_dst  : LongTensor [E_dyn]
        dyn_w    : FloatTensor [E_dyn]
    """
    params = _get_clf_params(model)
    n_targets = len(target_indices)
    tgt_lbl = target_pseudo_labels.to(device)  # [K, 2]

    # Step 1: 逐靶子计算梯度向量（使用伪标签）
    g_targets = []
    for i, t_idx in enumerate(target_indices):
        model.zero_grad()
        out, yhat, _ = model(data)
        t_t = torch.tensor([t_idx], device=device)
        lbl_t = tgt_lbl[i].unsqueeze(0)  # [1, 2]
        loss_t = criterion(out[t_t], lbl_t) + criterion(yhat[t_t], lbl_t)
        loss_t.backward()
        g_targets.append(_grad_vec(params).clone())
    model.zero_grad()

    G_target = torch.stack(g_targets, dim=0)   # [T, D]
    G_target = F.normalize(G_target, dim=1)

    # Step 2: 逐 batch 计算训练样本梯度
    all_src, all_dst, all_w = [], [], []
    n = len(seen_indices)

    for start in range(0, n, batch_size):
        batch = seen_indices[start: start + batch_size]
        batch_idx = torch.tensor(batch, device=device)

        model.zero_grad()
        out, yhat, _ = model(data)
        loss_b = criterion(out[batch_idx], data.y[batch_idx]) + \
                 criterion(yhat[batch_idx], data.y[batch_idx])
        loss_b.backward()
        g_batch = _grad_vec(params).clone()
        model.zero_grad()

        g_batch_norm = F.normalize(g_batch.unsqueeze(0), dim=1)  # [1, D]
        sims = (G_target @ g_batch_norm.T).squeeze(1)            # [T]

        M_eff = min(M, n_targets)
        topk_vals, topk_idx = torch.topk(sims, M_eff)
        weights = F.softmax(topk_vals, dim=0)

        for global_i in batch:
            for m in range(M_eff):
                all_src.append(global_i)
                all_dst.append(target_indices[topk_idx[m].item()])
                all_w.append(weights[m].item())

    if not all_src:
        return None, None, None

    dyn_src = torch.tensor(all_src, dtype=torch.long, device=device)
    dyn_dst = torch.tensor(all_dst, dtype=torch.long, device=device)
    dyn_w   = torch.tensor(all_w,   dtype=torch.float, device=device)
    return dyn_src, dyn_dst, dyn_w


def merge_graph(data_base, dyn_src, dyn_dst, dyn_w, device):
    base_ei = data_base.edge_index
    base_ea = data_base.edge_attr
    dyn_ei  = torch.stack([dyn_src, dyn_dst], dim=0)
    dyn_ea  = dyn_w.unsqueeze(1)
    new_ei  = torch.cat([base_ei, dyn_ei], dim=1)
    new_ea  = torch.cat([base_ea, dyn_ea], dim=0)
    return new_ei, new_ea


def forward_with_ew(model, data_cur, eff_edge_weight):
    x, edge_index, edge_attr = data_cur.x, data_cur.edge_index, data_cur.edge_attr
    x_cur = x
    for i in range(len(model.gc) - 1):
        x_cur = model.gc[i](x_cur, edge_index, eff_edge_weight)
        x_cur = F.relu(x_cur)
        x_cur = F.dropout(x_cur, model.dropout_rate, training=model.training)
    out = model.gc[-1](x_cur, edge_index, eff_edge_weight)
    out = model.softmax(out)
    y_hat = out.detach()
    for i in range(len(model.lpn)):
        y_hat = model.lpn[i](x_cur, edge_index, edge_attr, label=y_hat)
        y_hat = model.softmax(y_hat)
    return out.squeeze(), y_hat.squeeze(), x_cur

def evaluate(output, labels):
    preds  = output.max(1)[1]
    truths = labels.max(1)[1]
    tp = (preds * truths).sum().float()
    fp = (preds * (1 - truths)).sum().float()
    fn = ((1 - preds) * truths).sum().float()
    tn = ((1 - preds) * (1 - truths)).sum().float()
    acc  = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    prec = tp / (tp + fp + 1e-10)
    rec  = tp / (tp + fn + 1e-10)
    f1   = 2 * prec * rec / (prec + rec + 1e-10)
    return acc.item(), prec.item(), rec.item(), f1.item()

# ─── 训练状态 ──────────────────────────────────────────────────────────────────
cur_edge_index = data.edge_index.clone()
cur_edge_attr  = data.edge_attr.clone()
n_base_edges   = data.num_edges

seen_idx_t = torch.tensor(seen, device=device)
tgt_idx_t  = torch.tensor(target_indices, device=device)

max_test_acc = 0.0
best_precision = best_recall = best_f1 = 0.0
best_epoch = 0

# ─── 训练循环 ──────────────────────────────────────────────────────────────────
for epoch in range(args.epochs):
    t0 = time.time()

    # ── 梯度游走图更新（使用伪标签靶子梯度）─────────────────────────────────
    if epoch >= args.gwg_warmup and epoch % args.gwg_interval == 0:
        model.eval()
        dyn_src, dyn_dst, dyn_w = build_gwg_edges_pseudo(
            model, data, seen, target_indices, target_pseudo_labels,
            criterion, device, M=args.gwg_M, batch_size=args.gwg_batch
        )
        if dyn_src is not None:
            cur_edge_index, cur_edge_attr = merge_graph(
                data, dyn_src, dyn_dst, dyn_w, device)
            n_dyn = dyn_src.shape[0]
            logger.info(
                f'Epoch {epoch}: GWG updated — '
                f'base edges={n_base_edges}, dyn edges={n_dyn}, '
                f'total={cur_edge_index.shape[1]}'
            )
        model.train()

    # ── 构造当前 epoch 使用的 data 视图 ──────────────────────────────────────
    data_cur = Data(
        x          = data.x,
        edge_index = cur_edge_index,
        edge_attr  = cur_edge_attr,
        train_mask = data.train_mask,
        test_mask  = data.test_mask,
        y          = data.y
    )

    # ── 构造有效边权重 ───────────────────────────────────────────────────────
    model.train()
    n_cur_edges = cur_edge_index.shape[1]
    if n_cur_edges > n_base_edges:
        dyn_weights = cur_edge_attr[n_base_edges:, 0].detach()
        eff_edge_weight = torch.cat([model.edge_weight, dyn_weights], dim=0)
    else:
        eff_edge_weight = model.edge_weight

    # ══════════════════════════════════════════════════════════════════════════
    # ★ 第一次反向传播：监督梯度 g_sup
    # ══════════════════════════════════════════════════════════════════════════
    optimizer.zero_grad()
    out, yhat, x_feat = forward_with_ew(model, data_cur, eff_edge_weight)

    fcn_loss = criterion(out[seen_idx_t],  data.y[seen_idx_t])
    lpn_loss = criterion(yhat[seen_idx_t], data.y[seen_idx_t])

    mmd_loss = torch.tensor(0.0, device=device)
    if len(unseen_real) > 0 and len(seen_real) > 0:
        mmd_loss = mmd_loss + mmd_loss_fn(x_feat[unseen_real], x_feat[seen_real])
    if len(unseen_fake) > 0 and len(seen_fake) > 0:
        mmd_loss = mmd_loss + mmd_loss_fn(x_feat[unseen_fake], x_feat[seen_fake])

    loss = fcn_loss + lpn_loss + mmd_loss
    loss.backward()
    g_sup_dict = _collect_grads(model)

    # ══════════════════════════════════════════════════════════════════════════
    # ★ 第二次反向传播：anchor 伪标签梯度 g_anc（仅用于 EMA 统计）
    # ══════════════════════════════════════════════════════════════════════════
    optimizer.zero_grad()
    out2, yhat2, _ = forward_with_ew(model, data_cur, eff_edge_weight)

    loss_anc = criterion(out2[tgt_idx_t], target_pseudo_labels) + \
               criterion(yhat2[tgt_idx_t], target_pseudo_labels)
    loss_anc.backward()
    g_anc_dict = _collect_grads(model)

    # ══════════════════════════════════════════════════════════════════════════
    # ★ GSNR-ACR 融合掩码：mask * g_sup 写入 p.grad
    # ══════════════════════════════════════════════════════════════════════════
    optimizer.zero_grad()
    tau, masked_ratio = masker.compute_and_apply(g_sup_dict, g_anc_dict, epoch)

    if tau is not None and epoch % 50 == 0:
        logger.info(
            f'Epoch {epoch}: GSNR-ACR tau={tau:.4f}, '
            f'masked={masked_ratio*100:.1f}%'
        )

    optimizer.step()

    # ── 评估 ──────────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        n_cur_edges = cur_edge_index.shape[1]
        if n_cur_edges > n_base_edges:
            dyn_weights = cur_edge_attr[n_base_edges:, 0].detach()
            eff_ew = torch.cat([model.edge_weight, dyn_weights], dim=0)
        else:
            eff_ew = model.edge_weight
        out_e, yhat_e, _ = forward_with_ew(model, data_cur, eff_ew)

    acc_train, _, _, _ = evaluate(yhat_e[data.train_mask], data.y[data.train_mask])
    acc_test, prec_test, rec_test, f1_test = evaluate(
        yhat_e[data.test_mask], data.y[data.test_mask])

    logger.info(
        f'Epoch: {epoch+1:04d}, loss={loss.item():.4f}, '
        f'acc_train={acc_train:.4f}, '
        f'Test Acc={acc_test:.4f}, Prec={prec_test:.4f}, '
        f'Rec={rec_test:.4f}, F1={f1_test:.4f}, '
        f'time={time.time()-t0:.2f}s'
    )

    if acc_test > max_test_acc:
        max_test_acc   = acc_test
        best_precision = prec_test
        best_recall    = rec_test
        best_f1        = f1_test
        best_epoch     = epoch + 1

logger.info(
    f'\n=== Best Result === Epoch {best_epoch}, '
    f'Acc={max_test_acc:.4f}, Prec={best_precision:.4f}, '
    f'Rec={best_recall:.4f}, F1={best_f1:.4f}'
)
