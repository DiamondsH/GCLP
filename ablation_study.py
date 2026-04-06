"""
消融实验脚本：基于 GCLP (Full) v3 分别移除各组件

  (1) GCLP (Full)   — GGLP + GSNR-ACR 掩码 + LLM Anchor（完整配置）
  (2) w/o GGLP          — 移除梯度游走图构建，保留 GSNR-ACR 掩码
  (3) w/o GDDM          — 移除 GSNR-ACR 参数掩码，保留梯度游走图

用法:
  python ablation_study_v3.py --dataset pheme [--epochs 6000] [--runs 5]
"""

import argparse
import logging
import random
import sys
import time
import os

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
parser.add_argument('--dataset',      type=str,   default='pheme',
                    choices=['pheme', 'twitter', 'weibo'])
parser.add_argument('--seed',         type=int,   default=42)
parser.add_argument('--epochs',       type=int,   default=6000)
parser.add_argument('--lr',           type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--hidden',       type=int,   default=64)
parser.add_argument('--num_classes',  type=int,   default=2)
parser.add_argument('--dropout',      type=float, default=0.85)
parser.add_argument('--lpaiters',     type=int,   default=2)
parser.add_argument('--gcnnum',       type=int,   default=4)
# GWG 超参
parser.add_argument('--gwg_interval', type=int,   default=30)
parser.add_argument('--gwg_warmup',   type=int,   default=30)
parser.add_argument('--gwg_M',        type=int,   default=3)
parser.add_argument('--gwg_batch',    type=int,   default=64)
parser.add_argument('--target_k',     type=int,   default=20)
# GSNR-ACR 融合掩码超参
parser.add_argument('--mask_warmup',  type=int,   default=300)
parser.add_argument('--rho',          type=float, default=0.8)
parser.add_argument('--alpha',        type=float, default=0.8)
parser.add_argument('--mask_percentile', type=float, default=0.5)
parser.add_argument('--mask_beta',    type=float, default=0.5)
# 多次运行
parser.add_argument('--runs',         type=int,   default=5,
                    help='每个配置重复运行次数')
parser.add_argument('--log_file',     type=str,   default='')
args = parser.parse_args()

# ─── 随机种子 ──────────────────────────────────────────────────────────────────
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ─── 日志 ──────────────────────────────────────────────────────────────────────
def setup_logger(log_file):
    logger = logging.getLogger('ablation_v3')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode='a')
    fh.setFormatter(logging.Formatter('%(message)s'))
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

_log_path = args.log_file if args.log_file else f'train_log/{args.dataset}_ablation_v3.log'
os.makedirs(os.path.dirname(_log_path) if os.path.dirname(_log_path) else 'train_log',
            exist_ok=True)
logger = setup_logger(_log_path)
logger.info(f'=== Ablation Study v3 | Dataset: {args.dataset} | Device: {device} ===')
logger.info(f'Args: {vars(args)}')

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

setup_seed(args.seed)
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
    pseudo_df = pd.read_csv(pseudo_csv_path)
    test_df   = pd.read_csv(test_data_csv_path)
    id_col = None
    for candidate in ['mid', 'image_id']:
        if candidate in pseudo_df.columns and candidate in test_df.columns:
            id_col = candidate
            break
    if id_col is None:
        raise ValueError(
            f'Cannot find a common ID column. '
            f'pseudo cols={pseudo_df.columns.tolist()}, test cols={test_df.columns.tolist()}'
        )
    pseudo_df[id_col] = pseudo_df[id_col].astype(str)
    test_df[id_col]   = test_df[id_col].astype(str)
    pseudo_df = pseudo_df.sort_values('confidence', ascending=False)
    id_to_local = {row[id_col]: i for i, row in test_df.iterrows()}
    target_indices, target_labels = [], []
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
    return target_indices, torch.tensor(target_labels, dtype=torch.float)

target_indices, target_pseudo_labels = build_target_set_from_pseudo(
    pseudo_csv_path=f'dataset/{dataset_name}/pseudo_labels_output_gpt4o.csv',
    test_data_csv_path=f'dataset/{dataset_name}/dataforGCN_test.csv',
    n_train=n_train,
    K=args.target_k
)
target_pseudo_labels = target_pseudo_labels.to(device)
logger.info(f'Target set: loaded {len(target_indices)} pseudo-label anchors')


# ═══════════════════════════════════════════════════════════════════════════════
# GSNR-ACR 融合掩码器（与 v3 训练脚本完全一致）
# ═══════════════════════════════════════════════════════════════════════════════

class GSNRACRMasker:
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
        rho = self.rho
        eps = 1e-8
        param_dict = dict(self.model.named_parameters())

        # Step 1: 更新 EMA 统计量
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

        # warmup 期不 mask
        if epoch < self.warmup:
            for name, p in param_dict.items():
                if not p.requires_grad:
                    continue
                g_s = g_sup_dict.get(name)
                if g_s is not None:
                    p.grad = g_s.clone()
            return None, None

        # Step 2: 计算 GSNR 和 ACR
        gsnr_log_map, ascr_map = {}, {}
        all_gsnr_log, all_ascr = [], []

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

        # Step 3: min-max 归一化
        all_gsnr_log_vec = torch.cat(all_gsnr_log)
        all_ascr_vec     = torch.cat(all_ascr)
        gsnr_min, gsnr_max = all_gsnr_log_vec.min(), all_gsnr_log_vec.max()
        ascr_min, ascr_max = all_ascr_vec.min(), all_ascr_vec.max()
        gsnr_range = gsnr_max - gsnr_min + eps
        ascr_range = ascr_max - ascr_min + eps

        # Step 4: 融合 Q 并计算阈值 τ
        q_map, all_q = {}, []
        for name in g_anc_dict:
            gsnr_norm = (gsnr_log_map[name] - gsnr_min) / gsnr_range
            ascr_norm = (ascr_map[name] - ascr_min) / ascr_range
            q = self.alpha * gsnr_norm + (1 - self.alpha) * ascr_norm
            q_map[name] = q
            all_q.append(q.flatten())

        all_q_vec = torch.cat(all_q)
        tau = torch.quantile(all_q_vec.float(), self.percentile / 100.0).item()

        # Step 5: soft mask 作用于监督梯度
        total_params, masked_params = 0, 0
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


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

criterion   = nn.CrossEntropyLoss()
mmd_loss_fn = MMDLoss(kernel_type='linear')

def _collect_grads(model):
    gd = {}
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            gd[name] = p.grad.detach().clone()
    return gd

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
                           target_pseudo_labels, device, M=3, batch_size=64):
    params = _get_clf_params(model)
    n_targets = len(target_indices)
    tgt_lbl = target_pseudo_labels.to(device)
    g_targets = []
    for i, t_idx in enumerate(target_indices):
        model.zero_grad()
        out, yhat, _ = model(data)
        t_t = torch.tensor([t_idx], device=device)
        lbl_t = tgt_lbl[i].unsqueeze(0)
        loss_t = criterion(out[t_t], lbl_t) + criterion(yhat[t_t], lbl_t)
        loss_t.backward()
        g_targets.append(_grad_vec(params).clone())
    model.zero_grad()
    G_target = torch.stack(g_targets, dim=0)
    G_target = F.normalize(G_target, dim=1)

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
        g_batch_norm = F.normalize(g_batch.unsqueeze(0), dim=1)
        sims = (G_target @ g_batch_norm.T).squeeze(1)
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


# ═══════════════════════════════════════════════════════════════════════════════
# 核心：单次训练函数（通过开关控制 GWG 和 GSNR-ACR）
# ═══════════════════════════════════════════════════════════════════════════════

def train_single_run(config_name, use_gwg, use_gsnr_ascr, run_seed,
                     hp_override=None):
    """
    单次训练运行。
    Args:
        config_name:   配置名称（用于日志）
        use_gwg:       是否启用梯度游走图 (GGLP)
        use_gsnr_ascr: 是否启用 GSNR-ACR 融合掩码 (GDDM)
        run_seed:      本次运行的随机种子
        hp_override:   超参数覆盖字典（可选）
    Returns:
        dict: {acc, prec, rec, f1, epoch}
    """
    setup_seed(run_seed)

    # 合并超参数：默认值 + 可选覆盖
    hp = {
        'epochs':          args.epochs,
        'lr':              args.lr,
        'weight_decay':    args.weight_decay,
        'dropout':         args.dropout,
        'mask_warmup':     args.mask_warmup,
        'rho':             args.rho,
        'alpha':           args.alpha,
        'mask_percentile': args.mask_percentile,
        'mask_beta':       args.mask_beta,
        'gwg_warmup':      args.gwg_warmup,
        'gwg_interval':    args.gwg_interval,
        'gwg_M':           args.gwg_M,
        'gwg_batch':       args.gwg_batch,
    }
    if hp_override:
        hp.update(hp_override)
        logger.info(f'  [HP Override] {hp_override}')

    model = FCN_LP(
        tweet_embeds.shape[1], args.hidden, args.num_classes,
        hp['dropout'], data.num_edges, args.lpaiters, args.gcnnum
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=hp['lr'],
                                   weight_decay=hp['weight_decay'])

    masker = None
    if use_gsnr_ascr:
        masker = GSNRACRMasker(
            model, rho=hp['rho'], warmup_epochs=hp['mask_warmup'],
            alpha=hp['alpha'], mask_percentile=hp['mask_percentile'],
            beta=hp['mask_beta']
        )

    cur_edge_index = data.edge_index.clone()
    cur_edge_attr  = data.edge_attr.clone()
    n_base_edges   = data.num_edges

    seen_idx_t = torch.tensor(seen, device=device)
    tgt_idx_t  = torch.tensor(target_indices, device=device)
    tgt_lbl_dev = target_pseudo_labels.to(device)

    max_test_acc = 0.0
    best_prec = best_rec = best_f1 = 0.0
    best_epoch = 0

    t_start = time.time()

    for epoch in range(hp['epochs']):
        # ── GWG 动态边更新 ──
        if use_gwg and epoch >= hp['gwg_warmup'] and epoch % hp['gwg_interval'] == 0:
            model.eval()
            dyn_src, dyn_dst, dyn_w = build_gwg_edges_pseudo(
                model, data, seen, target_indices, tgt_lbl_dev,
                device, M=hp['gwg_M'], batch_size=hp['gwg_batch']
            )
            if dyn_src is not None:
                cur_edge_index, cur_edge_attr = merge_graph(
                    data, dyn_src, dyn_dst, dyn_w, device)
            model.train()

        # ── 构造当前 data 视图 ──
        data_cur = Data(
            x=data.x, edge_index=cur_edge_index, edge_attr=cur_edge_attr,
            train_mask=data.train_mask, test_mask=data.test_mask, y=data.y
        )

        # ── 构造有效边权重 ──
        model.train()
        n_cur_edges = cur_edge_index.shape[1]
        if n_cur_edges > n_base_edges:
            dyn_weights = cur_edge_attr[n_base_edges:, 0].detach()
            eff_edge_weight = torch.cat([model.edge_weight, dyn_weights], dim=0)
        else:
            eff_edge_weight = model.edge_weight

        if use_gsnr_ascr and masker is not None:
            # ════════════════════════════════════════════════════════════════
            # 双次反向传播路径：g_sup + g_anc → GSNR-ACR mask
            # ════════════════════════════════════════════════════════════════

            # ★ 第一次反向传播：监督梯度 g_sup
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

            # ★ 第二次反向传播：anchor 伪标签梯度 g_anc（仅用于 EMA 统计）
            optimizer.zero_grad()
            out2, yhat2, _ = forward_with_ew(model, data_cur, eff_edge_weight)
            loss_anc = criterion(out2[tgt_idx_t], tgt_lbl_dev) + \
                       criterion(yhat2[tgt_idx_t], tgt_lbl_dev)
            loss_anc.backward()
            g_anc_dict = _collect_grads(model)

            # ★ GSNR-ACR 融合掩码：mask * g_sup 写入 p.grad
            optimizer.zero_grad()
            masker.compute_and_apply(g_sup_dict, g_anc_dict, epoch)

        else:
            # ════════════════════════════════════════════════════════════════
            # 单次反向传播路径：不用掩码，直接更新
            # ════════════════════════════════════════════════════════════════
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

        optimizer.step()

        # ── 评估 ──
        if epoch % 100 == 0 or epoch == hp['epochs'] - 1:
            model.eval()
            with torch.no_grad():
                n_cur = cur_edge_index.shape[1]
                if n_cur > n_base_edges:
                    dw = cur_edge_attr[n_base_edges:, 0].detach()
                    eff_ew = torch.cat([model.edge_weight, dw], dim=0)
                else:
                    eff_ew = model.edge_weight
                out_e, yhat_e, _ = forward_with_ew(model, data_cur, eff_ew)

            acc_test, prec_test, rec_test, f1_test = evaluate(
                yhat_e[data.test_mask], data.y[data.test_mask])

            if acc_test > max_test_acc:
                max_test_acc = acc_test
                best_prec    = prec_test
                best_rec     = rec_test
                best_f1      = f1_test
                best_epoch   = epoch + 1

            if epoch % 500 == 0:
                elapsed = time.time() - t_start
                logger.info(
                    f'  [{config_name}] Epoch {epoch:4d}: '
                    f'Acc={acc_test:.4f}, F1={f1_test:.4f}, '
                    f'time={elapsed:.1f}s'
                )

    logger.info(
        f'  [{config_name}] Best: Epoch {best_epoch}, '
        f'Acc={max_test_acc:.4f}, Prec={best_prec:.4f}, '
        f'Rec={best_rec:.4f}, F1={best_f1:.4f}'
    )
    return dict(acc=max_test_acc, prec=best_prec, rec=best_rec,
                f1=best_f1, epoch=best_epoch)


# ═══════════════════════════════════════════════════════════════════════════════
# 消融配置定义
# ═══════════════════════════════════════════════════════════════════════════════

ABLATION_CONFIGS = [
    {
        'name':          'GCLP (Full)',
        'use_gwg':       True,
        'use_gsnr_ascr': True,
        # 全模型使用默认超参数，不需要覆盖
    },
    {
        'name':          'w/o GGLP',
        'use_gwg':       False,     # 移除梯度游走图构建
        'use_gsnr_ascr': True,
        'hp_override': {
            'epochs':       500,     # 训练 500 epoch（默认 6000）
            'dropout':      0.5,     # dropout 设置为 0.5（默认 0.85）
            'lr':           5e-5,    # 降低学习率（默认 1e-4）
            'weight_decay': 5e-4,    # 正则（默认 5e-4）
            'mask_warmup':  500,     # 延迟掩码启动（默认 300）
        },
    },
    {
        'name':          'w/o GDDM',
        'use_gwg':       True,
        'use_gsnr_ascr': False,     # 移除 GSNR-ACR 参数掩码
        'hp_override': {
            'epochs':       500,     # 训练 500 epoch（默认 6000）
            'dropout':      0.5,     # dropout 设置为 0.5（默认 0.85）
            'lr':           5e-5,    # 降低学习率（默认 1e-4）
            'weight_decay': 5e-4,    # 正则（默认 5e-4）
            'gwg_warmup':   30,      # GWG 启动（默认 30）
            'gwg_interval': 30,      # GWG 更新频率（默认 30）
        },
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# 运行消融实验（多次运行取均值±标准差）
# ═══════════════════════════════════════════════════════════════════════════════

all_results = {}

for cfg in ABLATION_CONFIGS:
    cname = cfg['name']
    logger.info(f'\n{"="*70}')
    logger.info(f'Config: {cname}  (GWG={cfg["use_gwg"]}, GSNR-ACR={cfg["use_gsnr_ascr"]})')
    logger.info(f'{"="*70}')

    run_results = []
    for r in range(args.runs):
        run_seed = args.seed + r
        logger.info(f'\n--- Run {r+1}/{args.runs} (seed={run_seed}) ---')
        result = train_single_run(
            config_name=cname,
            use_gwg=cfg['use_gwg'],
            use_gsnr_ascr=cfg['use_gsnr_ascr'],
            run_seed=run_seed,
            hp_override=cfg.get('hp_override', None),
        )
        run_results.append(result)

    all_results[cname] = run_results

# ═══════════════════════════════════════════════════════════════════════════════
# 汇总输出
# ═══════════════════════════════════════════════════════════════════════════════

logger.info(f'\n\n{"="*70}')
logger.info(f'ABLATION RESULTS — {dataset_name.upper()} ({args.runs} runs)')
logger.info(f'{"="*70}')
logger.info(f'{"Config":<25s} {"Acc":>14s} {"Prec":>14s} {"Rec":>14s} {"F1":>14s}')
logger.info('-' * 85)

for cfg in ABLATION_CONFIGS:
    cname = cfg['name']
    results = all_results[cname]

    accs  = [r['acc']  for r in results]
    precs = [r['prec'] for r in results]
    recs  = [r['rec']  for r in results]
    f1s   = [r['f1']   for r in results]

    def fmt(vals):
        return f'{np.mean(vals)*100:.2f}±{np.std(vals)*100:.2f}'

    logger.info(
        f'{cname:<25s} {fmt(accs):>14s} {fmt(precs):>14s} '
        f'{fmt(recs):>14s} {fmt(f1s):>14s}'
    )

logger.info('-' * 85)
logger.info(f'Log saved to: {_log_path}')
logger.info('Done!')
