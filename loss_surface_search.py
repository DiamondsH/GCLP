"""
loss_surface_hp_search.py

对 loss 曲面可视化的计算超参进行网格搜索，寻找最优参数组合：
  - 最大化 GRAVITAS (Full) 与 w/o GDDM 的曲面形状差异（归一化 L2）
  - 同时最小化 Full 模型曲面的粗糙度（Total Variation）
  目标函数: score = diff(Full, w/o GDDM) − λ × roughness(Full)

前置条件：
  先运行 ablation_study_v3_plot_loss.py（需加 --ckpt_dir 参数）生成两个 checkpoint 文件：
    {ckpt_dir}/{dataset}_GRAVITAS_Full_ckpt.pt
    {ckpt_dir}/{dataset}_wo_GDDM_ckpt.pt

用法：
  python loss_surface_hp_search.py --dataset pheme \\
      --ckpt_dir loss_surface_ckpts --out_dir hp_search_results

搜索参数（逗号分隔字符串）：
  --gs_grid_size   网格分辨率候选值（整数）
  --gs_lr_scale    PCA 扰动幅度缩放候选值（浮点）
  --gs_axis_range  扰动轴半径候选值（浮点，最终轴为 linspace(-v, v, N)）
  --gs_lam         目标函数平滑度惩罚系数候选值（浮点）
"""

import argparse
import itertools
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from torch_geometric.data import Data

from model import FCN_LP
from mmd import MMDLoss


# ─── 命令行参数 ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Loss 曲面可视化超参网格搜索')
parser.add_argument('--dataset',     type=str, default='pheme',
                    choices=['pheme', 'twitter', 'weibo'])
parser.add_argument('--ckpt_dir',    type=str, default='loss_surface_ckpts',
                    help='checkpoint 目录（由 ablation_study_v3_plot_loss.py 生成）')
parser.add_argument('--out_dir',     type=str, default='hp_search_results_loss',
                    help='搜索结果输出目录')
parser.add_argument('--num_windows', type=int, default=3,
                    help='用于评估的最后 N 个梯度窗口，-1 表示全部')
parser.add_argument('--topk',        type=int, default=5,
                    help='可视化输出的 Top-K 结果数量')
# 搜索空间
parser.add_argument('--gs_grid_size',  type=str, default='10,15,20,30,40',
                    help='网格分辨率候选值（整数，逗号分隔）')
parser.add_argument('--gs_lr_scale',   type=str, default='1.0,5.0,10.0,20.0,50.0',
                    help='PCA 扰动幅度缩放候选值（浮点，逗号分隔）')
parser.add_argument('--gs_axis_range', type=str, default='0.5,1.0,1.5,2.0,3.0',
                    help='扰动轴半径候选值（浮点，逗号分隔）')
parser.add_argument('--gs_lam',        type=str, default='0.1,0.3,0.5,0.7,1.0',
                    help='平滑度惩罚系数候选值（浮点，逗号分隔）')
args = parser.parse_args()

# ─── 设备 ─────────────────────────────────────────────────────────────────────
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# ─── 日志 ─────────────────────────────────────────────────────────────────────
os.makedirs(args.out_dir, exist_ok=True)
log_path = os.path.join(args.out_dir, f'{args.dataset}_hp_search.log')
logger = logging.getLogger('hp_search')
logger.setLevel(logging.INFO)
logger.handlers.clear()
fh = logging.FileHandler(log_path, mode='w')
fh.setFormatter(logging.Formatter('%(message)s'))
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(fh)
logger.addHandler(ch)

# ─── 搜索空间解析 ──────────────────────────────────────────────────────────────
def _parse_floats(s): return [float(x.strip()) for x in s.split(',')]
def _parse_ints(s):   return [int(x.strip())   for x in s.split(',')]

SEARCH_SPACE = {
    'grid_size':  _parse_ints(args.gs_grid_size),
    'lr_scale':   _parse_floats(args.gs_lr_scale),
    'axis_range': _parse_floats(args.gs_axis_range),
    'lam':        _parse_floats(args.gs_lam),
}
total_combos = 1
for v in SEARCH_SPACE.values():
    total_combos *= len(v)

logger.info(f'=== Loss 曲面超参网格搜索 | Dataset: {args.dataset} | Device: {device} ===')
logger.info(f'搜索空间: {SEARCH_SPACE}')
logger.info(f'总组合数: {total_combos}')

# ─── Checkpoint 加载 ───────────────────────────────────────────────────────────
_CKPT_NAMES = {
    'GRAVITAS (Full)': 'GRAVITAS_Full',
    'w/o GDDM':        'wo_GDDM',
}

def load_ckpt(config_name: str) -> dict:
    safe  = _CKPT_NAMES[config_name]
    path  = os.path.join(args.ckpt_dir, f'{args.dataset}_{safe}_ckpt.pt')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Checkpoint 不存在: {path}\n'
            f'请先运行: python ablation_study_v3_plot_loss.py '
            f'--dataset {args.dataset} --ckpt_dir {args.ckpt_dir}'
        )
    ckpt = torch.load(path, map_location='cpu')
    logger.info(f'  已加载: {path}  '
                f'({len(ckpt["grads_windows"])} 窗口, config="{ckpt["config"]}")')
    return ckpt

logger.info('\n加载 Checkpoint...')
ckpt_full = load_ckpt('GRAVITAS (Full)')
ckpt_gddm = load_ckpt('w/o GDDM')

# ─── 数据加载（用于 forward pass）─────────────────────────────────────────────
dataset_name = args.dataset
logger.info(f'\n加载数据集: {dataset_name}')

train_data   = pd.read_csv(f'dataset/{dataset_name}/dataforGCN_train.csv')
test_data    = pd.read_csv(f'dataset/{dataset_name}/dataforGCN_test.csv')
tweet_embeds = torch.load(f'dataset/{dataset_name}/TweetEmbeds.pt')
tweet_graph  = torch.load(f'dataset/{dataset_name}/TweetGraph.pt')

label_list_train = train_data['label'].tolist()
label_list_test  = test_data['label'].tolist()
n_train = len(label_list_train)

def _to_onehot(label_list):
    t = torch.zeros(len(label_list), 2)
    for j, lbl in enumerate(label_list):
        t[j] = torch.FloatTensor([1., 0.]) if lbl == 1 else torch.FloatTensor([0., 1.])
    return t

labels  = torch.cat([_to_onehot(label_list_train), _to_onehot(label_list_test)], dim=0)
n_total = len(labels)

base_edge_index = tweet_graph.coalesce().indices()
base_edge_attr  = tweet_graph.coalesce().values().unsqueeze(-1)

data = Data(
    x          = tweet_embeds.float(),
    edge_index = base_edge_index,
    edge_attr  = base_edge_attr,
    train_mask = torch.tensor([True]*n_train  + [False]*(n_total - n_train)).bool(),
    test_mask  = torch.tensor([False]*n_train + [True]*(n_total - n_train)).bool(),
    y          = labels
).to(device)

# ─── 从 Full checkpoint 还原数据划分索引 ──────────────────────────────────────
seen_real   = ckpt_full['seen_real']
seen_fake   = ckpt_full['seen_fake']
unseen_real = ckpt_full['unseen_real']
unseen_fake = ckpt_full['unseen_fake']
seen        = ckpt_full['seen_idx']
logger.info(f'数据划分: seen={len(seen)}, '
            f'unseen_real={len(unseen_real)}, unseen_fake={len(unseen_fake)}')

# ─── 损失 & MMD ───────────────────────────────────────────────────────────────
criterion   = nn.CrossEntropyLoss()
mmd_loss_fn = MMDLoss(kernel_type='linear')

# ─── 辅助函数（与原脚本保持一致）──────────────────────────────────────────────
def _get_clf_param_names(model):
    last_idx = len(model.gc) - 1
    names = set()
    for n, p in model.named_parameters():
        if p.requires_grad and (
            n.startswith(f'gc.{last_idx}.') or n.startswith('lpn.')
        ):
            names.add(n)
    return names


def forward_with_ew(model, data_cur, eff_edge_weight):
    x, edge_index, edge_attr = data_cur.x, data_cur.edge_index, data_cur.edge_attr
    x_cur = x
    for i in range(len(model.gc) - 1):
        x_cur = model.gc[i](x_cur, edge_index, eff_edge_weight)
        x_cur = F.relu(x_cur)
        x_cur = F.dropout(x_cur, model.dropout_rate, training=model.training)
    out   = model.gc[-1](x_cur, edge_index, eff_edge_weight)
    out   = model.softmax(out)
    y_hat = out.detach()
    for i in range(len(model.lpn)):
        y_hat = model.lpn[i](x_cur, edge_index, edge_attr, label=y_hat)
        y_hat = model.softmax(y_hat)
    return out.squeeze(), y_hat.squeeze(), x_cur


# ─── 模型加载 ──────────────────────────────────────────────────────────────────
def load_model(ckpt: dict) -> FCN_LP:
    kw    = ckpt['model_init_kwargs']
    model = FCN_LP(
        kw['in_channels'], kw['hidden'], kw['num_classes'],
        kw['dropout'],     kw['num_edges'],
        kw['lpaiters'],    kw['gcnnum']
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model

logger.info('\n加载模型...')
model_full     = load_model(ckpt_full)
model_gddm     = load_model(ckpt_gddm)
clf_names_full = _get_clf_param_names(model_full)
clf_names_gddm = _get_clf_param_names(model_gddm)
logger.info('模型加载完毕。')


# ─── Loss 曲面计算（带超参版本）────────────────────────────────────────────────
def compute_loss_grid_hp(model, clf_names, data_cur, eff_edge_weight,
                          seen_idx_t, grads_list,
                          grid_size, lr_scale, axis_range):
    """
    对 grads_list 做 PCA，沿两主方向扰动分类器参数后在网格上计算 loss。
    grid_size, lr_scale, axis_range 均为可调超参。
    返回 loss_grid (np.ndarray, shape=(grid_size, grid_size))；失败时返回 None。
    """
    if len(grads_list) < 2:
        return None

    G = torch.stack(grads_list, dim=0).cpu().numpy()
    try:
        pca = PCA(n_components=2)
        pca.fit(G)
        d1 = torch.tensor(pca.components_[0], dtype=torch.float32)
        d2 = torch.tensor(pca.components_[1], dtype=torch.float32)
    except Exception:
        return None

    # 将方向向量按参数形状分块
    orig_vals = {}
    param_map  = {}
    for name, p in model.named_parameters():
        if name in clf_names:
            orig_vals[name] = p.data.clone()
            param_map[name] = p

    def split_vec(vec):
        slices, offset = {}, 0
        for name, p in model.named_parameters():
            if name in clf_names:
                sz = p.numel()
                slices[name] = vec[offset: offset + sz].reshape(p.shape).to(p.device)
                offset += sz
        return slices

    d1_s = split_vec(d1)
    d2_s = split_vec(d2)

    ax_vals   = np.linspace(-axis_range, axis_range, grid_size)
    loss_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for i, a in enumerate(ax_vals):
            for j, b in enumerate(ax_vals):
                for name, p in param_map.items():
                    p.data = (orig_vals[name]
                              + float(a) * d1_s[name] * lr_scale
                              + float(b) * d2_s[name] * lr_scale)
                out, yhat, x_feat = forward_with_ew(model, data_cur, eff_edge_weight)
                fcn_l = criterion(out[seen_idx_t],  data_cur.y[seen_idx_t])
                lpn_l = criterion(yhat[seen_idx_t], data_cur.y[seen_idx_t])
                mmd_l = torch.tensor(0.0, device=device)
                if len(unseen_real) > 0 and len(seen_real) > 0:
                    mmd_l += mmd_loss_fn(x_feat[unseen_real], x_feat[seen_real])
                if len(unseen_fake) > 0 and len(seen_fake) > 0:
                    mmd_l += mmd_loss_fn(x_feat[unseen_fake], x_feat[seen_fake])
                loss_grid[i, j] = (fcn_l + lpn_l + mmd_l).item()

    # 恢复原始参数
    for name, p in param_map.items():
        p.data = orig_vals[name]

    return loss_grid


# ─── 评价指标 ──────────────────────────────────────────────────────────────────
def surface_roughness_tv(Z: np.ndarray) -> float:
    """Total Variation 粗糙度（越小越平滑）"""
    dh = np.diff(Z, axis=0)
    dv = np.diff(Z, axis=1)
    return float(np.mean(np.abs(dh)) + np.mean(np.abs(dv)))


def surface_difference_norml2(Z1: np.ndarray, Z2: np.ndarray) -> float:
    """归一化 L2 曲面形状差异（越大曲面形状越不同）"""
    Z1n = (Z1 - Z1.min()) / (Z1.max() - Z1.min() + 1e-8)
    Z2n = (Z2 - Z2.min()) / (Z2.max() - Z2.min() + 1e-8)
    return float(np.mean((Z1n - Z2n) ** 2))


def compute_objective(Z_full: np.ndarray, Z_gddm: np.ndarray,
                       lam: float) -> tuple:
    """
    目标函数: score = diff(Full, w/o GDDM) − λ × roughness(Full)
    返回 (score, diff, roughness)
    """
    diff      = surface_difference_norml2(Z_full, Z_gddm)
    roughness = surface_roughness_tv(Z_full)
    score     = diff - lam * roughness
    return score, diff, roughness


# ─── 窗口选择 ──────────────────────────────────────────────────────────────────
def select_windows(windows: list, num: int) -> list:
    """取最后 num 个窗口（-1 表示全部）"""
    return windows if num == -1 else windows[-num:]


# ─── 主网格搜索 ────────────────────────────────────────────────────────────────
def run_grid_search() -> pd.DataFrame:
    seen_idx_t   = torch.tensor(seen, device=device)
    windows_full = select_windows(ckpt_full['grads_windows'], args.num_windows)
    windows_gddm = select_windows(ckpt_gddm['grads_windows'], args.num_windows)
    n_pairs      = min(len(windows_full), len(windows_gddm))

    if n_pairs == 0:
        raise RuntimeError('没有可用的梯度窗口，请检查 checkpoint 文件。')

    logger.info(f'\n使用梯度窗口数: {n_pairs} '
                f'(Full: {len(windows_full)}, w/o GDDM: {len(windows_gddm)})')
    logger.info(f'开始网格搜索（共 {total_combos} 组合）...\n')

    # 每组搜索完立即保存图片的目录
    surfaces_dir = os.path.join(args.out_dir, 'surfaces')
    os.makedirs(surfaces_dir, exist_ok=True)
    epoch = ckpt_full['grads_windows'][-1]['epoch']   # 用于图标题

    combos  = list(itertools.product(
        SEARCH_SPACE['grid_size'],
        SEARCH_SPACE['lr_scale'],
        SEARCH_SPACE['axis_range'],
        SEARCH_SPACE['lam'],
    ))
    records = []
    t0      = time.time()
    log_interval = max(1, total_combos // 20)

    for ci, (grid_size, lr_scale, axis_range, lam) in enumerate(combos):
        scores, diffs, roughs = [], [], []
        Z_fulls, Z_gddms      = [], []   # 存储每窗口的曲面，用于平均后绘图

        for wi in range(n_pairs):
            wf = windows_full[wi]
            wg = windows_gddm[wi]

            # 构造当前窗口的图视图
            data_cur_full = Data(
                x=data.x,
                edge_index=wf['edge_index'].to(device),
                edge_attr=wf['edge_attr'].to(device),
                train_mask=data.train_mask,
                test_mask=data.test_mask,
                y=data.y
            )
            data_cur_gddm = Data(
                x=data.x,
                edge_index=wg['edge_index'].to(device),
                edge_attr=wg['edge_attr'].to(device),
                train_mask=data.train_mask,
                test_mask=data.test_mask,
                y=data.y
            )
            eff_ew_full = wf['eff_edge_weight'].to(device)
            eff_ew_gddm = wg['eff_edge_weight'].to(device)

            grads_f = [g.to(device) for g in wf['grads']]
            grads_g = [g.to(device) for g in wg['grads']]

            Z_full = compute_loss_grid_hp(
                model_full, clf_names_full,
                data_cur_full, eff_ew_full, seen_idx_t, grads_f,
                grid_size, lr_scale, axis_range
            )
            Z_gddm = compute_loss_grid_hp(
                model_gddm, clf_names_gddm,
                data_cur_gddm, eff_ew_gddm, seen_idx_t, grads_g,
                grid_size, lr_scale, axis_range
            )

            if Z_full is None or Z_gddm is None:
                continue

            sc, diff, rough = compute_objective(Z_full, Z_gddm, lam)
            scores.append(sc)
            diffs.append(diff)
            roughs.append(rough)
            Z_fulls.append(Z_full)
            Z_gddms.append(Z_gddm)

        if not scores:
            continue

        Z_full_avg = np.mean(Z_fulls, axis=0)
        Z_gddm_avg = np.mean(Z_gddms, axis=0)

        records.append({
            'grid_size':  grid_size,
            'lr_scale':   lr_scale,
            'axis_range': axis_range,
            'lam':        lam,
            'score_mean': float(np.mean(scores)),
            'score_std':  float(np.std(scores)),
            'diff_mean':  float(np.mean(diffs)),
            'rough_mean': float(np.mean(roughs)),
            'Z_full':     Z_full_avg,   # 多窗口平均曲面
            'Z_gddm':     Z_gddm_avg,
            'combo_idx':  ci,           # 用于关联已保存的 PNG
        })

        # ── 立即保存本组曲面图片 ──
        ax_vals = np.linspace(-axis_range, axis_range, grid_size)
        fig = plt.figure(figsize=(16, 7), facecolor='none')
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        _plot_surface_pair(ax1, ax2, Z_full_avg, Z_gddm_avg, ax_vals, epoch)
        tag = '[Default] ' if (grid_size == 20 and lr_scale == 10.0 and axis_range == 1.0) else ''
        fig.suptitle(
            f'{dataset_name} | {tag}[#{ci+1:04d}] '
            f'grid={grid_size}, lr_s={lr_scale:.1f}, ax={axis_range:.1f}, λ={lam:.1f} '
            f'| score={records[-1]["score_mean"]:.4f} '
            f'(diff={records[-1]["diff_mean"]:.4f}, rough={records[-1]["rough_mean"]:.4f})',
            fontsize=10
        )
        png_path = os.path.join(surfaces_dir, f'combo_{ci:04d}.png')
        fig.savefig(png_path, dpi=100, bbox_inches='tight', facecolor='none', transparent=True)
        plt.close(fig)

        if (ci + 1) % log_interval == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (ci + 1) * (total_combos - ci - 1)
            rec     = records[-1]
            logger.info(
                f'  [{ci+1:4d}/{total_combos}] '
                f'grid={grid_size:2d} lr_s={lr_scale:5.1f} '
                f'ax={axis_range:.1f} λ={lam:.1f} '
                f'→ score={rec["score_mean"]:.4f} '
                f'(diff={rec["diff_mean"]:.4f}, rough={rec["rough_mean"]:.4f}) '
                f'ETA {eta:.0f}s'
            )

    logger.info(f'\n{len(records)} 组曲面图片已保存至: {surfaces_dir}')
    df = pd.DataFrame(records).sort_values('score_mean', ascending=False).reset_index(drop=True)
    return df


# ─── 结果保存 ──────────────────────────────────────────────────────────────────
def save_results(df: pd.DataFrame) -> pd.DataFrame:
    # CSV 只保存数值列（Z_full/Z_gddm 是 numpy array，不写入）
    numeric_cols = ['grid_size', 'lr_scale', 'axis_range', 'lam',
                    'score_mean', 'score_std', 'diff_mean', 'rough_mean']
    csv_path = os.path.join(args.out_dir, f'{dataset_name}_hp_search_results.csv')
    df[numeric_cols].to_csv(csv_path, index=False)
    logger.info(f'\nCSV 保存 → {csv_path}')

    # Top-K 表格
    logger.info(f'\n{"="*80}')
    logger.info(f'Top-{args.topk} 超参组合（按 score_mean 降序）')
    logger.info(f'{"="*80}')
    logger.info(df[numeric_cols].head(args.topk).to_string(index=True))

    # 默认参数排名
    mask = (
        (df['grid_size']  == 20)  &
        (df['lr_scale']   == 10.0) &
        (df['axis_range'] == 1.0)
    )
    default_rows = df[mask]
    if not default_rows.empty:
        best_default_rank = default_rows.index[0] + 1
        best_default_row  = default_rows.iloc[0]
        logger.info(
            f'\n默认参数 (grid=20, lr_s=10.0, ax=1.0) 中最佳 λ 的排名: '
            f'#{best_default_rank}  '
            f'(λ={best_default_row["lam"]:.1f}, '
            f'score={best_default_row["score_mean"]:.4f})'
        )

    return df   # 返回完整 df，供绘图使用


def _plot_surface_pair(ax1, ax2, Z_full, Z_gddm, ax_vals, epoch):
    A, B = np.meshgrid(ax_vals, ax_vals)
    for ax, Z, label in [(ax1, Z_full, 'GRAVITAS (Full)'),
                          (ax2, Z_gddm, 'w/o GDDM')]:
        ax.set_facecolor('none')
        ax.plot_surface(A, B, Z, cmap='coolwarm', edgecolor='none')
        ax.grid(True)
        for axinfo in [ax.xaxis._axinfo, ax.yaxis._axinfo, ax.zaxis._axinfo]:
            axinfo['grid'].update(color='gray', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Beta')
        ax.set_zlabel('Loss')
        ax.set_title(f'{label} | Epoch {epoch}', fontsize=9)


def plot_all_surfaces(df: pd.DataFrame):
    """
    将全部组合的曲面按 score 排序合并为一个 PDF。
    直接使用搜索时已计算的平均曲面，无需重新计算。
    """
    epoch        = ckpt_full['grads_windows'][-1]['epoch']
    surfaces_dir = os.path.join(args.out_dir, 'surfaces')
    pdf_path     = os.path.join(args.out_dir, f'{dataset_name}_all_surfaces.pdf')

    with PdfPages(pdf_path) as pdf:
        for rank, row in df.iterrows():
            gs  = int(row['grid_size'])
            ls  = float(row['lr_scale'])
            ar  = float(row['axis_range'])
            lam = float(row['lam'])
            ci  = int(row['combo_idx'])

            tag = '[Default] ' if (gs == 20 and ls == 10.0 and ar == 1.0) else ''

            fig = plt.figure(figsize=(16, 7), facecolor='none')
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')
            _plot_surface_pair(ax1, ax2, row['Z_full'], row['Z_gddm'],
                               np.linspace(-ar, ar, gs), epoch)
            fig.suptitle(
                f'{dataset_name} | {tag}[Rank#{rank+1}] '
                f'grid={gs}, lr_s={ls:.1f}, ax={ar:.1f}, λ={lam:.1f} '
                f'| score={row["score_mean"]:.4f} '
                f'(diff={row["diff_mean"]:.4f}, rough={row["rough_mean"]:.4f}) '
                f'[PNG: combo_{ci:04d}.png]',
                fontsize=9
            )
            pdf.savefig(fig, dpi=150, bbox_inches='tight')
            plt.close(fig)

    logger.info(f'汇总 PDF（按 score 排序，共 {len(df)} 页）→ {pdf_path}')
    logger.info(f'单张图片目录 → {surfaces_dir}')


# ─── 入口 ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df = run_grid_search()
    df = save_results(df)
    plot_all_surfaces(df)
    logger.info(f'\n搜索完成！结果保存至: {args.out_dir}')
