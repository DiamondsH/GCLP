"""
UMAP 独立超参搜索：为 Naive MLP、w/o GGLP 和 Full Model 分别搜索最优 UMAP 超参。

三个模型的流形结构不同，共用超参无法同时最优化各方的可视化效果。
本脚本对每个模型独立遍历扩大后的 (n_neighbors, min_dist, metric) 搜索空间，
按 Silhouette Score 排名，最终用各自最优超参组合生成论文级对比图。

支持数据集: pheme, weibo, twitter

用法:
    python umap_independent_search.py [--dataset pheme] [--seed 42] [--device cuda:0]

输出:
    umap_ind_search/{dataset}/
    ├── naive_mlp/                      # Naive MLP 单模型搜索结果
    │   ├── gallery_{metric}.pdf        #   缩略总览
    │   └── ranking.csv                 #   排名表
    ├── wo_gglp/                        # w/o GGLP 单模型搜索结果
    │   ├── gallery_{metric}.pdf
    │   └── ranking.csv
    ├── full_model/                     # Full Model 单模型搜索结果
    │   ├── gallery_{metric}.pdf
    │   └── ranking.csv
    ├── pairs/                          # 每组超参的 1×3 三模型对比图
    │   └── nn{n}_md{d}_m{metric}.pdf
    ├── best_independent.pdf            # 各自最优超参的 1×3 论文对比图
    └── summary.csv                     # 三模型全部结果合并表
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import silhouette_score
import umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D

from model import FCN_LP

# ═════════════════════════════════════════════════════════════════════════════
# 学术论文全局样式
# ═════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    'font.family':          'serif',
    'font.serif':           ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset':     'stix',
    'font.size':            10,
    'axes.linewidth':       0.8,
    'axes.labelsize':       11,
    'axes.titlesize':       12,
    'axes.titleweight':     'bold',
    'xtick.major.width':    0.6,
    'ytick.major.width':    0.6,
    'xtick.minor.width':    0.4,
    'ytick.minor.width':    0.4,
    'xtick.labelsize':      9,
    'ytick.labelsize':      9,
    'xtick.direction':      'in',
    'ytick.direction':      'in',
    'legend.fontsize':      9,
    'legend.framealpha':    0.9,
    'legend.edgecolor':     '0.6',
    'legend.handletextpad': 0.4,
    'figure.dpi':           150,
    'savefig.dpi':          300,
    'savefig.bbox':         'tight',
    'savefig.pad_inches':   0.05,
})

COLOR_REAL = '#2166AC'
COLOR_FAKE = '#D6604D'

# ─── 参数 ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='twitter',
                    choices=['pheme', 'weibo', 'twitter'],
                    help='数据集名称: pheme | weibo | twitter')
parser.add_argument('--seed',   type=int, default=42)
parser.add_argument('--device', type=str,
                    default='cuda:0' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

device       = args.device
seed         = args.seed
dataset_name = args.dataset
dataset_tag  = dataset_name.upper()   # 用于图表标题，如 PHEME / WEIBO / TWITTER

# ═════════════════════════════════════════════════════════════════════════════
# 扩大后的搜索空间（三维：n_neighbors × min_dist × metric）
# ═════════════════════════════════════════════════════════════════════════════

N_NEIGHBORS_LIST = [5, 10, 15, 20, 30, 50, 100]
MIN_DIST_LIST    = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
METRIC_LIST      = ['euclidean', 'cosine', 'correlation']

# ─── 数据加载 ──────────────────────────────────────────────────────────────────
train_data   = pd.read_csv(f'dataset/{dataset_name}/dataforGCN_train.csv')
test_data    = pd.read_csv(f'dataset/{dataset_name}/dataforGCN_test.csv')
tweet_embeds = torch.load(f'dataset/{dataset_name}/TweetEmbeds.pt',
                          map_location='cpu')
tweet_graph  = torch.load(f'dataset/{dataset_name}/TweetGraph.pt',
                          map_location='cpu')

label_list_train = train_data["label"].tolist()
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
    train_mask = torch.tensor([True]*n_train + [False]*(n_total - n_train)).bool(),
    test_mask  = torch.tensor([False]*n_train + [True]*(n_total - n_train)).bool(),
    y          = labels,
).to(device)

test_labels = np.array(label_list_test)
real_mask = test_labels == 1
fake_mask = test_labels == 0


# ─── 模型加载与特征提取 ────────────────────────────────────────────────────────
def _forward_extract(model, data_cur, eff_edge_weight):
    x = data_cur.x
    edge_index = data_cur.edge_index
    for i in range(len(model.gc) - 1):
        x = model.gc[i](x, edge_index, eff_edge_weight)
        x = F.relu(x)
        x = F.dropout(x, model.dropout_rate, training=model.training)
    return x


# ─── Naive MLP 定义（与 naive_mlp_pheme.py 保持一致）────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),      # 0
            nn.ReLU(),                       # 1
            nn.Dropout(dropout),             # 2
            nn.Linear(hidden, hidden // 2), # 3
            nn.ReLU(),                       # 4
            nn.Dropout(dropout),             # 5
            nn.Linear(hidden // 2, num_classes),  # 6
        )

    def forward(self, x):
        return self.net(x)

    def get_embedding(self, x):
        """提取倒数第二层激活（第二个 ReLU 之后，分类层之前）."""
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == 4:   # 第二个 ReLU 输出，维度 = hidden // 2
                break
        return x


def load_naive_mlp_and_extract(ckpt_path, tweet_embeds_cpu, n_train, device):
    """加载 Naive MLP checkpoint，返回测试集的中间层 embeddings."""
    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt['args']

    model = MLP(
        in_dim=tweet_embeds_cpu.shape[1],
        hidden=saved_args['hidden'],
        num_classes=2,
        dropout=saved_args['dropout'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    X_test = tweet_embeds_cpu[n_train:].float().to(device)
    with torch.no_grad():
        embeddings = model.get_embedding(X_test).cpu().numpy()

    info = {
        'config': 'Naive MLP',
        'acc':    ckpt.get('test_acc', 0),
        'f1':     ckpt.get('test_f1',  0),
    }
    print(f"  [{info['config']}] Acc={info['acc']:.4f}, "
          f"F1={info['f1']:.4f}, shape={embeddings.shape}")
    return embeddings, info


def load_model_and_extract(ckpt_path, data, device):
    ckpt       = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt['args']
    hp         = ckpt['hp']

    model = FCN_LP(
        tweet_embeds.shape[1], saved_args['hidden'], saved_args['num_classes'],
        hp['dropout'], data.num_edges, saved_args['lpaiters'], saved_args['gcnnum'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    has_dynamic = 'cur_edge_index' in ckpt and ckpt['cur_edge_index'] is not None

    with torch.no_grad():
        if has_dynamic:
            cur_edge_index = ckpt['cur_edge_index'].to(device)
            cur_edge_attr  = ckpt['cur_edge_attr'].to(device)
            n_base = data.num_edges
            n_cur  = cur_edge_index.shape[1]
            if n_cur > n_base:
                dyn_w  = cur_edge_attr[n_base:, 0].detach()
                eff_ew = torch.cat([model.edge_weight, dyn_w], dim=0)
            else:
                eff_ew = model.edge_weight
            data_cur = Data(
                x=data.x, edge_index=cur_edge_index, edge_attr=cur_edge_attr,
                train_mask=data.train_mask, test_mask=data.test_mask, y=data.y,
            )
            x_feat = _forward_extract(model, data_cur, eff_ew)
        else:
            _, _, x_feat = model(data)

    embeddings = x_feat[data.test_mask].cpu().numpy()
    info = {
        'config': ckpt.get('config_name', 'unknown'),
        'acc':    ckpt.get('test_acc', 0),
        'f1':     ckpt.get('f1', 0),
    }
    print(f"  [{info['config']}] Acc={info['acc']:.4f}, "
          f"F1={info['f1']:.4f}, shape={embeddings.shape}")
    return embeddings, info


# ─── 提取 embeddings（只需一次）─────────────────────────────────────────────
print("=" * 70)
print("Extracting model embeddings ...")
emb_naive, info_naive = load_naive_mlp_and_extract(
    f'checkpoints/{dataset_name}_naive_mlp_best.pt',
    tweet_embeds.cpu(), n_train, device)
emb_wo,   info_wo   = load_model_and_extract(
    f'checkpoints/ablation/{dataset_name}_w_o_GGLP_seed{seed}_best.pt', data, device)
emb_full, info_full = load_model_and_extract(
    f'checkpoints/ablation/{dataset_name}_GRAVITAS_Full_seed{seed}_best.pt', data, device)
print("=" * 70)


# ═════════════════════════════════════════════════════════════════════════════
# 绘图工具
# ═════════════════════════════════════════════════════════════════════════════

def draw_panel(ax, umap_2d, labels, title, sil, param_text=None):
    """学术风格单面板散点图."""
    rm = labels == 1
    fm = labels == 0

    ax.scatter(umap_2d[fm, 0], umap_2d[fm, 1],
               c=COLOR_FAKE, label='Fake', s=12, alpha=0.55,
               edgecolors='none', rasterized=True)
    ax.scatter(umap_2d[rm, 0], umap_2d[rm, 1],
               c=COLOR_REAL, label='Real', s=12, alpha=0.55,
               edgecolors='none', rasterized=True)

    ax.set_title(title, pad=6)

    # Silhouette 标注（右下）
    ax.text(0.97, 0.03, f'Silhouette = {sil:.3f}',
            transform=ax.transAxes, fontsize=8.5,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec='0.7', alpha=0.85))

    # 超参标注（左下）
    if param_text:
        ax.text(0.03, 0.03, param_text,
                transform=ax.transAxes, fontsize=7,
                ha='left', va='bottom', color='0.35',
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec='0.8', alpha=0.7))

    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(loc='upper left', markerscale=1.5, handletextpad=0.3)


def draw_thumbnail(ax, umap_2d, sil):
    """Gallery 缩略小图（无轴标签）."""
    ax.scatter(umap_2d[fake_mask, 0], umap_2d[fake_mask, 1],
               c=COLOR_FAKE, s=2, alpha=0.4, edgecolors='none',
               rasterized=True)
    ax.scatter(umap_2d[real_mask, 0], umap_2d[real_mask, 1],
               c=COLOR_REAL, s=2, alpha=0.4, edgecolors='none',
               rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.02, f'{sil:.3f}', transform=ax.transAxes,
            fontsize=6.5, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.12', fc='white',
                      ec='0.7', alpha=0.8))


def make_tag(nn, md, metric):
    md_str = str(md).replace('.', '')
    m_abbr = metric[:3]
    return f"nn{nn}_md{md_str}_m{m_abbr}"


# ═════════════════════════════════════════════════════════════════════════════
# 单模型搜索函数
# ═════════════════════════════════════════════════════════════════════════════

def search_single_model(emb, model_name, short_name, out_sub):
    """对单个模型的 embeddings 遍历所有超参组合，返回排名 DataFrame."""
    os.makedirs(out_sub, exist_ok=True)
    combos = list(itertools.product(N_NEIGHBORS_LIST, MIN_DIST_LIST, METRIC_LIST))
    records = []
    umap_results = {}  # 缓存 UMAP 结果

    print(f"\n{'─'*70}")
    print(f"  Searching {model_name}  ({len(combos)} combinations)")
    print(f"{'─'*70}")

    for idx, (nn, md, metric) in enumerate(combos, 1):
        tag = make_tag(nn, md, metric)
        print(f"  [{idx:3d}/{len(combos)}] nn={nn:3d} md={md:.2f} metric={metric:<12s} ",
              end='', flush=True)

        reducer = umap.UMAP(n_neighbors=nn, min_dist=md, metric=metric,
                            random_state=42)
        u = reducer.fit_transform(emb)
        sil = silhouette_score(u, test_labels)

        umap_results[(nn, md, metric)] = u
        records.append({
            'n_neighbors': nn,
            'min_dist':    md,
            'metric':      metric,
            'silhouette':  round(sil, 4),
        })
        print(f"Sil={sil:.4f}")

    # ── 排名 CSV ──
    df = pd.DataFrame(records).sort_values('silhouette', ascending=False)
    df.insert(0, 'rank', range(1, len(df) + 1))
    df.to_csv(os.path.join(out_sub, 'ranking.csv'), index=False)

    # ── Gallery 总览（按 metric 分组，行=nn，列=md）──
    for metric in METRIC_LIST:
        n_rows = len(N_NEIGHBORS_LIST)
        n_cols = len(MIN_DIST_LIST)
        fig, axes_g = plt.subplots(n_rows, n_cols,
                                   figsize=(n_cols * 1.8, n_rows * 1.5))
        for ri, nn in enumerate(N_NEIGHBORS_LIST):
            for ci, md in enumerate(MIN_DIST_LIST):
                ax = axes_g[ri, ci]
                u = umap_results[(nn, md, metric)]
                sil = silhouette_score(u, test_labels)
                draw_thumbnail(ax, u, sil)
                if ri == 0:
                    ax.set_title(f'd={md}', fontsize=7, pad=3)
            axes_g[ri, 0].set_ylabel(f'nn={nn}', fontsize=7,
                                     fontweight='bold', rotation=0,
                                     labelpad=24, va='center')

        fig.suptitle(f'{short_name} — metric={metric}',
                     fontsize=10, fontweight='bold', y=1.01)

        legend_el = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=COLOR_REAL, markersize=5, label='Real'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=COLOR_FAKE, markersize=5, label='Fake'),
        ]
        fig.legend(handles=legend_el, loc='upper right', fontsize=7,
                   bbox_to_anchor=(0.99, 0.99))
        fig.tight_layout(rect=[0.05, 0.0, 1.0, 0.97], h_pad=0.6, w_pad=0.4)
        fig.savefig(os.path.join(out_sub, f'gallery_{metric}.pdf'),
                    format='pdf')
        plt.close(fig)

    # 打印 Top-5
    print(f"\n  Top-5 for {model_name}:")
    top5 = df.head(5)
    for _, row in top5.iterrows():
        print(f"    #{int(row['rank']):d}  nn={int(row['n_neighbors']):3d}  "
              f"md={row['min_dist']:.2f}  metric={row['metric']:<12s}  "
              f"Sil={row['silhouette']:.4f}")

    return df, umap_results


# ═════════════════════════════════════════════════════════════════════════════
# 执行独立搜索
# ═════════════════════════════════════════════════════════════════════════════

out_root = os.path.join('umap_ind_search', dataset_name)
os.makedirs(out_root, exist_ok=True)

df_naive, cache_naive = search_single_model(
    emb_naive, 'Naive MLP', 'Naive MLP',
    os.path.join(out_root, 'naive_mlp'))

df_wo, cache_wo = search_single_model(
    emb_wo, 'w/o GGLP', 'w/o GGLP',
    os.path.join(out_root, 'wo_gglp'))

df_full, cache_full = search_single_model(
    emb_full, 'Full Model (GRAVITAS)', 'Full',
    os.path.join(out_root, 'full_model'))


# ═════════════════════════════════════════════════════════════════════════════
# 每组超参 → 1×3 三模型对比图
# ═════════════════════════════════════════════════════════════════════════════

pairs_dir = os.path.join(out_root, 'pairs')
os.makedirs(pairs_dir, exist_ok=True)

combos_all = list(itertools.product(N_NEIGHBORS_LIST, MIN_DIST_LIST, METRIC_LIST))
sil_naive_map = {(int(r['n_neighbors']), float(r['min_dist']), r['metric']): r['silhouette']
                 for _, r in df_naive.iterrows()}
sil_wo_map    = {(int(r['n_neighbors']), float(r['min_dist']), r['metric']): r['silhouette']
                 for _, r in df_wo.iterrows()}
sil_full_map  = {(int(r['n_neighbors']), float(r['min_dist']), r['metric']): r['silhouette']
                 for _, r in df_full.iterrows()}

print(f"\n{'─'*70}")
print(f"  Saving {len(combos_all)} pair figures to {pairs_dir}/")
print(f"{'─'*70}")

pairs_pdf_path = os.path.join(pairs_dir, 'all_pairs.pdf')
with PdfPages(pairs_pdf_path) as pdf:
    for nn, md, metric in combos_all:
        key = (nn, md, metric)
        tag = make_tag(nn, md, metric)
        u_naive = cache_naive[key]
        u_wo    = cache_wo[key]
        u_full  = cache_full[key]
        sil_naive = sil_naive_map[key]
        sil_wo    = sil_wo_map[key]
        sil_full  = sil_full_map[key]
        fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4))
        draw_panel(axes[0], u_naive, test_labels, '(a) Naive',      sil_naive)
        draw_panel(axes[1], u_wo,   test_labels, '(b) w/o GGLP',   sil_wo)
        draw_panel(axes[2], u_full, test_labels, '(c) Ours',        sil_full)
        fig.suptitle(f'UMAP Feature Embedding — {dataset_tag} Test Set',
                     fontsize=12, fontweight='bold', y=1.02)
        fig.tight_layout(w_pad=2.5)
        fig.savefig(os.path.join(pairs_dir, f'{tag}.pdf'), format='pdf')
        pdf.savefig(fig)
        plt.close(fig)

        # ── 各模型单独面板图（保存到对应模型目录）──
        for single_emb, single_sil, single_title, single_dir in [
            (u_naive, sil_naive, '(a) Naive',    os.path.join(out_root, 'naive_mlp')),
            (u_wo,    sil_wo,    '(b) w/o GGLP', os.path.join(out_root, 'wo_gglp')),
            (u_full,  sil_full,  '(c) Ours',     os.path.join(out_root, 'full_model')),
        ]:
            fig_s, ax_s = plt.subplots(1, 1, figsize=(3.8, 3.4))
            draw_panel(ax_s, single_emb, test_labels, single_title, single_sil)
            fig_s.suptitle(f'UMAP Feature Embedding — {dataset_tag} Test Set',
                           fontsize=12, fontweight='bold', y=1.02)
            fig_s.tight_layout()
            fig_s.savefig(os.path.join(single_dir, f'{tag}.pdf'), format='pdf')
            plt.close(fig_s)

print(f"  Done — {len(combos_all)} pair PDFs saved.")
print(f"  All-in-one PDF: {pairs_pdf_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 合并汇总
# ═════════════════════════════════════════════════════════════════════════════

df_naive_tag = df_naive.rename(columns={'silhouette': 'sil_naive', 'rank': 'rank_naive'})
df_wo_tag    = df_wo.rename(columns={'silhouette': 'sil_wo_gglp', 'rank': 'rank_wo'})
df_full_tag  = df_full.rename(columns={'silhouette': 'sil_full', 'rank': 'rank_full'})

keys = ['n_neighbors', 'min_dist', 'metric']
df_merged = pd.merge(
    df_naive_tag[keys + ['sil_naive', 'rank_naive']],
    df_wo_tag[keys + ['sil_wo_gglp', 'rank_wo']],
    on=keys, how='outer',
)
df_merged = pd.merge(
    df_merged,
    df_full_tag[keys + ['sil_full', 'rank_full']],
    on=keys, how='outer',
)
df_merged.to_csv(os.path.join(out_root, 'summary.csv'), index=False)


# ═════════════════════════════════════════════════════════════════════════════
# 最优独立超参 → 论文级 1×3 对比图
# ═════════════════════════════════════════════════════════════════════════════

best_naive = df_naive.iloc[0]
best_wo    = df_wo.iloc[0]
best_full  = df_full.iloc[0]

best_key_naive = (int(best_naive['n_neighbors']),
                  float(best_naive['min_dist']),
                  best_naive['metric'])
best_key_wo    = (int(best_wo['n_neighbors']),
                  float(best_wo['min_dist']),
                  best_wo['metric'])
best_key_full  = (int(best_full['n_neighbors']),
                  float(best_full['min_dist']),
                  best_full['metric'])

u_naive_best = cache_naive[best_key_naive]
u_wo_best    = cache_wo[best_key_wo]
u_full_best  = cache_full[best_key_full]
sil_naive_best = float(best_naive['silhouette'])
sil_wo_best    = float(best_wo['silhouette'])
sil_full_best  = float(best_full['silhouette'])

print("\n" + "=" * 70)
print("Best independent hyperparameters:")
print(f"  Naive MLP : nn={best_key_naive[0]}, md={best_key_naive[1]}, "
      f"metric={best_key_naive[2]}, Sil={sil_naive_best:.4f}")
print(f"  w/o GGLP  : nn={best_key_wo[0]}, md={best_key_wo[1]}, "
      f"metric={best_key_wo[2]}, Sil={sil_wo_best:.4f}")
print(f"  Full Model: nn={best_key_full[0]}, md={best_key_full[1]}, "
      f"metric={best_key_full[2]}, Sil={sil_full_best:.4f}")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4))

draw_panel(
    axes[0], u_naive_best, test_labels,
    '(a) Naive', sil_naive_best)

draw_panel(
    axes[1], u_wo_best, test_labels,
    '(b) w/o GGLP', sil_wo_best)

draw_panel(
    axes[2], u_full_best, test_labels,
    '(c) Ours', sil_full_best)

fig.suptitle(f'UMAP Feature Embedding — {dataset_tag} Test Set',
             fontsize=12, fontweight='bold', y=1.02)
fig.tight_layout(w_pad=2.5)

best_path = os.path.join(out_root, 'best_independent.pdf')
fig.savefig(best_path, format='pdf')
plt.close(fig)

# ─── 完成 ─────────────────────────────────────────────────────────────────────
total_combos = len(N_NEIGHBORS_LIST) * len(MIN_DIST_LIST) * len(METRIC_LIST)
print(f"\nAll outputs saved to  {out_root}/")
print(f"  - naive_mlp/   : 3 gallery PDFs + ranking.csv")
print(f"  - wo_gglp/     : 3 gallery PDFs + ranking.csv")
print(f"  - full_model/  : 3 gallery PDFs + ranking.csv")
print(f"  - pairs/       : {total_combos} 1×3 pair PDFs + all_pairs.pdf")
print(f"  - best_independent.pdf  (final 1×3 comparison figure)")
print(f"  - summary.csv")
print("Done!")
