import argparse
import os
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.ticker as ticker

# ============================================================
# 0. 参数解析
# ============================================================
parser = argparse.ArgumentParser(description='Hyperparameter sensitivity 3D surface plot (alpha vs mask_percentile)')
parser.add_argument('--dataset', type=str, default='twitter', choices=['pheme', 'weibo', 'twitter'])
parser.add_argument('--log_dir', type=str, default=os.path.join('train_log'),
                    help='Base directory containing hparam search results')
parser.add_argument('--out_dir', type=str, default='.', help='Output directory for figures')
args = parser.parse_args()

DATASET = args.dataset

# ============================================================
# 1. 全局字体与样式（学术论文风格）
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 13,
    'axes.labelsize': 15,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 150,
})

# ============================================================
# 2. 数据集路径配置
# ============================================================
DATASET_CFG = {
    'pheme':   {'csv': os.path.join(args.log_dir, 'pheme', 'hparam_results_pheme.csv'),   'bad_lines': 'error'},
    'weibo':   {'csv': os.path.join(args.log_dir, 'weibo', 'hparam_results_weibo.csv'),   'bad_lines': 'error'},
    'twitter': {'csv': os.path.join(args.log_dir, 'twitter', 'hparam_results_twitter_full.csv'), 'bad_lines': 'error'},
}

cfg = DATASET_CFG[DATASET]

# ============================================================
# 3. 读取数据，自动寻找其余超参的最优组合
# ============================================================
df = pd.read_csv(cfg['csv'], on_bad_lines=cfg['bad_lines'])

other_params = ['target_k', 'rho', 'dropout']
mean_by_other = df.groupby(other_params)['best_acc'].mean().reset_index()
best_other = mean_by_other.loc[mean_by_other['best_acc'].idxmax()]
best_k  = best_other['target_k']
best_rho = best_other['rho']
best_do  = best_other['dropout']

print(f"[{DATASET}] Best fixed hyperparams: target_k={best_k}, rho={best_rho}, dropout={best_do}")

filtered = df[
    (df['target_k'] == best_k) &
    (df['rho']      == best_rho) &
    (df['dropout']  == best_do)
]

pivot = (
    filtered
    .groupby(['alpha', 'mask_percentile'])['best_acc']
    .mean()
    .unstack('mask_percentile')
)

alpha_values = pivot.index.values
mp_values    = pivot.columns.values
Z            = pivot.values

print(f"[{DATASET}] acc matrix (alpha x mask_percentile):\n{pivot.round(4)}")

log_mp = np.log10(mp_values)
X, Y   = np.meshgrid(log_mp, alpha_values)

# ============================================================
# 4. 三维曲面图
# ============================================================
fig = plt.figure(figsize=(9, 7))
ax  = fig.add_subplot(111, projection='3d')

Z_plot = np.ma.masked_invalid(Z)

surf = ax.plot_surface(
    X, Y, Z_plot,
    cmap=cm.viridis,
    edgecolor='gray',
    linewidth=0.5,
    alpha=0.92,
    rstride=1, cstride=1,
    antialiased=True,
    shade=True,
)

# 标注最优点
valid_mask = ~np.isnan(Z)
if valid_mask.any():
    flat_best = np.nanargmax(Z)
    best_idx  = np.unravel_index(flat_best, Z.shape)
    ax.scatter(
        X[best_idx], Y[best_idx], Z[best_idx],
        color='red', s=60, zorder=5,
        label=(
            f'Best Acc = {Z[best_idx]:.4f}\n'
            r'$\alpha$=' + f'{alpha_values[best_idx[0]]:.1f}'
            r', $p$=' + f'{mp_values[best_idx[1]]:g}'
        )
    )
    ax.legend(fontsize=11, loc='upper left')

# ============================================================
# 5. 坐标轴格式化
# ============================================================
mp_labels = [f'{v:g}' for v in mp_values]

ax.set_xticks(log_mp)
ax.set_xticklabels(mp_labels)
ax.set_xlabel(r'$p$', labelpad=12)

ax.set_yticks(alpha_values)
ax.set_yticklabels([f'{v:.1f}' for v in alpha_values])
ax.set_ylabel(r'$\alpha$', labelpad=12)

ax.set_zlabel('Accuracy', labelpad=10)
ax.zaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

z_valid = Z[valid_mask]
z_margin = (z_valid.max() - z_valid.min()) * 0.3 + 1e-4
ax.set_zlim(z_valid.min() - z_margin, z_valid.max() + z_margin)

# ============================================================
# 6. 视角与网格美化
# ============================================================
ax.view_init(elev=25, azim=225)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('lightgray')
ax.yaxis.pane.set_edgecolor('lightgray')
ax.zaxis.pane.set_edgecolor('lightgray')
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# ============================================================
# 7. 颜色条
# ============================================================
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, pad=0.04)
cbar.set_label('Accuracy', fontsize=13)
cbar.ax.tick_params(labelsize=11)

# ============================================================
# 8. 保存
# ============================================================
fig.subplots_adjust(left=0.05, right=0.92, top=0.92, bottom=0.05)

os.makedirs(args.out_dir, exist_ok=True)
plt.savefig(os.path.join(args.out_dir, f'sensitivity_{DATASET}_alpha_mp.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(args.out_dir, f'sensitivity_{DATASET}_alpha_mp.png'), bbox_inches='tight', dpi=300)
print(f"[{DATASET}] Figures saved to {args.out_dir}")
