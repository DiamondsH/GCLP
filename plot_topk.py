import argparse
import os
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ===== 参数解析 =====
parser = argparse.ArgumentParser(description='3D bar chart of Top-K sensitivity across datasets')
parser.add_argument('--log_dir', type=str, default=os.path.join('train_log'),
                    help='Base directory containing Top-K hparam search results')
parser.add_argument('--out_dir', type=str, default='.', help='Output directory for figures')
args = parser.parse_args()

# ===== 学术字体设置 =====
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 26,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'axes.titlesize': 28,
    'figure.dpi': 150,
})

# ===== 读取数据 =====
datasets = {
    'Twitter': os.path.join(args.log_dir, 'twitter', 'hparam_results_targetk_twitter.csv'),
    'Weibo':   os.path.join(args.log_dir, 'weibo', 'hparam_results_targetk_weibo.csv'),
    'PHEME':   os.path.join(args.log_dir, 'pheme', 'hparam_results_targetk_pheme.csv'),
}

all_topk = [10, 20, 30, 40, 50]

dataset_names = list(datasets.keys())

acc_matrix = np.full((len(dataset_names), len(all_topk)), np.nan)

for i, (name, path) in enumerate(datasets.items()):
    df = pd.read_csv(path)
    df = df.sort_values('target_k')
    for j, k in enumerate(all_topk):
        row = df[df['target_k'] == k]
        if not row.empty:
            acc_matrix[i, j] = row['best_acc'].values[0]

# ===== 柱子底部基准 =====
valid_vals = acc_matrix[~np.isnan(acc_matrix)]
z_base = np.floor(valid_vals.min() * 200) / 200 - 0.002

# ===== 配色 =====
cmap = plt.cm.Blues
norm = matplotlib.colors.Normalize(
    vmin=valid_vals.min() - 0.005,
    vmax=valid_vals.max()
)

# ===== 3D 柱状图 =====
fig = plt.figure(figsize=(11, 7.5))
ax = fig.add_subplot(111, projection='3d')

n_datasets = len(dataset_names)
n_topk = len(all_topk)

_x = np.arange(n_topk)
_y = np.arange(n_datasets)

width = 0.55
depth = 0.55

for i in range(n_datasets):
    for j in range(n_topk):
        val = acc_matrix[i, j]
        if np.isnan(val):
            continue
        height = val - z_base
        color = cmap(norm(val))
        ax.bar3d(
            j - width / 2,
            i - depth / 2,
            z_base,
            width, depth, height,
            shade=True, color=color, alpha=0.92,
            zsort='average'
        )

# ===== 轴标签与刻度 =====
ax.set_xlabel('Top-$k$', fontsize=26, labelpad=10)
ax.set_ylabel('Dataset', fontsize=26, labelpad=10)
ax.set_zlabel('Accuracy', fontsize=26, labelpad=8)

ax.set_xticks(_x)
ax.set_xticklabels([str(k) for k in all_topk], fontsize=20)
ax.set_yticks(_y)
ax.set_yticklabels(dataset_names, fontsize=22)

z_top = np.ceil(valid_vals.max() * 200) / 200 + 0.002
ax.set_zlim(z_base, z_top)

z_step = round((z_top - z_base) / 4, 3)
z_ticks = np.arange(z_base, z_top + z_step * 0.1, z_step)
ax.set_zticks(z_ticks)
ax.zaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

# ===== 视角 =====
ax.view_init(elev=22, azim=-50)

# ===== Colorbar =====
mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
mappable.set_array(valid_vals)
cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, aspect=18, pad=0.08)
cbar.set_label('Accuracy', fontsize=22)
cbar.ax.tick_params(labelsize=18)
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

# ===== 保存 =====
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
os.makedirs(args.out_dir, exist_ok=True)
plt.savefig(os.path.join(args.out_dir, 'topk_sensitivity.png'), dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.savefig(os.path.join(args.out_dir, 'topk_sensitivity.pdf'), bbox_inches='tight', pad_inches=0.05)
print(f"Figures saved to {args.out_dir}")
