"""
UMAP 表征可视化：对比 w/o GGLP 与 Full Model 在 PHEME 测试集上的特征嵌入分离度

用法:
    python umap_visualization.py [--seed 42] [--device cuda:0]
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import silhouette_score
import umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import FCN_LP

# ─── 参数 ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--output', type=str, default='umap_comparison.pdf')
args = parser.parse_args()

device = args.device
seed = args.seed

# ─── 数据加载 ──────────────────────────────────────────────────────────────────
dataset_name = 'pheme'
train_data = pd.read_csv(f'dataset/{dataset_name}/dataforGCN_train.csv')
test_data  = pd.read_csv(f'dataset/{dataset_name}/dataforGCN_test.csv')
tweet_embeds = torch.load(f'dataset/{dataset_name}/TweetEmbeds.pt', map_location='cpu')
tweet_graph  = torch.load(f'dataset/{dataset_name}/TweetGraph.pt', map_location='cpu')

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
    y          = labels
).to(device)

# 测试集真实标签: 1=Real, 0=Fake
test_labels = np.array(label_list_test)


# ─── 模型加载与特征提取 ────────────────────────────────────────────────────────
def _forward_extract(model, data_cur, eff_edge_weight):
    """手动前向传播提取 GCN 最后隐藏层输出（分类头之前）"""
    x = data_cur.x
    edge_index = data_cur.edge_index
    for i in range(len(model.gc) - 1):
        x = model.gc[i](x, edge_index, eff_edge_weight)
        x = F.relu(x)
        x = F.dropout(x, model.dropout_rate, training=model.training)
    return x


def load_model_and_extract(ckpt_path, data, device):
    """加载 checkpoint 并提取测试集上 GCN 最后隐藏层特征"""
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt['args']
    hp = ckpt['hp']

    model = FCN_LP(
        tweet_embeds.shape[1], saved_args['hidden'], saved_args['num_classes'],
        hp['dropout'], data.num_edges, saved_args['lpaiters'], saved_args['gcnnum']
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 判断是否有动态图边（Full Model 会有）
    has_dynamic = 'cur_edge_index' in ckpt and ckpt['cur_edge_index'] is not None

    with torch.no_grad():
        if has_dynamic:
            cur_edge_index = ckpt['cur_edge_index'].to(device)
            cur_edge_attr  = ckpt['cur_edge_attr'].to(device)
            n_base = data.num_edges
            n_cur  = cur_edge_index.shape[1]

            # 构造有效边权重：基础边权重 + 动态边权重
            if n_cur > n_base:
                dyn_w = cur_edge_attr[n_base:, 0].detach()
                eff_ew = torch.cat([model.edge_weight, dyn_w], dim=0)
            else:
                eff_ew = model.edge_weight

            data_cur = Data(
                x=data.x, edge_index=cur_edge_index, edge_attr=cur_edge_attr,
                train_mask=data.train_mask, test_mask=data.test_mask, y=data.y
            )
            x_feat = _forward_extract(model, data_cur, eff_ew)
        else:
            # 无动态图，直接 forward
            _, _, x_feat = model(data)

    # 提取测试集特征
    embeddings = x_feat[data.test_mask].cpu().numpy()

    info = {
        'config': ckpt.get('config_name', 'unknown'),
        'acc':    ckpt.get('test_acc', 0),
        'f1':     ckpt.get('f1', 0),
    }
    print(f"Loaded [{info['config']}]: Acc={info['acc']:.4f}, F1={info['f1']:.4f}, "
          f"embeddings shape={embeddings.shape}")
    return embeddings, info


# ─── 加载两个模型 ──────────────────────────────────────────────────────────────
ckpt_wo_gglp = f'checkpoints/ablation/pheme_w_o_GGLP_seed{seed}_best.pt'
ckpt_full    = f'checkpoints/ablation/pheme_GRAVITAS_Full_seed{seed}_best.pt'

print("=" * 60)
print("Loading w/o GGLP model...")
emb_wo_gglp, info_wo_gglp = load_model_and_extract(ckpt_wo_gglp, data, device)

print("\nLoading Full Model...")
emb_full, info_full = load_model_and_extract(ckpt_full, data, device)
print("=" * 60)


# ─── UMAP 降维 ────────────────────────────────────────────────────────────────
print("\nRunning UMAP for w/o GGLP...")
reducer_wo = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
umap_wo = reducer_wo.fit_transform(emb_wo_gglp)

print("Running UMAP for Full Model...")
reducer_full = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
umap_full = reducer_full.fit_transform(emb_full)


# ─── Silhouette Score ─────────────────────────────────────────────────────────
sil_wo   = silhouette_score(umap_wo,   test_labels)
sil_full = silhouette_score(umap_full, test_labels)
print(f"\nSilhouette Score — w/o GGLP: {sil_wo:.4f}, Full Model: {sil_full:.4f}")


# ─── 绘图 ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_configs = [
    (axes[0], umap_wo,   sil_wo,   'w/o GGLP',   info_wo_gglp),
    (axes[1], umap_full, sil_full, 'Full Model',  info_full),
]

for ax, umap_emb, sil, title, info in plot_configs:
    real_mask = test_labels == 1
    fake_mask = test_labels == 0

    ax.scatter(umap_emb[fake_mask, 0], umap_emb[fake_mask, 1],
               c='#E74C3C', label='Fake', s=15, alpha=0.6, edgecolors='none')
    ax.scatter(umap_emb[real_mask, 0], umap_emb[real_mask, 1],
               c='#3498DB', label='Real', s=15, alpha=0.6, edgecolors='none')

    ax.set_title(f'{title}\nSilhouette Score = {sil:.4f}', fontsize=13, fontweight='bold')
    ax.set_xlabel('UMAP-1', fontsize=11)
    ax.set_ylabel('UMAP-2', fontsize=11)
    ax.legend(loc='best', fontsize=10, framealpha=0.8)
    ax.tick_params(labelsize=9)

fig.suptitle('UMAP Visualization on PHEME Test Set', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()

# 保存
output_dir = 'umap_output'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, args.output)
fig.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"\nSaved to {output_path}")
plt.close()
