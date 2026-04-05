# GCLP

## Requirements

```bash
pip install -r requirements.txt
```

## Dataset

Download the dataset from Baidu Netdisk and extract it to the project root directory:

- Link: https://pan.baidu.com/s/1dopAnGEupZU0NqyI1lbi1A?pwd=3h6a
- Password: `3h6a`

After extraction, the directory structure should be:

```
GCLP/
├── dataset/
│   ├── pheme/
│   │   ├── dataforGCN_train.csv
│   │   ├── dataforGCN_test.csv
│   │   ├── TweetEmbeds.pt
│   │   ├── TweetGraph.pt
│   │   └── pseudo_labels_output_deepseek.csv
│   ├── twitter/   (same structure)
│   └── weibo/     (same structure)
├── train.py
├── ...
```

## Usage

### Training

```bash
python train.py --dataset pheme
python train.py --dataset twitter
python train.py --dataset weibo
```

### Ablation Study

```bash
python ablation_study.py --dataset pheme --runs 5
```

### Hyperparameter Search

```bash
python hparam_search.py --dataset all --gpu_ids 0 1 2 3
```

### Visualization

```bash
# UMAP visualization
python umap_visualization.py --dataset pheme

# UMAP hyperparameter search
python umap_hparam_search.py --dataset pheme

# Loss surface search
python loss_surface_search.py --dataset pheme

# Sensitivity plot (alpha vs mask_percentile)
python plot_sensitivity.py --dataset twitter --log_dir train_log

# Top-K sensitivity plot
python plot_topk.py --log_dir train_log
```

## Project Structure

| File | Description |
|------|-------------|
| `model.py` | FCN_LP model architecture |
| `LPN_layer.py` | Label Propagation layer (LPAconv) |
| `mmd.py` | MMD loss for domain adaptation |
| `train.py` | Main training script with GWG + GSNR-ASCR |
| `ablation_study.py` | Ablation study (Full / w/o GGLP / w/o GDDM) |
| `hparam_search.py` | Parallel hyperparameter grid search |
| `umap_visualization.py` | UMAP embedding visualization |
| `umap_hparam_search.py` | UMAP hyperparameter search |
| `loss_surface_search.py` | Loss surface visualization search |
| `plot_sensitivity.py` | Hyperparameter sensitivity surface plot |
| `plot_topk.py` | Top-K sensitivity bar chart |
