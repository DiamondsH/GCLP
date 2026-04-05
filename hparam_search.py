"""
并行超参数搜索脚本 for train.py (v2 - 多 GPU 并行)
  - 通过 CUDA_VISIBLE_DEVICES 将任务分配到不同 GPU，充分利用多卡资源
  - 自动跳过已完成的运行（日志中含 Best Result）
  - 线程安全 CSV 写入 + 进度统计
  - 不实时打印每行训练输出（多卡并行时太混乱），结果统一由 logger 写入日志文件

用法:
  # ★ 接着跑 twitter，使用指定 GPU（pheme/weibo 已完成，twitter 跑到 36/972）
  python hparam_search_gsnr_ascr_v3_parallel.py --dataset twitter --resume_from 35 --gpu_ids 1 2 4 5 6 7

  # 全部数据集从头跑（默认 GPU 0~5）
  python hparam_search_gsnr_ascr_v3_parallel.py

  # 指定任意 GPU 组合
  python hparam_search_gsnr_ascr_v3_parallel.py --dataset twitter --resume_from 35 --gpu_ids 0 1 2 3

  # dry run（只打印组合数）
  python hparam_search_gsnr_ascr_v3_parallel.py --dry_run
"""

import argparse
import csv
import itertools
import os
import queue
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── 搜索空间 ────────────────────────────────────────────────────────────────────
SEARCH_SPACE = {
    'target_k':        [20, 40, 60],
    'rho':             [0.8, 0.9, 0.99],
    'dropout':         [0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
    'mask_percentile': [0.5, 1, 2, 5, 10, 50],
    'gwg_M':           [3],
    'alpha':           [0.7, 0.8, 0.9],
}

# 固定参数（不搜索）
FIXED_PARAMS = {
    'epochs':       6000,
    'mask_warmup':  300,
    'mask_beta':    0.5,
    'lr':           1e-4,
    'weight_decay': 5e-4,
    'hidden':       64,
    'num_classes':  2,
    'lpaiters':     2,
    'gcnnum':       3,
    'gwg_interval': 30,
    'gwg_warmup':   30,
    'gwg_batch':    64,
    'seed':         42,
}

# ─── 参数 ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='all',
                    choices=['all', 'pheme', 'twitter', 'weibo'],
                    help='指定数据集，all 表示依次跑 pheme → weibo → twitter')
parser.add_argument('--dry_run', action='store_true', help='仅打印组合数，不运行')
parser.add_argument('--resume_from', type=int, default=0,
                    help='跳过前 N 个组合（仅对单 --dataset 生效；'
                         '如接着跑 twitter 36/972，传 --resume_from 35）')
parser.add_argument('--gpu_ids', type=int, nargs='+', default=None,
                    help='指定使用的 GPU 编号列表，例如 --gpu_ids 1 2 4 5 6 7；'
                         '不指定则默认使用 0,1,...,num_gpus-1')
parser.add_argument('--num_gpus', type=int, default=6,
                    help='并行 GPU 数量（仅在未指定 --gpu_ids 时生效，默认 6）')
args = parser.parse_args()

# 解析最终使用的 GPU 列表
gpu_id_list = args.gpu_ids if args.gpu_ids is not None else list(range(args.num_gpus))

DATASET_ORDER = ['pheme', 'weibo', 'twitter']
datasets_to_run = DATASET_ORDER if args.dataset == 'all' else [args.dataset]

# ─── 生成所有组合 ────────────────────────────────────────────────────────────────
param_names  = list(SEARCH_SPACE.keys())
param_values = [SEARCH_SPACE[k] for k in param_names]
all_combos   = list(itertools.product(*param_values))
total        = len(all_combos)

print(f'数据集     : {" → ".join(datasets_to_run)}')
print(f'搜索空间   : {" × ".join(str(len(v)) for v in param_values)} = {total} 组合/数据集')
print(f'总运行数   : {total} × {len(datasets_to_run)} = {total * len(datasets_to_run)}')
print(f'使用 GPU   : {gpu_id_list}  (共 {len(gpu_id_list)} 张)')
for k, v in SEARCH_SPACE.items():
    print(f'  {k}: {v}')

if args.dry_run:
    print('\n[dry_run] 不运行训练，退出。')
    sys.exit(0)

# ─── 工具函数 ────────────────────────────────────────────────────────────────────
def parse_best_result(log_path):
    """从日志文件解析 === Best Result === 行。"""
    best = {'epoch': 0, 'acc': 0.0, 'prec': 0.0, 'recall': 0.0, 'f1': 0.0}
    if not os.path.exists(log_path):
        return best
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if '=== Best Result ===' in line:
                m = re.search(
                    r'Epoch\s+(\d+).*Acc=([\d.]+).*Prec=([\d.]+).*Rec=([\d.]+).*F1=([\d.]+)',
                    line
                )
                if m:
                    best['epoch'] = int(m.group(1))
                    best['acc']   = float(m.group(2))
                    best['prec']  = float(m.group(3))
                    best['recall']= float(m.group(4))
                    best['f1']    = float(m.group(5))
    return best

# ─── 训练脚本路径 ────────────────────────────────────────────────────────────────
train_script = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'train.py'
)

# ─── 全局共享状态 ────────────────────────────────────────────────────────────────
gpu_pool    = queue.Queue()          # GPU 资源池
print_lock  = threading.Lock()       # 控制台打印锁
counter_lock = threading.Lock()      # 进度计数锁

for g in gpu_id_list:
    gpu_pool.put(g)

def safe_print(*a, **kw):
    with print_lock:
        print(*a, **kw, flush=True)


# ─── 单任务执行函数（在线程中运行）────────────────────────────────────────────────
def run_one_job(idx, combo, dataset_name, base_log_dir, total,
                completed_counter, skipped_counter):
    """
    在从 GPU 池取得的 GPU 上执行一次超参训练。
    返回: (idx, hparams, best_result_dict, log_file)
    """
    hparams  = dict(zip(param_names, combo))
    tag      = (f"tk{hparams['target_k']}_rho{hparams['rho']}_do{hparams['dropout']}"
                f"_mp{hparams['mask_percentile']}_M{hparams['gwg_M']}_a{hparams['alpha']}")
    log_file = os.path.join(base_log_dir, f'run_{idx:05d}_{tag}.log')

    empty = {'epoch': 0, 'acc': 0.0, 'prec': 0.0, 'recall': 0.0, 'f1': 0.0}

    # ── 已完成则直接跳过 ────────────────────────────────────────────────────────
    if os.path.exists(log_file):
        existing = parse_best_result(log_file)
        if existing['epoch'] > 0:
            with counter_lock:
                skipped_counter[0] += 1
                done_total = completed_counter[0] + skipped_counter[0]
            safe_print(f'[SKIP][{dataset_name} {idx+1}/{total}] '
                       f'已存在: {tag} | Acc={existing["acc"]:.4f} '
                       f'(跳过 {skipped_counter[0]}, 已完成 {completed_counter[0]})')
            return idx, hparams, existing, log_file

    # ── 从池中取一个 GPU ─────────────────────────────────────────────────────────
    gpu_id = gpu_pool.get()
    t0 = time.time()
    safe_print(f'\n[GPU-{gpu_id}][{dataset_name} {idx+1}/{total}] START  {tag}')

    try:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        cmd = [sys.executable, train_script,
               '--dataset', dataset_name,
               '--log_file', log_file]
        for k, v in hparams.items():
            cmd.extend([f'--{k}', str(v)])
        for k, v in FIXED_PARAMS.items():
            cmd.extend([f'--{k}', str(v)])

        # 子进程：stdout/stderr 写入日志由 train 脚本的 logger 完成；
        # 此处只读出 stdout 防止管道阻塞，不实时打印（多卡并行时输出混乱）。
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding='utf-8', errors='replace'
        )
        proc.communicate()   # 等待完成并丢弃 stdout（已有日志文件）
        elapsed = time.time() - t0

        if proc.returncode != 0:
            safe_print(f'[GPU-{gpu_id}][{dataset_name} {idx+1}/{total}] '
                       f'ERROR 退出码={proc.returncode}, 耗时={elapsed:.1f}s | {tag}')
            return idx, hparams, empty, log_file

        best = parse_best_result(log_file)
        with counter_lock:
            completed_counter[0] += 1
            done_total = completed_counter[0] + skipped_counter[0]

        safe_print(f'[GPU-{gpu_id}][{dataset_name} {idx+1}/{total}] '
                   f'DONE {elapsed:.1f}s | Acc={best["acc"]:.4f} F1={best["f1"]:.4f} | {tag} '
                   f'(已完成 {completed_counter[0]}, 跳过 {skipped_counter[0]}, 共 {done_total}/{total})')
        return idx, hparams, best, log_file

    except Exception as e:
        elapsed = time.time() - t0
        safe_print(f'[GPU-{gpu_id}][{dataset_name} {idx+1}/{total}] '
                   f'EXCEPTION {e} 耗时={elapsed:.1f}s | {tag}')
        return idx, hparams, empty, log_file

    finally:
        gpu_pool.put(gpu_id)   # 归还 GPU


# ─── 逐数据集并行搜索 ────────────────────────────────────────────────────────────
overall_summary = {}

for ds_idx, dataset_name in enumerate(datasets_to_run):
    print('\n' + '#' * 100)
    print(f'# 数据集 [{ds_idx+1}/{len(datasets_to_run)}]: {dataset_name}')
    print('#' * 100)

    base_log_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'train_log',
        dataset_name
    )
    os.makedirs(base_log_dir, exist_ok=True)

    csv_path   = os.path.join(base_log_dir, f'hparam_results_{dataset_name}.csv')
    csv_header = ['run_id'] + param_names + [
        'best_epoch', 'best_acc', 'best_prec', 'best_recall', 'best_f1', 'log_file'
    ]

    # resume_from 只在 --dataset 单独指定（非 all）时生效，避免误跳过其他数据集
    resume_this = args.resume_from if args.dataset != 'all' else 0

    # CSV：续写模式（存在时追加，不覆盖；否则新建并写表头）
    csv_mode = 'a' if os.path.exists(csv_path) else 'w'
    csv_file   = open(csv_path, csv_mode, newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_lock   = threading.Lock()
    if csv_mode == 'w':
        csv_writer.writerow(csv_header)
        csv_file.flush()

    def write_csv_row(idx, hparams, best):
        with csv_lock:
            csv_writer.writerow(
                [idx] + [hparams[k] for k in param_names] +
                [best['epoch'], best['acc'], best['prec'], best['recall'], best['f1'],
                 os.path.join(base_log_dir, f'run_{idx:05d}_'
                              f"tk{hparams['target_k']}_rho{hparams['rho']}"
                              f"_do{hparams['dropout']}_mp{hparams['mask_percentile']}"
                              f"_M{hparams['gwg_M']}_a{hparams['alpha']}.log")]
            )
            csv_file.flush()

    ds_best_acc   = 0.0
    ds_best_combo = None
    ds_best_lock  = threading.Lock()

    completed_counter = [0]   # 用列表模拟可变引用
    skipped_counter   = [0]

    print(f'开始搜索 {dataset_name}，共 {total} 组合，'
          f'从第 {resume_this} 个开始，GPU {gpu_id_list} 并行')

    # 过滤掉明确跳过的前 N 个（--resume_from）
    jobs = [(idx, combo) for idx, combo in enumerate(all_combos) if idx >= resume_this]

    ds_t0 = time.time()

    with ThreadPoolExecutor(max_workers=len(gpu_id_list)) as executor:
        future_map = {
            executor.submit(
                run_one_job,
                idx, combo, dataset_name, base_log_dir, total,
                completed_counter, skipped_counter
            ): idx
            for idx, combo in jobs
        }

        for future in as_completed(future_map):
            try:
                idx, hparams, best, log_file = future.result()
            except Exception as e:
                safe_print(f'[FATAL] future 异常: {e}')
                continue

            # 写 CSV
            write_csv_row(idx, hparams, best)

            # 更新该数据集最优（线程安全）
            if best['acc'] > 0.0:
                with ds_best_lock:
                    if best['acc'] > ds_best_acc:
                        ds_best_acc   = best['acc']
                        ds_best_combo = hparams.copy()
                        safe_print(f'  ★ [{dataset_name}] 新的最优! '
                                   f'Acc={ds_best_acc:.4f} F1={best["f1"]:.4f} | {hparams}')

    csv_file.close()
    ds_elapsed = time.time() - ds_t0

    overall_summary[dataset_name] = {
        'acc': ds_best_acc, 'combo': ds_best_combo, 'csv': csv_path
    }

    print('\n' + '=' * 100)
    print(f'{dataset_name} 搜索完成! 共 {total} 组合, 耗时 {ds_elapsed/3600:.2f}h')
    print(f'结果 CSV  : {csv_path}')
    print(f'最优 Acc  : {ds_best_acc:.4f}')
    print(f'最优超参  : {ds_best_combo}')

# ─── 最终汇总 ────────────────────────────────────────────────────────────────────
print('\n' + '#' * 100)
print('# 所有数据集搜索完成 — 最终汇总')
print('#' * 100)
for ds_name, info in overall_summary.items():
    print(f'  {ds_name:10s} | Best Acc={info["acc"]:.4f} | {info["combo"]}')
    print(f'  {"":10s} | CSV: {info["csv"]}')
print('#' * 100)
