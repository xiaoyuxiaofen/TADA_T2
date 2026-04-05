#!/usr/bin/env python3
"""
predict_tad.py — TADA_T2 GPU 极速预测

功能：
  - 序列清洗（剔除 *, X 等非法氨基酸）
  - 矢量化 kappa/omega（无需 localCider）
  - GPU 特征计算 + 推理
  - 自动生成缺失的 lookup 表

用法:
    # 基本用法（自动检测/生成 lookup 表）
    python3 predict_tad.py sequences.fasta -o results.tsv

    # 指定已有的 lookup 表路径
    python3 predict_tad.py sequences.fasta --alpha_lookup ./alpha.npy --cider_lookup ./cider.npy

    # 仅生成 lookup 表（不预测）
    python3 predict_tad.py --gen_alpha
    python3 predict_tad.py --gen_cider
"""
import os
import sys
import argparse
import time
import itertools

# ── 参数解析（在 import TF 之前，加快 --help 响应）─────────
parser = argparse.ArgumentParser(
    description='TADA_T2 GPU 极速预测',
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('fasta', nargs='?', default=None, help='输入 FASTA 文件')
parser.add_argument('-o', '--output', default=None, help='输出 TSV 路径')
parser.add_argument('--batch_size', type=int, default=10000, help='每批处理序列数（默认 10000）')
parser.add_argument('--overlap', type=int, default=39, help='>40aa 序列的窗口重叠（默认 39）')
parser.add_argument('--alpha_lookup', default=None, help='alphaPredict 5-mer 查找表路径')
parser.add_argument('--cider_lookup', default=None, help='localCider 5-mer 查找表路径')
parser.add_argument('--gen_alpha', action='store_true', help='仅生成 alphaPredict 查找表并退出')
parser.add_argument('--gen_cider', action='store_true', help='仅生成 localCider 查找表并退出')
args = parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# 查找表自动生成
# ═══════════════════════════════════════════════════════════════

_AA = 'ACDEFGHIKLMNPQRSTVWY'
_TOTAL_5MER = 20 ** 5  # 3,200,000


def generate_alpha_lookup(output_path):
    """生成 alphaPredict 5-mer 查找表"""
    try:
        import alphaPredict.alpha as alpha
    except ImportError:
        print("❌ 需要安装 alphaPredict: pip install alphaPredict", file=sys.stderr)
        sys.exit(1)

    print(f"🧮 生成 alphaPredict 5-mer 查找表 ({_TOTAL_5MER:,} × 1)...")
    all_5mers = [''.join(c) for c in itertools.product(_AA, repeat=5)]
    lookup = np.zeros(_TOTAL_5MER, dtype=np.float32)
    BATCH = 10000
    t0 = time.time()

    for start in range(0, _TOTAL_5MER, BATCH):
        end = min(start + BATCH, _TOTAL_5MER)
        for idx in range(start, end):
            try:
                scores = alpha.predict(all_5mers[idx])
                lookup[idx] = sum(scores) / len(scores)
            except Exception:
                lookup[idx] = 0.0
        elapsed = time.time() - t0
        rate = end / elapsed if elapsed > 0 else 0
        eta = (_TOTAL_5MER - end) / rate if rate > 0 else 0
        print(f"\r  {end:>10,} / {_TOTAL_5MER:,} ({end/_TOTAL_5MER*100:.1f}%) | ETA: {eta:.0f}s",
              end='', flush=True)

    np.save(output_path, lookup)
    print(f"\n   ✅ 已保存: {output_path} ({lookup.nbytes/1024/1024:.1f} MB)")
    return output_path


def generate_cider_lookup(output_path):
    """生成 localCider 5-mer 查找表"""
    try:
        from localcider.sequenceParameters import SequenceParameters
    except ImportError:
        print("❌ 需要安装 localCider: pip install localcider", file=sys.stderr)
        sys.exit(1)

    print(f"🧮 生成 localCider 5-mer 查找表 ({_TOTAL_5MER:,} × 8)...")
    all_5mers = [''.join(c) for c in itertools.product(_AA, repeat=5)]
    lookup = np.zeros((_TOTAL_5MER, 8), dtype=np.float32)
    BATCH = 10000
    t0 = time.time()
    errors = 0

    for start in range(0, _TOTAL_5MER, BATCH):
        end = min(start + BATCH, _TOTAL_5MER)
        for idx in range(start, end):
            try:
                sp = SequenceParameters(all_5mers[idx])
                lookup[idx] = [
                    sp.get_mean_hydropathy(), sp.get_WW_hydropathy(),
                    sp.get_NCPR(), sp.get_fraction_disorder_promoting(),
                    sp.get_FCR(), sp.get_mean_net_charge(),
                    sp.get_fraction_negative(), sp.get_fraction_positive(),
                ]
            except Exception:
                errors += 1
        elapsed = time.time() - t0
        rate = end / elapsed if elapsed > 0 else 0
        eta = (_TOTAL_5MER - end) / rate if rate > 0 else 0
        print(f"\r  {end:>10,} / {_TOTAL_5MER:,} ({end/_TOTAL_5MER*100:.1f}%) | ETA: {eta:.0f}s",
              end='', flush=True)

    np.save(output_path, lookup)
    print(f"\n   ✅ 已保存: {output_path} ({lookup.nbytes/1024/1024:.1f} MB)")
    if errors:
        print(f"   ⚠️  {errors} 个 5-mer 计算失败（已填充 0）")
    return output_path


def find_or_generate_lookup(name, env_key, cli_path, gen_func, gen_flag):
    """
    查找 lookup 表：CLI 路径 → 环境变量 → 包数据目录 → 当前目录 → 自动生成。
    返回最终路径。
    """
    import numpy as np  # 确保可用

    # CLI 指定路径
    if cli_path:
        if os.path.exists(cli_path):
            print(f"   ✅ {name}: {cli_path}（CLI 指定）")
            os.environ[env_key] = os.path.abspath(cli_path)
            return cli_path
        else:
            print(f"   ❌ {name}: {cli_path} 不存在", file=sys.stderr)
            sys.exit(1)

    # 环境变量
    env_path = os.environ.get(env_key)
    if env_path and os.path.exists(env_path):
        print(f"   ✅ {name}: {env_path}（环境变量）")
        return env_path

    # 包数据目录
    try:
        import importlib.resources
        pkg_path = str(importlib.resources.files('TADA_T2.data') / f'{name}.npy')
        if os.path.exists(pkg_path):
            print(f"   ✅ {name}: {pkg_path}（包数据目录）")
            os.environ[env_key] = pkg_path
            return pkg_path
    except Exception:
        pass

    # 同目录（features.py 所在目录）
    try:
        import TADA_T2.backend.features as _f
        local_path = os.path.join(os.path.dirname(_f.__file__), f'{name}.npy')
        if os.path.exists(local_path):
            print(f"   ✅ {name}: {local_path}（同目录）")
            os.environ[env_key] = local_path
            return local_path
    except Exception:
        pass

    # 当前工作目录
    cwd_path = os.path.join(os.getcwd(), f'{name}.npy')
    if os.path.exists(cwd_path):
        print(f"   ✅ {name}: {cwd_path}（当前目录）")
        os.environ[env_key] = os.path.abspath(cwd_path)
        return cwd_path

    # 都没找到 → 自动生成
    print(f"   ⚠️  {name} 未找到，自动生成中...")
    os.makedirs(os.path.dirname(cwd_path) or '.', exist_ok=True)
    gen_func(cwd_path)
    os.environ[env_key] = os.path.abspath(cwd_path)
    return cwd_path


# ═══════════════════════════════════════════════════════════════
# --gen_alpha / --gen_cider 单独生成模式
# ═══════════════════════════════════════════════════════════════

import numpy as np

if args.gen_alpha:
    generate_alpha_lookup(args.alpha_lookup or 'alpha_5mer_lookup.npy')
    sys.exit(0)

if args.gen_cider:
    generate_cider_lookup(args.cider_lookup or 'cider_5mer_lookup.npy')
    sys.exit(0)

if args.fasta is None:
    parser.print_help()
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# 确保 lookup 表存在（CLI > 环境变量 > 包目录 > 自动生成）
# ═══════════════════════════════════════════════════════════════

print("🔍 检查查找表...")
find_or_generate_lookup('alpha_5mer_lookup', 'TADA_ALPHA_LOOKUP',
                        args.alpha_lookup, generate_alpha_lookup, args.gen_alpha)
find_or_generate_lookup('cider_5mer_lookup', 'TADA_CIDER_LOOKUP',
                        args.cider_lookup, generate_cider_lookup, args.gen_cider)
print()

# ── 现在安全 import TF + TADA_T2 ───────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=90000)]
        )
    except RuntimeError:
        pass

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from TADA_T2.backend.features import (
    _encode_sequences_to_tensor, _compute_all_features_gpu, scale_features_predict,
    _SCALER_MEAN, _SCALER_SCALE, _SCALER_MIN, _SCALER_RANGE,
)
from TADA_T2.backend.model import TadaModel
from TADA_T2.backend.predictor import get_model_path


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

_VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')


def compute_kappa_omega_fast(sequences):
    """批量计算 kappa/omega，纯 NumPy"""
    n = len(sequences)
    kappas = np.zeros(n, dtype=np.float32)
    omegas = np.zeros(n, dtype=np.float32)
    for i, seq in enumerate(sequences):
        L = len(seq)
        if L < 5:
            continue
        pos_idx = np.array([j for j, aa in enumerate(seq) if aa in ('K', 'R', 'H')], dtype=np.int32)
        neg_idx = np.array([j for j, aa in enumerate(seq) if aa in ('D', 'E')], dtype=np.int32)
        n_win = L - 5 + 1
        if n_win <= 0:
            continue
        f_plus = len(pos_idx) / L
        f_minus = len(neg_idx) / L

        def _wc(idxs):
            if len(idxs) == 0:
                return np.zeros(n_win, dtype=np.int32)
            l = np.searchsorted(idxs, np.arange(n_win), side='left')
            r = np.searchsorted(idxs, np.arange(n_win) + 5, side='left')
            return (r - l).astype(np.int32)

        n_pos = _wc(pos_idx)
        n_neg = _wc(neg_idx)
        n_charged = n_pos + n_neg

        if f_plus + f_minus >= 1e-12:
            like = ((n_pos > 0) & (n_neg == 0)) | ((n_neg > 0) & (n_pos == 0))
            chg = n_charged > 0
            nc = int(chg.sum())
            if nc > 0:
                p_obs = float(like.sum()) / nc
                p_rand = (f_plus**2 + f_minus**2) / (f_plus + f_minus)**2
                if p_rand >= 1e-12:
                    kappas[i] = p_obs / p_rand

        if f_plus >= 1e-12 and f_minus >= 1e-12:
            opp = (n_pos > 0) & (n_neg > 0)
            chg = n_charged > 0
            nc = int(chg.sum())
            if nc > 0:
                p_obs = float(opp.sum()) / nc
                p_rand = 2 * f_plus * f_minus / (f_plus + f_minus)**2
                if p_rand >= 1e-12:
                    omegas[i] = p_obs / p_rand

    return tf.constant(kappas, dtype=tf.float32), tf.constant(omegas, dtype=tf.float32)


def progress_iter(iterable, total, desc="Processing"):
    if HAS_TQDM:
        return tqdm(iterable, total=total, desc=desc, unit="seq",
                    bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}', mininterval=0.5)
    class SimpleProgress:
        def __init__(self, it, total, desc):
            self._it, self._total, self._desc = it, total, desc
            self._count, self._last_pct = 0, -1
        def __iter__(self):
            print(f"{self._desc}: 0%", end="", flush=True)
            for item in self._it:
                self._count += 1
                pct = int(self._count / self._total * 100)
                if pct >= self._last_pct + 5:
                    print(f" {pct}%", end="", flush=True)
                    self._last_pct = pct
                yield item
            print(" 100% ✓")
    return SimpleProgress(iterable, total, desc)


def parse_fasta(path):
    sequences = {}
    dup_count = 0
    header = None
    seq_buf = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    raw_name = header[1:].split()[0]
                    name = raw_name
                    c = 1
                    while name in sequences:
                        name = f"{raw_name}_{c}"
                        c += 1
                        dup_count += 1
                    sequences[name] = ''.join(seq_buf)
                header = line
                seq_buf = []
            else:
                seq_buf.append(line)
        if header is not None:
            raw_name = header[1:].split()[0]
            name = raw_name
            c = 1
            while name in sequences:
                name = f"{raw_name}_{c}"
                c += 1
                dup_count += 1
            sequences[name] = ''.join(seq_buf)
    return sequences, dup_count


def clean_sequences(sequences):
    cleaned = {}
    affected = 0
    total_removed = 0
    for name, seq in sequences.items():
        new_seq = ''.join(aa for aa in seq if aa in _VALID_AA)
        if len(new_seq) != len(seq):
            affected += 1
            total_removed += len(seq) - len(new_seq)
        cleaned[name] = new_seq
    return cleaned, affected, total_removed


def sliding_windows(seq, window_size=40, overlap=39):
    step = max(1, window_size - overlap)
    wins = []
    for i in range(0, len(seq) - window_size + 1, step):
        wins.append((seq[i:i + window_size], i + 1, i + window_size))
    return wins


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    if not os.path.exists(args.fasta):
        print(f"❌ 文件不存在: {args.fasta}", file=sys.stderr)
        sys.exit(1)

    # ── 解析 FASTA ──────────────────────────────────────────
    print(f"📂 读取: {args.fasta}")
    t0 = time.time()
    sequences, dup_count = parse_fasta(args.fasta)
    n_raw = len(sequences)
    print(f"   {n_raw} 条序列", end="")
    if dup_count:
        print(f"（{dup_count} 个重复名称已自动重命名）", end="")
    print(f"  解析耗时 {time.time()-t0:.1f}s")

    # ── 清洗 ────────────────────────────────────────────────
    print("🧹 清洗序列（剔除非标准氨基酸）...")
    sequences, affected, total_removed = clean_sequences(sequences)
    if affected:
        print(f"   ⚠️  {affected} 条序列含有非法字符，共剔除 {total_removed} 个字符")
    else:
        print("   ✅ 所有序列均为标准氨基酸，无需清洗")

    names = list(sequences.keys())
    seqs = list(sequences.values())
    n = len(names)

    n_short = sum(1 for s in seqs if len(s) < 40)
    n_long = sum(1 for s in seqs if len(s) > 40)
    n_exact = n - n_short - n_long
    print(f"   =40aa: {n_exact}  |  >40aa: {n_long}  |  <40aa: {n_short}")

    # ── 加载模型 ────────────────────────────────────────────
    print("\n🔧 加载模型到 GPU...")
    model = TadaModel().create_model()
    model.load_weights(get_model_path())
    _ = model.predict(np.zeros((1, 36, 42), dtype=np.float32), verbose=0)
    print("   ✅ 就绪\n")

    # ── 窗口切分 ────────────────────────────────────────────
    print("🔨 切分滑动窗口...")
    window_pool = []
    seq_window_map = {}
    global_idx = 0
    skip_indices = set()

    for name, seq in progress_iter(zip(names, seqs), n, desc="  切分"):
        if len(seq) < 40:
            skip_indices.add(global_idx)
            seq_window_map[global_idx] = {
                'name': name, 'seq': seq, 'seq_length': len(seq), 'is_short': True
            }
        elif len(seq) == 40:
            window_pool.append((global_idx, seq, 1, 40))
        else:
            wins = sliding_windows(seq, 40, args.overlap)
            for subseq, start, end in wins:
                window_pool.append((global_idx, subseq, start, end))
        global_idx += 1

    n_windows = len(window_pool)
    print(f"   总窗口数: {n_windows:,}")

    # ── kappa/omega ──────────────────────────────────────────
    print("🧮 预计算 kappa/omega (NumPy 矢量化)...")
    t_ko = time.time()
    all_kappas, all_omegas = compute_kappa_omega_fast(seqs)
    print(f"   ✅ {n} 条序列耗时 {time.time()-t_ko:.1f}s")

    # ── GPU 推理 ────────────────────────────────────────────
    print(f"\n🚀 GPU 批量推理（batch={args.batch_size:,}）...")
    t_start = time.time()

    best = {}
    for idx in skip_indices:
        info = seq_window_map[idx]
        best[idx] = {'score': 0.0, 'subseq': 'X' * info['seq_length'],
                      'start': 1, 'end': info['seq_length']}

    batch_seqs = [w[1] for w in window_pool]
    batch_meta = [(w[0], w[2], w[3]) for w in window_pool]

    for batch_start in progress_iter(range(0, n_windows, args.batch_size),
                                      (n_windows + args.batch_size - 1) // args.batch_size,
                                      desc="  推理"):
        batch_end = min(batch_start + args.batch_size, n_windows)
        batch = batch_seqs[batch_start:batch_end]
        meta = batch_meta[batch_start:batch_end]

        batch_kappa = tf.gather(all_kappas, [m[0] for m in meta])
        batch_omega = tf.gather(all_omegas, [m[0] for m in meta])

        encoded = _encode_sequences_to_tensor(batch, 40)
        features = _compute_all_features_gpu(encoded, 36, 5, 40, batch_kappa, batch_omega)
        features = scale_features_predict(features, _SCALER_MEAN, _SCALER_SCALE,
                                           _SCALER_MIN, _SCALER_RANGE)
        preds = model.predict(features, verbose=0)

        for i, (idx, start, end) in enumerate(meta):
            score = float(preds[i][0])
            if idx not in best or score > best[idx]['score']:
                best[idx] = {'score': score, 'subseq': batch[i], 'start': start, 'end': end}

    elapsed = time.time() - t_start
    wps = n_windows / elapsed if elapsed > 0 else 0
    sps = n / elapsed if elapsed > 0 else 0
    print(f"   ✅ 完成 | {elapsed:.1f}s | {wps:,.0f} windows/s | {sps:,.0f} seqs/s")

    # ── 输出 ────────────────────────────────────────────────
    header = "序列名称\t序列长度\tTAD最高分\t最佳40aa片段\t起始位置\t终止位置"
    lines = []
    for i, (name, seq) in enumerate(zip(names, seqs)):
        r = best.get(i, {'score': 0, 'subseq': '', 'start': 0, 'end': 0})
        lines.append(f"{name}\t{len(seq)}\t{r['score']:.6f}\t{r['subseq']}\t{r['start']}\t{r['end']}")

    if args.output:
        with open(args.output, 'w') as f:
            f.write(header + '\n')
            f.write('\n'.join(lines) + '\n')
        print(f"\n💾 结果: {args.output}")
    else:
        print(f"\n{header}")
        for line in lines[:20]:
            print(line)
        if len(lines) > 20:
            print(f"... 共 {len(lines)} 条，用 -o 参数保存完整结果")

    # ── 汇总 ────────────────────────────────────────────────
    scores = [best[i]['score'] for i in range(n) if i in best]
    n_tad = sum(1 for s in scores if s >= 0.5)
    print(f"\n{'='*50}")
    print(f"📊 汇总")
    print(f"{'='*50}")
    print(f"  总序列数:     {n:,}")
    print(f"  TAD (≥0.5):   {n_tad:,} ({n_tad/n*100:.1f}%)")
    print(f"  非TAD (<0.5): {n - n_tad:,} ({(n-n_tad)/n*100:.1f}%)")
    print(f"  分数范围:     {min(scores):.4f} ~ {max(scores):.4f}")
    print(f"  平均分数:     {np.mean(scores):.4f}")
    print(f"  中位数分数:   {np.median(scores):.4f}")
    print(f"  GPU 总耗时:   {elapsed:.1f}s")
    print(f"  吞吐量:       {sps:,.0f} seqs/s ({n_windows:,.0f} windows/s)")


if __name__ == '__main__':
    main()
