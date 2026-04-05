#!/usr/bin/env python3
"""
预计算 localCider 5-mer 理化性质查找表。

对全部 20^5 = 3,200,000 个 5-mer，调用 localCider 的 SequenceParameters
计算 8 个理化性质，保存为 (3200000, 8) 的 float32 数组。

用法:
    python generate_cider_lookup.py

输出:
    cider_5mer_lookup.npy  (~100 MB)
"""
import itertools
import time
import numpy as np
from localcider.sequenceParameters import SequenceParameters

_AA = 'ACDEFGHIKLMNPQRSTVWY'
TOTAL = 20 ** 5  # 3,200,000
BATCH_SIZE = 10000

PROP_NAMES = [
    'mean_hydropathy',
    'WW_hydropathy',
    'ncpr',
    'fraction_disorder_promoting',
    'fcr',
    'mean_net_charge',
    'fraction_negative',
    'fraction_positive',
]


def compute_properties(mer5):
    sp = SequenceParameters(mer5)
    return [
        sp.get_mean_hydropathy(),
        sp.get_WW_hydropathy(),
        sp.get_NCPR(),
        sp.get_fraction_disorder_promoting(),
        sp.get_FCR(),
        sp.get_mean_net_charge(),
        sp.get_fraction_negative(),
        sp.get_fraction_positive(),
    ]


def main():
    print(f"Generating {TOTAL:,} × 8 localCider property lookup table...")
    print(f"This will take several minutes.\n")

    all_5mers = [''.join(combo) for combo in itertools.product(_AA, repeat=5)]
    lookup = np.zeros((TOTAL, 8), dtype=np.float32)

    t_start = time.time()
    errors = 0

    for batch_start in range(0, TOTAL, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, TOTAL)
        for idx in range(batch_start, batch_end):
            try:
                lookup[idx] = compute_properties(all_5mers[idx])
            except Exception:
                lookup[idx] = 0.0
                errors += 1

        elapsed = time.time() - t_start
        done = batch_end
        pct = done / TOTAL * 100
        rate = done / elapsed if elapsed > 0 else 0
        eta = (TOTAL - done) / rate if rate > 0 else float('inf')
        print(f"\r  {done:>10,} / {TOTAL:,} ({pct:.1f}%) | "
              f"{rate:.0f} 5-mer/s | ETA: {eta:.0f}s", end='', flush=True)

    total_time = time.time() - t_start
    print(f"\n\nDone in {total_time:.1f}s ({TOTAL/total_time:.0f} 5-mer/s)")
    if errors:
        print(f"  ⚠️ {errors} errors (filled with 0)")

    out_path = "cider_5mer_lookup.npy"
    np.save(out_path, lookup)
    print(f"\nSaved to {out_path} ({lookup.nbytes / 1024 / 1024:.1f} MB)")


if __name__ == '__main__':
    main()
