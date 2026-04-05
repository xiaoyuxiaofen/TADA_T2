'''
GPU-accelerated feature computation using TensorFlow.

All feature engineering runs on GPU via TF ops:
  - Amino acid counting via one-hot + explicit 5-position window slicing
  - Physicochemical properties via precomputed localCider 5-mer lookup table
  - Secondary structure via precomputed alphaPredict 5-mer lookup table
  - Kappa/Omega via NumPy vectorized computation (no localCider dependency)
  - Feature scaling via vectorized TF ops
'''
import importlib.resources
import os
import numpy as np
import tensorflow as tf


def get_scaler_path():
    scaler_arr_path = importlib.resources.files('TADA_T2.data') / 'scaler_metric.npy'
    return str(scaler_arr_path)


def get_alpha_lookup_path():
    """Path to alphaPredict 5-mer lookup table."""
    # 1. 环境变量指定
    env = os.environ.get('TADA_ALPHA_LOOKUP')
    if env and os.path.exists(env):
        return env
    # 2. 包数据目录
    try:
        p = importlib.resources.files('TADA_T2.data') / 'alpha_5mer_lookup.npy'
        if os.path.exists(str(p)):
            return str(p)
    except Exception:
        pass
    # 3. 同目录
    local = os.path.join(os.path.dirname(__file__), 'alpha_5mer_lookup.npy')
    if os.path.exists(local):
        return local
    raise FileNotFoundError(
        'alpha_5mer_lookup.npy not found. '
        'Run: python3 predict_tad.py --gen_alpha 或手动运行 generate_alpha_lookup.py'
    )


def get_cider_lookup_path():
    """Path to localCider 5-mer lookup table."""
    env = os.environ.get('TADA_CIDER_LOOKUP')
    if env and os.path.exists(env):
        return env
    try:
        p = importlib.resources.files('TADA_T2.data') / 'cider_5mer_lookup.npy'
        if os.path.exists(str(p)):
            return str(p)
    except Exception:
        pass
    local = os.path.join(os.path.dirname(__file__), 'cider_5mer_lookup.npy')
    if os.path.exists(local):
        return local
    raise FileNotFoundError(
        'cider_5mer_lookup.npy not found. '
        'Run: python3 predict_tad.py --gen_cider 或手动运行 generate_cider_lookup.py'
    )


# ═══════════════════════════════════════════════════════════════
# Lookup tables
# ═══════════════════════════════════════════════════════════════

_AA = 'ACDEFGHIKLMNPQRSTVWY'
_AA_TO_IDX = {aa: i for i, aa in enumerate(_AA)}

_CHARGE = tf.constant([
    1.0, 0.0, -1.0, -1.0, 0.0, 0.0, 1.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0
], dtype=tf.float32)

_ALIPHATICS = tf.constant([0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0], dtype=tf.float32)
_AROMATICS = tf.constant([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1], dtype=tf.float32)
_BRANCHING = tf.constant([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0], dtype=tf.float32)
_CHARGED = tf.constant([0,0,1,1,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0], dtype=tf.float32)
_NEGATIVES = tf.constant([0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=tf.float32)
_PHOSPHORYLATABLES = tf.constant([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1], dtype=tf.float32)
_POLARS = tf.constant([0,0,1,1,0,0,0,0,1,0,0,1,0,1,1,0,0,0,0,1], dtype=tf.float32)
_HYDROPHOBICS = tf.constant([0,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,1,1,0], dtype=tf.float32)
_POSITIVES = tf.constant([0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0], dtype=tf.float32)
_SULFUR_CONTAINING = tf.constant([0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0], dtype=tf.float32)
_TINYS = tf.constant([1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0], dtype=tf.float32)

_CATEGORY_MASKS = tf.stack([
    _ALIPHATICS, _AROMATICS, _BRANCHING, _CHARGED, _NEGATIVES,
    _PHOSPHORYLATABLES, _POLARS, _HYDROPHOBICS, _POSITIVES,
    _SULFUR_CONTAINING, _TINYS
], axis=0)

_ORIG_AA_PERM = tf.constant([14, 8, 2, 3, 13, 11, 6, 15, 16, 19, 1, 18, 10, 0, 7, 9, 4, 17, 12, 5], dtype=tf.int32)
_5MER_POWERS = tf.constant([160000, 8000, 400, 20, 1], dtype=tf.int32)

# ── 加载查找表（支持环境变量指定路径）──────────────────────────
_ALPHA_LOOKUP = tf.constant(np.load(get_alpha_lookup_path()), dtype=tf.float32)
_CIDER_LOOKUP = tf.constant(np.load(get_cider_lookup_path()), dtype=tf.float32)


# ═══════════════════════════════════════════════════════════════
# Sequence encoding
# ═══════════════════════════════════════════════════════════════

def _encode_sequences_to_tensor(sequences, max_len):
    batch_size = len(sequences)
    encoded = tf.fill((batch_size, max_len), -1)
    rows, cols, vals = [], [], []
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            if j >= max_len:
                break
            if aa in _AA_TO_IDX:
                rows.append(i)
                cols.append(j)
                vals.append(_AA_TO_IDX[aa])
    indices = tf.constant(list(zip(rows, cols)), dtype=tf.int64)
    values = tf.constant(vals, dtype=tf.int32)
    encoded = tf.tensor_scatter_nd_update(encoded, indices, values)
    return encoded


# ═══════════════════════════════════════════════════════════════
# GPU feature kernel
# ═══════════════════════════════════════════════════════════════

@tf.function(jit_compile=True)
def _compute_all_features_gpu(encoded, num_windows, window_size, seq_length,
                               kappa_vals, omega_vals):
    _W = 5
    one_hot = tf.one_hot(encoded, depth=20, dtype=tf.float32)
    nw = tf.shape(one_hot)[1] - _W + 1

    window_counts = (
        one_hot[:, 0:nw, :] +
        one_hot[:, 1:nw + 1, :] +
        one_hot[:, 2:nw + 2, :] +
        one_hot[:, 3:nw + 3, :] +
        one_hot[:, 4:nw + 4, :]
    )

    category_counts = tf.matmul(window_counts, tf.transpose(_CATEGORY_MASKS))

    enc_f = tf.cast(encoded, tf.float32)
    mer5 = tf.stack([
        enc_f[:, 0:nw], enc_f[:, 1:nw + 1], enc_f[:, 2:nw + 2],
        enc_f[:, 3:nw + 3], enc_f[:, 4:nw + 4],
    ], axis=2)
    mer5_idx = tf.cast(
        tf.reduce_sum(mer5 * tf.cast(_5MER_POWERS, tf.float32), axis=-1), tf.int32
    )
    mer5_idx = tf.maximum(mer5_idx, 0)

    sstructure = tf.gather(_ALPHA_LOOKUP, mer5_idx)[:, :, tf.newaxis]
    window_props = tf.gather(_CIDER_LOOKUP, mer5_idx)

    kappa_tiled = tf.tile(tf.expand_dims(tf.expand_dims(kappa_vals, 1), 1), [1, num_windows, 1])
    omega_tiled = tf.tile(tf.expand_dims(tf.expand_dims(omega_vals, 1), 1), [1, num_windows, 1])

    base_features = tf.concat([
        kappa_tiled, omega_tiled,
        window_props[:, :, :1], window_props[:, :, 1:2],
        window_props[:, :, 2:3], window_props[:, :, 3:4],
        window_props[:, :, 4:5], window_props[:, :, 5:6],
        window_props[:, :, 6:7], window_props[:, :, 7:8],
        category_counts, sstructure,
    ], axis=-1)

    window_counts_ordered = tf.gather(window_counts, _ORIG_AA_PERM, axis=-1)
    all_features = tf.concat([base_features, window_counts_ordered], axis=-1)
    return all_features


# ═══════════════════════════════════════════════════════════════
# Vectorized kappa/omega
# ═══════════════════════════════════════════════════════════════

def _compute_kappa_omega_cpu(sequences):
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


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════

def create_features(sequences, SEQUENCE_WINDOW=5, STEPS=1, LENGTH=40, PROPERTIES=42):
    max_len = LENGTH
    num_windows = (LENGTH - SEQUENCE_WINDOW) // STEPS + 1
    encoded = _encode_sequences_to_tensor(sequences, max_len)
    kappa_vals, omega_vals = _compute_kappa_omega_cpu(sequences)
    features = _compute_all_features_gpu(encoded, num_windows, SEQUENCE_WINDOW, LENGTH,
                                          kappa_vals, omega_vals)
    return features


@tf.function(jit_compile=True)
def scale_features_predict(features, scaler_mean, scaler_scale, scaler_min, scaler_range):
    scaled = (features - scaler_mean) / scaler_scale
    scaled = (scaled - scaler_min) / scaler_range
    return scaled


def load_scaler_params():
    scaler_metric = np.load(get_scaler_path())
    mean_ = tf.constant(scaler_metric[:, 0], dtype=tf.float32)
    scale_ = tf.constant(scaler_metric[:, 2], dtype=tf.float32)
    min_ = tf.constant(scaler_metric[:, 5], dtype=tf.float32)
    range_ = tf.constant(scaler_metric[:, 9], dtype=tf.float32)
    return mean_, scale_, min_, range_


_SCALER_MEAN, _SCALER_SCALE, _SCALER_MIN, _SCALER_RANGE = load_scaler_params()


def scale_features(features):
    return scale_features_predict(features, _SCALER_MEAN, _SCALER_SCALE, _SCALER_MIN, _SCALER_RANGE)
