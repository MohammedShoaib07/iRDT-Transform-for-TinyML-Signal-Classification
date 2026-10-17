import numpy as np
from numba import jit
from scipy.optimize import differential_evolution

@jit(nopython=True)
def irdt(signal: np.int32, M=8, w=256, chan=(1, 2, 4, 8, 16, 32, 64, 100)):
    """
    Improved Reaction-Diffusion Transform.

    Parameters
    ----------
    signal : 1-D int32 array
    M      : number of time segments in output
    w      : window size (power of 2 recommended)
    chan   : tuple of delay values (filter banks)

    Returns
    -------
    feat : (M, len(chan)) int32 array  — the iRDT spectrogram
    """
    m = len(chan)
    windows = len(signal) // w
    spectrum = np.zeros((windows, m), dtype=np.int32)
    limit = w // 4

    for i in range(windows):
        base = i * w
        for k in range(m):
            d = chan[k]
            s = 0
            for t in range(limit, w - limit):
                s += abs(signal[base + t - d] + signal[base + t + d]
                         - 2 * signal[base + t])
            spectrum[i, k] = s

    fer = (windows - 2) // M
    feat = np.zeros((M, m), dtype=np.int32)
    lin = 1
    for k in range(M):
        for col in range(m):
            s = 0
            for i in range(lin, lin + fer):
                s += spectrum[i, col]
            feat[k, col] = s // (2 * w)
        lin += fer
    return feat


# ─────────────────────────────────────────────
# 2.  Float wrapper (for sklearn / numpy pipelines)
# ─────────────────────────────────────────────
def irdt_features(signal_float, M=8, w=256, chan=(1, 2, 4, 8, 16, 32, 64, 100)):
    """
    Accepts a float signal in [-1, 1], converts to int32, returns
    flattened float32 feature vector.
    """
    sig_int = (signal_float * 32767).astype(np.int32)
    feat = irdt(sig_int, M=M, w=w, chan=tuple(int(c) for c in chan))
    return feat.astype(np.float32).flatten()


# ─────────────────────────────────────────────
# 3.  Delay optimizer  (Differential Evolution)
# ─────────────────────────────────────────────
def optimize_delays(X, y, n_channels=6, M=8, w=256,
                    max_delay=None, popsize=8, maxiter=30, seed=42):
    """
    Find the best delay vector for a given dataset using
    Differential Evolution + a fast 1-NN accuracy proxy.

    Parameters
    ----------
    X          : list/array of 1-D float signals (already normalised to [-1,1])
    y          : integer class labels
    n_channels : number of delay channels to optimise
    M, w       : iRDT parameters
    max_delay  : maximum allowed delay (default: w//2 - 1)
    popsize    : DE population size multiplier
    maxiter    : DE max iterations

    Returns
    -------
    best_chan : tuple of int delays (sorted)
    best_acc  : float, cross-val accuracy with best delays
    history   : list of (iteration, accuracy) tuples
    """
    if max_delay is None:
        max_delay = w // 2 - 1

    X = [np.array(x, dtype=np.float32) for x in X]
    y = np.array(y)
    history = []

    def _extract(chan_float):
        chan = tuple(sorted(set(max(1, int(round(c))) for c in chan_float)))
        if len(chan) < 2:
            return None, None
        feats = np.array([irdt_features(x, M=M, w=w, chan=chan) for x in X],
                         dtype=np.float32)
        return chan, feats

    def _knn1_acc(feats, labels):
        """Leave-one-out 1-NN accuracy (fast, no sklearn needed)."""
        n = len(feats)
        correct = 0
        for i in range(n):
            dists = np.sum((feats - feats[i]) ** 2, axis=1)
            dists[i] = np.inf
            if labels[np.argmin(dists)] == labels[i]:
                correct += 1
        return correct / n

    call_count = [0]

    def objective(chan_float):
        chan, feats = _extract(chan_float)
        if feats is None:
            return 1.0
        acc = _knn1_acc(feats, y)
        call_count[0] += 1
        if call_count[0] % popsize == 0:
            history.append((call_count[0], round(acc, 4)))
            print(f"  DE iter ~{call_count[0]//popsize:3d}  "
                  f"channels={chan}  1-NN acc={acc:.4f}")
        return 1.0 - acc   # minimise

    bounds = [(1, max_delay)] * n_channels
    result = differential_evolution(
        objective, bounds,
        popsize=popsize, maxiter=maxiter,
        seed=seed, tol=1e-4, polish=False,
        mutation=(0.5, 1.0), recombination=0.7
    )

    best_chan, _ = _extract(result.x)
    best_acc = 1.0 - result.fun
    print(f"\nBest delays : {best_chan}")
    print(f"Best 1-NN acc: {best_acc:.4f}")
    return best_chan, best_acc, history
