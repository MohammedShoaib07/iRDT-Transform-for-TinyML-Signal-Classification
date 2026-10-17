# hardware_deploy.py
# Run this on your local machine (or Raspberry Pi) after downloading irdt_model.pkl
# It loads the trained model, runs inference, and benchmarks latency + memory.

import numpy as np
import pickle
import time
import os
import sys

sys.path.append(os.path.dirname(__file__))
from irdt_core import irdt_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'irdt_model.pkl')


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        print(f"[ERROR] Model file not found: {path}")
        print("  -> Run train_colab.py in Colab first, then download irdt_model.pkl here.")
        return None
    with open(path, 'rb') as f:
        results = pickle.load(f)
    print(f"[OK] Loaded model from {path}")
    print(f"     Domains available: {list(results.keys())}")
    return results


def benchmark_irdt(signal, chan, M=8, w=256, n_runs=500):
    """Measure average inference time for iRDT feature extraction."""
    # warm-up
    for _ in range(5):
        irdt_features(signal, M=M, w=w, chan=chan)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        irdt_features(signal, M=M, w=w, chan=chan)
    elapsed = (time.perf_counter() - t0) / n_runs * 1000  # ms per call
    return elapsed


def run_single_inference(signal, domain_result):
    """Extract features + classify one signal. Returns label + latency."""
    chan = domain_result['best_chan']
    M    = domain_result['M']
    w    = domain_result['w']
    clf  = domain_result['model']

    t0   = time.perf_counter()
    feat = irdt_features(signal, M=M, w=w, chan=chan).reshape(1, -1)
    pred = clf.predict(feat)[0]
    lat  = (time.perf_counter() - t0) * 1000  # ms
    return pred, lat


def memory_footprint(domain_result):
    """Rough model size in KB."""
    import io
    buf = io.BytesIO()
    pickle.dump(domain_result['model'], buf)
    return buf.tell() / 1024


def main():
    results = load_model()
    if results is None:
        return

    print("\n" + "="*60)
    print("  iRDT Hardware Benchmark")
    print("="*60)

    for domain, res in results.items():
        chan = res['best_chan']
        M    = res['M']
        w    = res['w']

        # Synthetic test signal (replace with real sensor data)
        seg_len = w * (M + 4)
        signal  = np.random.randn(seg_len).astype(np.float32) * 0.3

        # Latency benchmark
        lat_ms = benchmark_irdt(signal, chan, M=M, w=w, n_runs=500)

        # Single inference
        pred, inf_ms = run_single_inference(signal, res)

        # Model size
        model_kb = memory_footprint(res)

        print(f"\n  Domain : {domain}")
        print(f"  Delays : {chan}")
        print(f"  iRDT feature extraction : {lat_ms:.3f} ms  (avg over 500 runs)")
        print(f"  Full inference (feat+clf): {inf_ms:.3f} ms")
        print(f"  Model size               : {model_kb:.1f} KB")
        print(f"  5-fold CV accuracy       : {res['opt_acc']:.4f}")
        print(f"  Baseline (fixed delays)  : {res['fixed_acc']:.4f}")
        gain = (res['opt_acc'] - res['fixed_acc']) * 100
        print(f"  Delay optimization gain  : {gain:+.2f}%")

    print("\n" + "="*60)
    print("  Benchmark complete.")
    print("="*60)


if __name__ == '__main__':
    main()
