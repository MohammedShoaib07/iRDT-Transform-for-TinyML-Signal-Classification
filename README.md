# iRDT Transform for TinyML Signal Classifiers

This repository contains an efficient, lightweight feature-extraction pipeline based on the Improved Reaction-Diffusion Transform (iRDT). The project targets low-complexity signal classification for TinyML and embedded deployment, with emphasis on hardware-oriented processing and minimal computational cost.

The implementation is designed for high-speed feature extraction without multiplication-heavy operations and is suitable for audio, biomedical, and other time-series signals.

<p align="center">
  <img width="480" height="360" alt="RDT explanation" src="https://github.com/user-attachments/assets/2a6c6523-7a5d-4153-9dbc-569c8dc30f91" />
</p>

## Why iRDT?

The iRDT transform offers a compact and efficient alternative to conventional feature extractors such as MFCCs. It is especially useful when:

- computational resources are limited,
- deployment targets are MCU, FPGA, or edge devices,
- signal classification must be fast and energy-efficient,
- a compact feature representation is preferred over heavy DSP pipelines.

The transform is intentionally simple, highly parallelizable, and compatible with fixed-point / integer implementations, which makes it attractive for TinyML workflows.

## Project Overview

This repository includes:

- the core iRDT implementation in Python,
- delay optimization utilities for channel selection,
- example models and preprocessing logic,
- a keyword spotting demonstration,
- Arduino-compatible deployment support for embedded inference.

This project extends and revises earlier RDT-based work with a simplified and optimized implementation intended for modern low-power ML pipelines.

## Repository Layout

- `irdt_core.py` — core iRDT transform, feature extraction, and optimizer utilities
- `hardware_deploy.py` — benchmark and inference helper for local/embedded deployment
- `convert_to_header.py` — converts a TensorFlow Lite model to Arduino header format
- `irdt_arduino/` — Arduino sketch and generated model data for embedded inference
- `Keyword_Spotting/` — keyword-spotting example assets and demo notebook
- `nrdt_2025.ipynb` and `train_colab.ipynb` — notebook-based training and experimentation workflows
- `irdt_model.pkl` and `irdt_ecg_model.tflite` — trained model artifacts

## Features

- Multiplication-free or low-complexity transform design
- JIT-friendly implementation for improved runtime speed
- Delay optimization using differential evolution
- Support for integer and floating-point feature extraction
- Compatibility with ML pipelines and embedded deployment
- Suitable for audio and biomedical signal classification tasks

## Getting Started

### Requirements

The project is built using Python and depends on common scientific ML packages:

- Python 3.x
- NumPy
- SciPy
- Numba
- scikit-learn (optional, depending on workflow)

Install dependencies with:

```bash
pip install numpy scipy numba scikit-learn
```

### Basic Usage

```python
import numpy as np
from irdt_core import irdt_features

signal = np.random.randn(1024).astype(np.float32)
features = irdt_features(signal, M=8, w=256, chan=(1, 2, 4, 8, 16, 32, 64, 100))
print(features.shape)
```

### Delay Optimization

```python
from irdt_core import optimize_delays

best_chan, best_acc, history = optimize_delays(
    X, y,
    n_channels=6,
    M=8,
    w=256,
    popsize=8,
    maxiter=30,
    seed=42,
)

print(best_chan)
print(best_acc)
```

## Example Applications

The repository includes several use cases:

- keyword spotting,
- sound or speech classification,
- ECG or biomedical signal analysis,
- hardware-aware inference for edge devices.

The keyword spotting example can be opened directly from the notebooks in the `Keyword_Spotting/` folder.

## Embedded and Hardware Deployment

The project also includes deployment support for microcontroller-based inference:

- `hardware_deploy.py` benchmarks latency and memory footprint,
- `convert_to_header.py` converts a `.tflite` model to Arduino C++ header format,
- `irdt_arduino/irdt_arduino.ino` demonstrates inference on an Arduino-compatible device.

This makes the repository useful for TinyML prototyping and low-power edge deployment.

## Citation and Related Work

### Relevant papers

[0] R. Dogaru and I. Dogaru, "Why RDT? Computational aspects in favor of the RDT transform," 2025.  
[PDF](https://github.com/radu-dogaru/rdt_transform_for_tiny_ml_signal_classifiers/blob/main/why_RDT.pdf)

[1] R. Dogaru and I. Dogaru, "A low complexity solution for epilepsy detection using an improved version of the reaction-diffusion transform," 2017 5th International Symposium on Electrical and Electronics Engineering (ISEEE), Galati, Romania, 2017, pp. 1-6.  
https://ieeexplore.ieee.org/document/8170678

[2] R. Dogaru and I. Dogaru, "State of the Art Recognition of Emotions from Speech, Using a Low Complexity Solution Based on Reaction-Diffusion Transform," 2022 International Symposium on Electronics and Telecommunications (ISETC), Timisoara, Romania, 2022, pp. 1-4.  
https://ieeexplore.ieee.org/document/10010234

## History

The reaction-diffusion transform concept emerged from cellular automata and emergent behavior studies in the early 2000s. It was later extended for sound and signal recognition tasks, where it proved to be a strong low-complexity alternative to more computationally demanding spectral methods.

This repository reflects the continued evolution of that idea into a compact, hardware-friendly TinyML-oriented feature transform.

## License

This project is distributed under the terms of the LICENSE file in the repository root.

## Contact

Copyright Radu and Ioana Dogaru  
Email: radu.dogaru@upb.ro

## Colab Notebook

For easier access to datasets and interactive experiments, the main notebook can be launched in Google Colab:

<a href="https://colab.research.google.com/github/radu-dogaru/rdt_transform_for_tiny_ml_signal_classifiers/blob/main/nrdt_2025.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


