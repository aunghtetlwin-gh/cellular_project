# BER Performance Comparison of Classical and AI-Based BPSK Detection over Time-Varying Rayleigh Fading Channels with Doppler Effect

## Project Overview

This project investigates bit error rate (BER) performance for BPSK detection under realistic wireless channel conditions characterized by time-varying Rayleigh fading with Doppler effects. The study compares three detection approaches:

1. **Classical Detection**: Coherent receiver with perfect channel state information (CSI)
2. **AI-Based Detection (MLP)**: Machine learning classifier trained on equalized signals
3. **AI-Based Detection (LSTM)**: Time-aware neural network exploiting sequential channel correlation

### Motivation

In mobile communication systems, receiver mobility introduces Doppler frequency shift, causing the channel to vary rapidly with time. This project extends standard static fading analysis to **time-correlated fading** scenarios, reflecting real-world conditions across slow-to-fast mobility regimes.

## Requirements

- Python 3.9+
- numpy
- matplotlib
- scikit-learn
- scipy
- torch

## Quick Start

1. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install numpy matplotlib scikit-learn scipy torch
```

3. Run simulations:

   - **Baseline static Rayleigh fading analysis:**
     ```bash
     python3 baseline_bpsk_rayleigh.py
     ```

   - **Classical detector under Doppler fading:**
     ```bash
     python3 doppler_classical.py
     ```

   - **AI comparison in static scenario:**
     ```bash
     python3 ai_detector.py
     ```

   - **Classical vs LSTM under time-varying Doppler:**
     ```bash
     python3 doppler_lstm.py
     ```

## Project Files

| File | Purpose |
|------|---------|
| `baseline_bpsk_rayleigh.py` | Baseline BPSK simulation over i.i.d. Rayleigh fading; compares theory vs simulation |
| `ai_detector.py` | MLP-based detector trained on equalized signals in static scenario |
| `doppler_classical.py` | Classical coherent BPSK detection under time-varying Rayleigh fading with controlled Doppler spread |
| `doppler_lstm.py` | LSTM-based detector trained and evaluated under multiple Doppler frequencies; compares performance with classical detector |
| `.gitignore` | Git ignore rules to exclude virtual environment and Python cache files |

## Key Features

### Time-Varying Fading Model
- **Jakes autocorrelation** with Bessel function for realistic channel correlation
- **Doppler frequency** controls fade rate: slow fading (fd = 1 Hz), medium (fd = 10 Hz), fast (fd = 100 Hz)
- Normalized channel power for fair comparison across scenarios

### Classical Detector
- Perfect coherent detection with zero-forcing equalization
- Assumes perfect CSI
- Baseline for AI-based approaches

### LSTM Detector
- Input features: real/imaginary parts of received signal and channel estimate
- Temporal context: 8-symbol window exploiting channel correlation
- Training on mixed SNR/Doppler conditions for robustness
- Batch training with BCE loss on 60K training samples

## Generated Outputs

All figures are saved to the `figures/` directory:

- `figures/theory_vs_simulation.png` — Theory vs simulation BER in static i.i.d. Rayleigh
- `figures/classical_vs_ai.png` — Classical vs MLP detector comparison
- `figures/doppler_classical_ber.png` — Classical detector BER across Doppler frequencies
- `figures/doppler_classical_vs_lstm.png` — Classical vs LSTM comparison under Doppler fading

## Key Results

### Static Rayleigh Fading
- Simulation accurately follows theoretical BER for coherent BPSK
- MLP detector performance is comparable to classical detector with perfect CSI
- Under perfect CSI, simple ML models show limited BER gain

### Time-Varying Doppler Fading
- **Performance degrades** with increasing Doppler frequency (faster channel variation)
- **Slow fading (fd = 1 Hz)**: BER remains relatively stable across SNR
- **Fast fading (fd = 100 Hz)**: Significant performance degradation due to rapid channel changes
- **LSTM capability**: Successfully learns temporal channel patterns; maintains reasonable BER under time-varying conditions
- **Classical detector**: Robust due to perfect CSI but does not leverage sequential structure

### Classical vs LSTM Comparison
- **Perfect CSI advantage**: Classical detector benefits from direct channel knowledge
- **LSTM learning**: Exploits channel correlation over time; particularly effective for moderate Doppler scenarios
- **Trade-off**: LSTM offers flexibility but cannot fully compensate for imperfect CSI in extreme mobility conditions

## Insights & Future Work

### Insights
1. Time-correlated fading fundamentally differs from i.i.d. scenarios; detection performance is Doppler-dependent
2. LSTM-based detection is viable and relevant for sequential fading channels
3. Classical detector with perfect CSI remains highly competitive; AI advantage emerges under realistic imperfect CSI

### Future Improvements
- Evaluate under **imperfect/delayed CSI** and channel estimation error
- Integrate **explicit channel estimation** into the receiver pipeline
- Explore **deeper LSTM architectures** and attention mechanisms
- Investigate **non-linear channels** and OFDM scenarios
- Analyze **computational complexity** trade-offs between classical and AI approaches

## References

- Jakes, W. C., & Cox, D. C. (1991). Microwave Mobile Communications. IEEE Press.
- Goldsmith, A. (2005). Wireless Communications. Cambridge University Press.
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.

---

**Author**: Aung Htet Lwin (st125773@ait.asia)  
**Course**: Cellular Mobile Systems  
**Date**: May 2026