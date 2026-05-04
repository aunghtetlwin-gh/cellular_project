# BER Comparison: Classical and AI-Based BPSK Detection over Rayleigh Fading

Author: Aung Htet Lwin (st125773)  
Course project for Cellular Mobile Systems, Asian Institute of Technology (AIT).

## Overview

This project evaluates Bit Error Rate (BER) performance of BPSK over a Rayleigh fading channel using:

- A classical coherent detector with perfect CSI
- A simple neural-network-based detector (MLPClassifier)

The work focuses on two comparisons:

- Theory vs Monte Carlo simulation
- Classical vs AI-based detection

## Key Idea

Signal model:

$$
y = h x + n
$$

- $x$: transmitted BPSK symbol ($0 \rightarrow -1$, $1 \rightarrow +1$)
- $h$: Rayleigh fading coefficient
- $n$: complex AWGN

Coherent equalization and hard decision:

$$
y_{eq} = \frac{y}{h}, \quad \hat{b} = \mathbb{1}[\Re(y_{eq}) > 0]
$$

Theoretical BER for coherent BPSK in Rayleigh fading:

$$
P_b = \frac{1}{2}\left(1 - \sqrt{\frac{\gamma}{1+\gamma}}\right)
$$

## Repository Structure

```text
project/
├── baseline_bpsk_rayleigh.py      # Theory vs simulation BER curve
├── ai_detector.py                 # Classical vs AI BER curve
├── README.md
├── report_notes.md
├── note.coffee
├── Aung_Htet_Lwin_cellular_project_report.pdf
├── figures/                       # Output plots
└── refs/
```

## Requirements

- Python 3.9+
- numpy
- matplotlib
- scikit-learn

## Quick Start

1. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install numpy matplotlib scikit-learn
```

3. Run baseline simulation (theory vs simulation):

```bash
python3 baseline_bpsk_rayleigh.py
```

4. Run AI comparison (classical vs AI):

```bash
python3 ai_detector.py
```

## Outputs

Generated plots are saved to the `figures/` directory:

- `figures/theory_vs_simulation.png`
- `figures/classical_vs_ai.png`

## Result Summary

- Simulation closely follows theoretical BER for coherent BPSK in Rayleigh fading.
- The AI detector performs similarly to the classical detector in this setup.
- Under perfect CSI, the classical detector is already strong; AI adds complexity with limited BER gain.

## Notes

The AI model in `ai_detector.py` uses:

- Features: real and imaginary parts of equalized signal
- Classifier: `MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu')`

This project is intended as a clear baseline study and a starting point for more advanced scenarios (imperfect CSI, nonlinear channels, richer models, or larger feature sets).