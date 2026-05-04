import numpy as np
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)

# -------------------------
# Simulation parameters
# -------------------------
N = 100000
snr_db = np.arange(0, 21, 2)
ber_sim = []

# -------------------------
# Monte Carlo Simulation
# -------------------------
for snr in snr_db:
    snr_linear = 10 ** (snr / 10)
    noise_variance = 1 / (2 * snr_linear)

    # Transmitter
    bits = np.random.randint(0, 2, N)
    symbols = 2 * bits - 1   # BPSK: 0->-1, 1->+1

    # Rayleigh fading channel
    h = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)

    # Complex AWGN noise
    noise = np.sqrt(noise_variance) * (
        np.random.randn(N) + 1j * np.random.randn(N)
    )

    # Received signal
    y = h * symbols + noise

    # Coherent detection with perfect CSI
    y_equalized = y / h
    detected_bits = (np.real(y_equalized) > 0).astype(int)

    # BER
    ber = np.mean(bits != detected_bits)
    ber_sim.append(ber)

# Theoretical BER for coherent BPSK over Rayleigh fading
snr_lin = 10 ** (snr_db / 10)
ber_theory = 0.5 * (1 - np.sqrt(snr_lin / (1 + snr_lin)))

# Plot
plt.figure(figsize=(8, 5))
plt.semilogy(snr_db, ber_sim, 'o-', label='Simulation')
plt.semilogy(snr_db, ber_theory, '--', label='Theory')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BPSK over Rayleigh Fading: Theory vs Simulation')
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.savefig('figures/theory_vs_simulation.png', dpi=300)
plt.show()