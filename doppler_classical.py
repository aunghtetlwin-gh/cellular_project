import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0

np.random.seed(42)

# =========================
# Parameters
# =========================
N = 100000
EbN0_dB = np.arange(0, 21, 2)
Ts = 1e-3
fd_list = [1, 10, 100]   # slow -> medium -> fast fading

os.makedirs("figures", exist_ok=True)


def generate_time_correlated_fading(n_samples: int, fd: float, ts: float) -> np.ndarray:
    """
    Generate a simple time-correlated Rayleigh fading process
    using Jakes-like autocorrelation with Bessel J0.
    """
    R = j0(2 * np.pi * fd * ts * np.arange(n_samples))

    h_real = np.convolve(np.sqrt(np.abs(R)), np.random.randn(n_samples), mode="same")
    h_imag = np.convolve(np.sqrt(np.abs(R)), np.random.randn(n_samples), mode="same")

    h = (h_real + 1j * h_imag) / np.sqrt(2)

    # Normalize average power
    power = np.mean(np.abs(h) ** 2)
    if power > 0:
        h = h / np.sqrt(power)

    return h


def theoretical_rayleigh_ber(ebn0_db: np.ndarray) -> np.ndarray:
    ebn0_linear = 10 ** (ebn0_db / 10)
    return 0.5 * (1 - np.sqrt(ebn0_linear / (1 + ebn0_linear)))


def main():
    ber = np.zeros((len(fd_list), len(EbN0_dB)))

    # Fixed bit sequence for fair comparison
    bits = np.random.randint(0, 2, N)
    tx = 2 * bits - 1

    for k, fd in enumerate(fd_list):
        print(f"\nSimulating Doppler frequency fd = {fd} Hz")

        h = generate_time_correlated_fading(N, fd, Ts)

        for i, snr_db in enumerate(EbN0_dB):
            ebn0 = 10 ** (snr_db / 10)
            n0 = 1 / ebn0

            noise = np.sqrt(n0 / 2) * (
                np.random.randn(N) + 1j * np.random.randn(N)
            )

            rx = h * tx + noise

            # Perfect CSI equalization
            rx_eq = rx / h

            bits_hat = (np.real(rx_eq) > 0).astype(int)
            ber[k, i] = np.mean(bits != bits_hat)

            print(f"SNR = {snr_db:2d} dB | BER = {ber[k, i]:.6f}")

    # Plot Doppler cases
    plt.figure(figsize=(8, 5))
    for k, fd in enumerate(fd_list):
        plt.semilogy(EbN0_dB, ber[k], marker='o', label=f'fd = {fd} Hz')

    # Optional: add theory reference
    ber_theory = theoretical_rayleigh_ber(EbN0_dB)
    plt.semilogy(EbN0_dB, ber_theory, 'k--', label='Theory (i.i.d. Rayleigh)')

    plt.grid(True, which='both')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BPSK over Time-Varying Rayleigh Fading with Doppler')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/doppler_classical_ber.png', dpi=300)
    plt.show()

    print("\nDone. Saved: figures/doppler_classical_ber.png")


if __name__ == "__main__":
    main()