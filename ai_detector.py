import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Reproducibility
np.random.seed(42)

# -------------------------
# Parameters
# -------------------------
N = 100000                  # Number of bits per SNR point
snr_db = np.arange(0, 21, 2)  # 0, 2, 4, ..., 20 dB

ber_classical_list = []
ber_ai_list = []

# -------------------------
# Loop over SNR values
# -------------------------
for snr in snr_db:
    snr_linear = 10 ** (snr / 10)
    noise_variance = 1 / (2 * snr_linear)

    # -------------------------
    # Transmitter
    # -------------------------
    bits = np.random.randint(0, 2, N)
    symbols = 2 * bits - 1   # BPSK: 0 -> -1, 1 -> +1

    # -------------------------
    # Rayleigh fading channel
    # -------------------------
    h = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)

    # -------------------------
    # Complex AWGN noise
    # -------------------------
    noise = np.sqrt(noise_variance) * (
        np.random.randn(N) + 1j * np.random.randn(N)
    )

    # -------------------------
    # Received signal
    # -------------------------
    y = h * symbols + noise

    # -------------------------
    # Classical coherent detection
    # -------------------------
    y_equalized = y / h
    detected_bits_classical = (np.real(y_equalized) > 0).astype(int)
    ber_classical = np.mean(bits != detected_bits_classical)
    ber_classical_list.append(ber_classical)

    # -------------------------
    # AI dataset preparation
    # Features = [real(y_equalized), imag(y_equalized)]
    # Labels = original bits
    # -------------------------
    X = np.column_stack((
        np.real(y_equalized),
        np.imag(y_equalized)
    ))
    y_labels = bits

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_labels, test_size=0.3, random_state=42
    )

    # -------------------------
    # Simple neural network detector
    # -------------------------
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        max_iter=20,
        random_state=42
    )

    model.fit(X_train, y_train)

    # AI predictions
    y_pred = model.predict(X_test)

    # AI BER
    ber_ai = np.mean(y_pred != y_test)
    ber_ai_list.append(ber_ai)

    print(f"SNR = {snr:2d} dB | Classical BER = {ber_classical:.6f} | AI BER = {ber_ai:.6f}")

# -------------------------
# Plot: Classical vs AI
# -------------------------
plt.figure(figsize=(8, 5))
plt.semilogy(snr_db, ber_classical_list, 'o-', label='Classical Detector')
plt.semilogy(snr_db, ber_ai_list, 's-', label='AI Detector')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('Classical vs AI-Based BPSK Detection over Rayleigh Fading')
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.savefig('figures/classical_vs_ai.png', dpi=300)
plt.show()

# -------------------------
# Print final results
# -------------------------
print("\nSNR values (dB):")
print(snr_db)

print("\nClassical BER:")
print(ber_classical_list)

print("\nAI BER:")
print(ber_ai_list)