import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.special import j0

np.random.seed(42)
torch.manual_seed(42)

# =========================
# Parameters
# =========================
N_train = 100000
N_test = 100000
EbN0_dB = np.arange(0, 21, 2)
fd_list = [1, 10, 100]
Ts = 1e-3
SEQ_LEN = 8
EPOCHS = 12
BATCH_SIZE = 256
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("figures", exist_ok=True)


def generate_time_correlated_fading(n_samples: int, fd: float, ts: float) -> np.ndarray:
    """
    Generate a simple time-correlated Rayleigh fading process
    using a Doppler-based Jakes-like autocorrelation.
    """
    R = j0(2 * np.pi * fd * ts * np.arange(n_samples))

    h_real = np.convolve(np.sqrt(np.abs(R)), np.random.randn(n_samples), mode="same")
    h_imag = np.convolve(np.sqrt(np.abs(R)), np.random.randn(n_samples), mode="same")

    h = (h_real + 1j * h_imag) / np.sqrt(2)

    power = np.mean(np.abs(h) ** 2)
    if power > 0:
        h = h / np.sqrt(power)

    return h


def generate_sequence_block(n_samples: int, fd: float, snr_db: float, ts: float):
    """
    Generate one continuous time-series block for a given Doppler and SNR.
    This preserves sequence structure for LSTM training.
    Features:
      [rx.real, rx.imag, h.real, h.imag]
    """
    bits = np.random.randint(0, 2, n_samples)
    tx = 2 * bits - 1

    h = generate_time_correlated_fading(n_samples, fd, ts)

    ebn0 = 10 ** (snr_db / 10)
    n0 = 1 / ebn0
    noise = np.sqrt(n0 / 2) * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))

    rx = h * tx + noise

    X = np.column_stack([rx.real, rx.imag, h.real, h.imag]).astype(np.float32)
    y = bits.astype(np.float32)

    return X, y


def build_mixed_training_dataset(total_samples: int, fd_choices: list, snr_choices: np.ndarray, ts: float):
    """
    Build training data from multiple continuous sequence blocks across
    different Doppler/SNR settings, then concatenate them.
    """
    num_blocks = len(fd_choices) * len(snr_choices)
    block_size = total_samples // num_blocks

    X_parts = []
    y_parts = []

    for fd in fd_choices:
        for snr_db in snr_choices:
            X_block, y_block = generate_sequence_block(block_size, fd, snr_db, ts)
            X_parts.append(X_block)
            y_parts.append(y_block)

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    return X, y


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    xs = []
    ys = []

    for i in range(len(X) - seq_len + 1):
        xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len - 1])

    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


class LSTMDetector(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)


def train_model(model, X_train, y_train):
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0

        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss = {avg_loss:.4f}")


def evaluate_classical_and_lstm(model, fd: float, ts: float, seq_len: int):
    ber_classical = []
    ber_lstm = []

    for snr_db in EbN0_dB:
        bits = np.random.randint(0, 2, N_test)
        tx = 2 * bits - 1

        h = generate_time_correlated_fading(N_test, fd, ts)

        ebn0 = 10 ** (snr_db / 10)
        n0 = 1 / ebn0
        noise = np.sqrt(n0 / 2) * (np.random.randn(N_test) + 1j * np.random.randn(N_test))

        rx = h * tx + noise

        # =========================
        # Classical detector
        # =========================
        rx_eq = rx / h
        bits_hat_classical = (np.real(rx_eq) > 0).astype(int)
        ber_c = np.mean(bits != bits_hat_classical)
        ber_classical.append(ber_c)

        # =========================
        # LSTM detector
        # =========================
        X_test = np.column_stack([rx.real, rx.imag, h.real, h.imag]).astype(np.float32)
        X_seq, y_seq = make_sequences(X_test, bits.astype(np.float32), seq_len)

        X_seq_t = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)

        model.eval()
        with torch.no_grad():
            y_pred = model(X_seq_t).cpu().numpy().flatten()

        bits_hat_lstm = (y_pred > 0.5).astype(int)
        ber_l = np.mean(y_seq.astype(int) != bits_hat_lstm)
        ber_lstm.append(ber_l)

        print(
            f"fd = {fd:3d} Hz | SNR = {snr_db:2d} dB | "
            f"Classical BER = {ber_c:.6f} | LSTM BER = {ber_l:.6f}"
        )

    return ber_classical, ber_lstm


def main():
    print(f"Using device: {DEVICE}")

    # =========================
    # Training data
    # =========================
    print("Generating training data...")
    X_train, y_train = build_mixed_training_dataset(
        total_samples=N_train,
        fd_choices=fd_list,
        snr_choices=np.arange(0, 21, 2),
        ts=Ts
    )

    X_train_seq, y_train_seq = make_sequences(X_train, y_train, SEQ_LEN)

    print("Training LSTM...")
    model = LSTMDetector().to(DEVICE)
    train_model(model, X_train_seq, y_train_seq)

    # =========================
    # Evaluation
    # =========================
    plt.figure(figsize=(8, 5))

    for fd in fd_list:
        print(f"\nEvaluating for fd = {fd} Hz")
        ber_classical, ber_lstm = evaluate_classical_and_lstm(model, fd, Ts, SEQ_LEN)

        plt.semilogy(EbN0_dB, ber_classical, 'o--', label=f'Classical fd={fd} Hz')
        plt.semilogy(EbN0_dB, ber_lstm, 's-', label=f'LSTM fd={fd} Hz')

    plt.grid(True, which='both')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('Classical vs LSTM Detection under Doppler Fading')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('figures/doppler_classical_vs_lstm.png', dpi=300)
    plt.show()

    print("\nDone. Saved: figures/doppler_classical_vs_lstm.png")


if __name__ == "__main__":
    main()