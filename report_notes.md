Project topic:
BER Performance Comparison of Classical and AI-Based BPSK Detection over Rayleigh Fading Channels

BPSK (Binary Phase Shift Keying) is a binary digital modulation scheme in which bit 0 and bit 1 are mapped to two opposite signal phases, commonly represented as -1 and +1.


Baseline:
Coherent BPSK detection over Rayleigh fading with perfect CSI

In simple flow:

generate random bits
map bits to BPSK symbols
pass through Rayleigh fading channel
add noise
equalize using channel value
detect bit
compute BER
compare with theoretical BER


Metric:
Bit Error Rate (BER)

Theory:
Received signal y = h*x + n
where h is Rayleigh fading coefficient and n is complex AWGN noise

Detection:
Equalize by y/h
If real(y/h) > 0, detect bit 1
Else detect bit 0


Simulation Parameters
- Modulation: BPSK
- Channel: Rayleigh fading
- Noise: Complex AWGN
- Detection: Coherent detection with perfect CSI
- Number of transmitted bits: 100000
- SNR range: 0 dB to 20 dB with step size 2 dB
- Performance metric: BER


Progress Update

Completed:
- Selected project topic
- Understood basic concepts: BPSK, Rayleigh fading, BER
- Implemented classical BPSK over Rayleigh fading simulation
- Validated simulation against theoretical BER
- Generated BER vs SNR plot showing close agreement between theory and simulation

Current Stage:
- Preparing AI-based detector using a simple neural network
- Defining dataset features from equalized received signal
- Planning fair BER comparison between classical and AI-based detectors

Next Steps:
- Build AI dataset using real and imaginary parts of equalized signal
- Train a simple feedforward neural network
- Evaluate AI detector BER across SNR values
- Compare BER and complexity between classical and AI-based methods