# plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def main():
    """Reads simulation data from CSV and plots it."""
    try:
        df = pd.read_csv("results.csv")
    except FileNotFoundError:
        print("Error: results.csv not found.")
        print("Please run the C++ simulation executable first.")
        return

    sample_rate = 1.0 / (df['time'].iloc[1] - df['time'].iloc[0])
    print(f"Detected sample rate: {sample_rate:.2f} Hz")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=not True)
    
    # Plot waveforms
    axes[0].plot(df['time'], df['input'])
    axes[0].set_title("Input Signal")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(df['time'], df['output'])
    axes[1].set_title("Amplifier Output")
    axes[1].grid(True, alpha=0.3)

    # Plot spectrum
    y_raw = df['output'].to_numpy()
    freqs, psd = signal.periodogram(y_raw, sample_rate, window="hann")
    axes[2].semilogy(freqs, psd)
    axes[2].set_title("Output Spectrum")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_xlim(0, 5000)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
