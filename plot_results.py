# plot_results.py
#
# Renders results.csv (produced by the C++ simulation) as a four-panel
# figure: the input/output waveforms zoomed to a couple of cycles (so the
# tube stages' asymmetric clipping is actually visible), the input/output
# transfer curve (the clearest single view of that asymmetry), and the
# full-length output spectrum with harmonics annotated.

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import signal

# Palette: validated categorical slots (blue / orange / aqua) plus chart
# chrome, one set per surface. See dataviz palette reference for the
# validation this ordering clears (CVD-safe adjacent pairs, contrast).
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink_primary": "#0b0b0b",
        "ink_secondary": "#52514e",
        "ink_muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "reference": "#c3c2b7",
        "series_input": "#2a78d6",
        "series_output": "#eb6834",
        "series_transfer": "#1baf7a",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink_primary": "#ffffff",
        "ink_secondary": "#c3c2b7",
        "ink_muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "reference": "#383835",
        "series_input": "#3987e5",
        "series_output": "#d95926",
        "series_transfer": "#199e70",
    },
}

CYCLES_TO_SHOW = 2
N_HARMONICS = 10
HARMONIC_FLOOR_DB = -80.0  # ignore harmonic peaks below this relative level


def load_results(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise SystemExit(
            f"Error: {csv_path} not found.\n"
            "Run the C++ simulation executable first to generate it."
        )
    sample_rate = 1.0 / (df["time"].iloc[1] - df["time"].iloc[0])
    return df, sample_rate


def compute_spectrum(y, sample_rate):
    freqs, psd = signal.periodogram(y, sample_rate, window="hann")
    return freqs, psd


def find_fundamental(freqs, psd, min_freq=50.0):
    start = np.searchsorted(freqs, min_freq)
    peak_idx = start + np.argmax(psd[start:])
    return freqs[peak_idx], psd[peak_idx]


def find_harmonics(freqs, psd, fundamental_freq, fundamental_power, n_harmonics):
    """Locate the local PSD peak nearest each expected harmonic and keep the
    ones that actually rise above the noise floor."""
    bin_width = freqs[1] - freqs[0]
    search_bins = max(1, int(round((fundamental_freq * 0.05) / bin_width)))
    harmonics = []
    for k in range(1, n_harmonics + 1):
        target_freq = k * fundamental_freq
        if target_freq > freqs[-1]:
            break
        center = int(round(target_freq / bin_width))
        lo, hi = max(0, center - search_bins), min(len(psd), center + search_bins + 1)
        local_idx = lo + np.argmax(psd[lo:hi])
        level_db = 10 * np.log10(psd[local_idx] / fundamental_power)
        if level_db >= HARMONIC_FLOOR_DB:
            harmonics.append((k, freqs[local_idx], psd[local_idx], level_db))
    return harmonics


def estimate_thd(harmonics):
    """Approximate THD (%) from periodogram peak levels: ratio of RMS of
    harmonics 2..N to the fundamental. Illustrative, not lab-grade."""
    if len(harmonics) < 2:
        return None
    fundamental_power = harmonics[0][2]
    distortion_power = sum(p for _, _, p, _ in harmonics[1:])
    return 100.0 * np.sqrt(distortion_power / fundamental_power)


def style_axes(ax, theme):
    ax.set_facecolor(theme["surface"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(theme["axis"])
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors=theme["ink_muted"], labelsize=9)
    ax.xaxis.label.set_color(theme["ink_secondary"])
    ax.yaxis.label.set_color(theme["ink_secondary"])
    ax.grid(True, color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def plot_waveform(ax, t_ms, y, color, theme, title, ylabel):
    ax.plot(t_ms, y, color=color, linewidth=1.8, solid_joinstyle="round")
    ax.axhline(0, color=theme["reference"], linewidth=0.8, zorder=0)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=theme["ink_primary"], fontsize=11, loc="left")
    style_axes(ax, theme)


def plot_transfer_curve(ax, x_in, y_out, theme):
    ax.plot(x_in, y_out, color=theme["series_transfer"], linewidth=1.4, alpha=0.85)
    ax.axhline(0, color=theme["reference"], linewidth=0.8, zorder=0)
    ax.axvline(0, color=theme["reference"], linewidth=0.8, zorder=0)
    ax.set_xlabel("Input (V)")
    ax.set_ylabel("Output (V)")
    ax.set_title(
        "Transfer Curve (asymmetric clipping)",
        color=theme["ink_primary"], fontsize=11, loc="left",
    )
    style_axes(ax, theme)


def plot_spectrum(ax, freqs, psd, fundamental_power, harmonics, theme):
    psd_db = 10 * np.log10(np.maximum(psd, 1e-300) / fundamental_power)
    ax.plot(freqs, psd_db, color=theme["series_output"], linewidth=1.4)

    for k, freq, _, level_db in harmonics:
        ax.axvline(freq, color=theme["axis"], linewidth=0.8, linestyle=(0, (1, 2)))
        label = "f₀" if k == 1 else f"{k}f₀"
        ax.annotate(
            label,
            xy=(freq, level_db),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=theme["ink_secondary"],
        )

    thd = estimate_thd(harmonics)
    if thd is not None:
        ax.text(
            0.98, 0.94, f"THD ≈ {thd:.1f}% (approx., through H{harmonics[-1][0]})",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=theme["ink_secondary"],
        )

    ax.set_xlim(0, 5000)
    ax.set_ylim(-140, 10)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Level (dBc)")
    ax.set_title("Output Spectrum", color=theme["ink_primary"], fontsize=11, loc="left")
    ax.yaxis.set_major_locator(MultipleLocator(20))
    style_axes(ax, theme)


def build_figure(df, sample_rate, theme_name):
    theme = THEMES[theme_name]

    y_out = df["output"].to_numpy()
    freqs, psd = compute_spectrum(y_out, sample_rate)
    fundamental_freq, fundamental_power = find_fundamental(freqs, psd)
    harmonics = find_harmonics(freqs, psd, fundamental_freq, fundamental_power, N_HARMONICS)

    window_duration = CYCLES_TO_SHOW / fundamental_freq
    t0 = df["time"].iloc[0]
    df_wave = df[df["time"] < t0 + window_duration]
    t_ms = (df_wave["time"] - t0) * 1000.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor(theme["surface"])

    plot_waveform(axes[0, 0], t_ms, df_wave["input"], theme["series_input"], theme, "Input Signal", "Input (V)")
    plot_waveform(axes[0, 1], t_ms, df_wave["output"], theme["series_output"], theme, "Amplifier Output", "Output (V)")
    plot_transfer_curve(axes[1, 0], df["input"], df["output"], theme)
    plot_spectrum(axes[1, 1], freqs, psd, fundamental_power, harmonics, theme)

    fig.suptitle(
        f"Tube Preamp — K-Method Simulation  (fundamental ≈ {fundamental_freq:.0f} Hz)",
        color=theme["ink_primary"], fontsize=13, x=0.02, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="results.csv", help="Input CSV from the simulator")
    parser.add_argument("--out", default="results.png", help="Output image path")
    parser.add_argument("--theme", choices=THEMES.keys(), default="light")
    parser.add_argument("--no-show", action="store_true", help="Skip the interactive window")
    args = parser.parse_args()

    df, sample_rate = load_results(args.csv)
    print(f"Detected sample rate: {sample_rate:.2f} Hz")

    fig = build_figure(df, sample_rate, args.theme)
    fig.savefig(args.out, dpi=200, facecolor=fig.get_facecolor())
    print(f"Saved {args.out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
