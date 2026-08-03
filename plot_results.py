# plot_results.py
#
# Renders results.csv (produced by the C++ simulation) as a six-panel
# diagnostic figure:
#
#   1. Input waveform (zoomed to a couple of cycles)
#   2. Output waveform with the quiescent level and both clip ceilings marked
#   3. Input/output transfer curve with the small-signal tangent, so
#      compression reads as departure from the straight line
#   4. Incremental gain vs input voltage (the compression knees directly)
#   5. Output spectrum with harmonics annotated and estimated THD
#   6. Harmonic levels as bars, split even vs odd (asymmetry vs symmetry
#      of the clipping)
#
# A numeric summary (operating point, gain, clip levels, THD, even/odd
# distortion split) is printed to the console for log-friendly analysis.

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import signal
from scipy.signal import savgol_filter

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
GAIN_CURVE_BINS = 120
# Inputs this close to zero count as "near quiescent" for the operating-point
# and small-signal-gain estimates, as a fraction of the input peak.
QUIESCENT_INPUT_FRACTION = 0.02
SMALL_SIGNAL_INPUT_FRACTION = 0.10


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


# -----------------------------------------------------------------------------
#                            SIGNAL ANALYSIS
# -----------------------------------------------------------------------------

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


def even_odd_split(harmonics):
    """RSS distortion level (dBc) of even and odd harmonics >= 2, or None if
    that parity contributed nothing above the floor."""
    fundamental_power = harmonics[0][2]
    result = {}
    for parity, name in ((0, "even"), (1, "odd")):
        power = sum(p for k, _, p, _ in harmonics[1:] if k % 2 == parity)
        result[name] = (
            10 * np.log10(power / fundamental_power) if power > 0 else None
        )
    return result


def analyze_operating_point(x_in, y_out):
    """Quiescent level, clip extremes, and small-signal gain, estimated from
    the recorded waveforms alone (no access to the internals of the sim).

    The quiescent output is read off samples where the input is ~zero; with
    the cathode cap effectively a short at audio rates the transfer curve is
    single-valued, so both zero-crossing directions agree.
    """
    peak_in = np.max(np.abs(x_in))
    near_zero = np.abs(x_in) < QUIESCENT_INPUT_FRACTION * peak_in
    quiescent = float(np.mean(y_out[near_zero])) if near_zero.any() else float(np.mean(y_out))

    small = np.abs(x_in) < SMALL_SIGNAL_INPUT_FRACTION * peak_in
    if np.count_nonzero(small) >= 2:
        gain = float(np.polyfit(x_in[small], y_out[small], 1)[0])
    else:
        gain = None

    return {
        "quiescent": quiescent,
        "out_min": float(np.min(y_out)),
        "out_max": float(np.max(y_out)),
        "gain": gain,
        "peak_in": float(peak_in),
    }


def incremental_gain_curve(x_in, y_out, n_bins=GAIN_CURVE_BINS):
    """dVout/dVin as a function of input voltage, from the binned transfer
    curve. Bins with too few samples (the extreme edges) are dropped."""
    order = np.argsort(x_in)
    xs, ys = x_in[order], y_out[order]
    edges = np.linspace(xs[0], xs[-1], n_bins + 1)
    idx = np.clip(np.searchsorted(edges, xs, side="right") - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    valid = counts > 3
    x_mean = np.bincount(idx, weights=xs, minlength=n_bins)[valid] / counts[valid]
    y_mean = np.bincount(idx, weights=ys, minlength=n_bins)[valid] / counts[valid]
    x_mid = x_mean[:-1] + np.diff(x_mean) / 2
    gain = np.diff(y_mean) / np.diff(x_mean)
    # Differencing bin means amplifies sampling noise; a light polynomial
    # smoothing keeps the knees while removing the jitter.
    if len(gain) >= 11:
        gain = savgol_filter(gain, 11, 2)
    return x_mid, gain


# -----------------------------------------------------------------------------
#                              PLOT PANELS
# -----------------------------------------------------------------------------

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


def annotate_level(ax, y, text, theme, va="bottom"):
    ax.axhline(y, color=theme["reference"], linewidth=0.9, linestyle=(0, (4, 3)), zorder=0)
    ax.annotate(
        text, xy=(1.0, y), xycoords=("axes fraction", "data"),
        xytext=(-4, 3 if va == "bottom" else -3), textcoords="offset points",
        ha="right", va=va, fontsize=8, color=theme["ink_muted"],
    )


def plot_input_waveform(ax, t_ms, x_in, theme):
    ax.plot(t_ms, x_in, color=theme["series_input"], linewidth=1.8, solid_joinstyle="round")
    ax.axhline(0, color=theme["reference"], linewidth=0.8, zorder=0)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Input (V)")
    ax.set_title("Input Signal", color=theme["ink_primary"], fontsize=11, loc="left")
    style_axes(ax, theme)


def plot_output_waveform(ax, t_ms, y_out, op, theme):
    ax.plot(t_ms, y_out, color=theme["series_output"], linewidth=1.8, solid_joinstyle="round")
    annotate_level(ax, op["out_max"], f"clip {op['out_max']:.0f} V", theme, va="bottom")
    annotate_level(ax, op["quiescent"], f"quiescent {op['quiescent']:.0f} V", theme, va="bottom")
    annotate_level(ax, op["out_min"], f"clip {op['out_min']:.0f} V", theme, va="top")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Output (V)")
    ax.set_title("Amplifier Output", color=theme["ink_primary"], fontsize=11, loc="left")
    style_axes(ax, theme)


def plot_transfer_curve(ax, x_in, y_out, op, theme):
    # Small-signal tangent first, so the curve's departure from it reads as
    # compression at a glance.
    if op["gain"] is not None:
        xs = np.array([x_in.min(), x_in.max()])
        ax.plot(
            xs, op["quiescent"] + op["gain"] * xs,
            color=theme["ink_muted"], linewidth=1.0, linestyle=(0, (4, 3)),
            label=f"small-signal ({op['gain']:.0f} V/V)",
        )
    ax.plot(
        x_in, y_out, color=theme["series_transfer"], linewidth=1.4, alpha=0.85,
        label="measured",
    )
    ax.plot(
        [0], [op["quiescent"]], marker="o", markersize=6,
        markerfacecolor=theme["surface"], markeredgecolor=theme["series_transfer"],
        markeredgewidth=1.6, linestyle="none",
    )
    ax.set_xlabel("Input (V)")
    ax.set_ylabel("Output (V)")
    ax.set_title(
        "Transfer Curve vs Small-Signal Tangent",
        color=theme["ink_primary"], fontsize=11, loc="left",
    )
    leg = ax.legend(
        loc="upper left", fontsize=8, frameon=False, handlelength=1.6,
    )
    for text in leg.get_texts():
        text.set_color(theme["ink_secondary"])
    style_axes(ax, theme)


def plot_incremental_gain(ax, x_in, y_out, op, theme):
    x_mid, gain = incremental_gain_curve(x_in, y_out)
    ax.plot(x_mid, gain, color=theme["series_transfer"], linewidth=1.8)
    if op["gain"] is not None:
        annotate_level(ax, op["gain"], f"small-signal {op['gain']:.0f} V/V", theme)
    ax.axhline(0, color=theme["reference"], linewidth=0.8, zorder=0)
    ax.set_xlabel("Input (V)")
    ax.set_ylabel("dVout / dVin (V/V)")
    ax.set_title(
        "Incremental Gain (compression knees)",
        color=theme["ink_primary"], fontsize=11, loc="left",
    )
    style_axes(ax, theme)


def plot_spectrum(ax, freqs, psd, fundamental_power, harmonics, fmax, theme):
    psd_db = 10 * np.log10(np.maximum(psd, 1e-300) / fundamental_power)
    ax.plot(freqs, psd_db, color=theme["series_output"], linewidth=1.4)

    for k, freq, _, level_db in harmonics:
        if freq > fmax:
            continue
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

    ax.set_xlim(0, fmax)
    ax.set_ylim(-140, 10)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Level (dBc)")
    ax.set_title("Output Spectrum", color=theme["ink_primary"], fontsize=11, loc="left")
    ax.yaxis.set_major_locator(MultipleLocator(20))
    style_axes(ax, theme)


def plot_harmonic_bars(ax, harmonics, theme):
    bars = [(k, level_db) for k, _, _, level_db in harmonics if k >= 2]
    if not bars:
        ax.set_axis_off()
        return

    floor = HARMONIC_FLOOR_DB
    present = {k for k, _ in bars}
    for k, level_db in bars:
        color = theme["series_output"] if k % 2 else theme["series_input"]
        ax.bar(k, level_db - floor, bottom=floor, width=0.62, color=color)
        ax.annotate(
            f"{level_db:.0f}",
            xy=(k, level_db), xytext=(0, 4), textcoords="offset points",
            ha="center", fontsize=8, color=theme["ink_secondary"],
        )

    # Mark harmonics that fell below the detection floor instead of leaving a
    # silent gap in the sequence.
    all_ks = range(2, max(present) + 1)
    for k in all_ks:
        if k not in present:
            ax.annotate(
                f"< {floor:.0f}",
                xy=(k, floor), xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=7, color=theme["ink_muted"],
            )

    split = even_odd_split(harmonics)
    parts = [
        f"{name} RSS {level:.0f} dBc"
        for name, level in split.items() if level is not None
    ]
    if parts:
        ax.text(
            0.98, 0.94, " · ".join(parts),
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=theme["ink_secondary"],
        )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=theme["series_input"]),
        plt.Rectangle((0, 0), 1, 1, color=theme["series_output"]),
    ]
    leg = ax.legend(
        handles, ["even", "odd"], loc="upper left", fontsize=8,
        frameon=False, handlelength=1.2, handleheight=1.0,
    )
    for text in leg.get_texts():
        text.set_color(theme["ink_secondary"])

    ax.set_xticks(list(all_ks))
    ax.set_xticklabels([f"H{k}" for k in all_ks])
    ax.set_ylim(floor, 8)
    ax.set_xlabel("Harmonic")
    ax.set_ylabel("Level (dBc)")
    ax.set_title(
        "Harmonic Levels (even vs odd)",
        color=theme["ink_primary"], fontsize=11, loc="left",
    )
    style_axes(ax, theme)
    ax.grid(False, axis="x")


# -----------------------------------------------------------------------------
#                          FIGURE & SUMMARY
# -----------------------------------------------------------------------------

def build_figure(df, sample_rate, theme_name, fmax):
    theme = THEMES[theme_name]

    x_in = df["input"].to_numpy()
    y_out = df["output"].to_numpy()

    freqs, psd = compute_spectrum(y_out, sample_rate)
    fundamental_freq, fundamental_power = find_fundamental(freqs, psd)
    harmonics = find_harmonics(freqs, psd, fundamental_freq, fundamental_power, N_HARMONICS)
    op = analyze_operating_point(x_in, y_out)

    window_duration = CYCLES_TO_SHOW / fundamental_freq
    t0 = df["time"].iloc[0]
    df_wave = df[df["time"] < t0 + window_duration]
    t_ms = (df_wave["time"] - t0) * 1000.0

    fig, axes = plt.subplots(3, 2, figsize=(12, 12.5))
    fig.patch.set_facecolor(theme["surface"])

    plot_input_waveform(axes[0, 0], t_ms, df_wave["input"], theme)
    plot_output_waveform(axes[0, 1], t_ms, df_wave["output"].to_numpy(), op, theme)
    plot_transfer_curve(axes[1, 0], x_in, y_out, op, theme)
    plot_incremental_gain(axes[1, 1], x_in, y_out, op, theme)
    plot_spectrum(axes[2, 0], freqs, psd, fundamental_power, harmonics, fmax, theme)
    plot_harmonic_bars(axes[2, 1], harmonics, theme)

    fig.suptitle(
        f"Tube Preamp — K-Method Simulation  (fundamental ≈ {fundamental_freq:.0f} Hz)",
        color=theme["ink_primary"], fontsize=13, x=0.02, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    return fig, {
        "fundamental_freq": fundamental_freq,
        "harmonics": harmonics,
        "thd": estimate_thd(harmonics),
        "even_odd": even_odd_split(harmonics),
        "op": op,
    }


def print_summary(sample_rate, analysis):
    op = analysis["op"]
    print(f"Detected sample rate:  {sample_rate:.2f} Hz")
    print(f"Fundamental:           {analysis['fundamental_freq']:.1f} Hz")
    print(f"Input peak:            {op['peak_in']:.2f} V")
    print(f"Quiescent output:      {op['quiescent']:.1f} V")
    print(f"Output clip levels:    {op['out_min']:.1f} V .. {op['out_max']:.1f} V "
          f"({op['out_max'] - op['out_min']:.1f} Vpp)")
    if op["gain"] is not None:
        print(f"Small-signal gain:     {op['gain']:.1f} V/V")
    if analysis["thd"] is not None:
        print(f"THD (through H{analysis['harmonics'][-1][0]}):     {analysis['thd']:.1f} %")
    for name, level in analysis["even_odd"].items():
        if level is not None:
            print(f"{name.capitalize():>5} harmonics RSS:   {level:.1f} dBc")
    levels = ", ".join(
        f"H{k} {level:.0f}" for k, _, _, level in analysis["harmonics"][1:]
    )
    if levels:
        print(f"Harmonic levels (dBc): {levels}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="results.csv", help="Input CSV from the simulator")
    parser.add_argument("--out", default="results.png", help="Output image path")
    parser.add_argument("--theme", choices=THEMES.keys(), default="light")
    parser.add_argument("--fmax", type=float, default=5000.0,
                        help="Spectrum frequency-axis limit in Hz")
    parser.add_argument("--no-show", action="store_true", help="Skip the interactive window")
    args = parser.parse_args()

    df, sample_rate = load_results(args.csv)

    fig, analysis = build_figure(df, sample_rate, args.theme, args.fmax)
    print_summary(sample_rate, analysis)

    fig.savefig(args.out, dpi=200, facecolor=fig.get_facecolor())
    print(f"Saved {args.out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
