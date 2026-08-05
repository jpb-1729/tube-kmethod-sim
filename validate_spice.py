# validate_spice.py
#
# Cross-checks the C++ K-method simulation against ngspice solving the SAME
# circuit with the SAME device equations. The two tubes are expressed as
# behavioral current sources (B-sources) implementing exactly the Koren anode
# equation and softplus grid-current model from main.cpp, so any disagreement
# isolates the numerical method (bilinear discretization + per-sample Newton)
# rather than device-model differences.
#
# Three comparisons, each with a pass/fail tolerance:
#   1. DC operating point: the C++ steady-state Newton solve (printed by the
#      simulator as a "DC-OP" line) vs ngspice .op node voltages.
#   2. Steady-state transfer curve: binned (grid drive, output) pairs from the
#      last cycles of both transients.
#   3. Harmonic levels and THD: coherent FFT over an integer number of
#      fundamental cycles on both outputs.
#
# Usage: python3 validate_spice.py [--binary build/SimpleTubeSim] [--keep]
# Exit code 0 = all checks pass, 1 = any failure. Requires ngspice on PATH.

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# --- Circuit and stimulus parameters. Must mirror TubeAmpConfig / main() in
# --- main.cpp; the DC-OP cross-check will fail loudly if they drift apart.
RA1 = 100e3
RG = 68e3
RK1 = 0.820e3
RK2 = 50e3
CK = 25e-6
BIAS_V = 300.0
GAIN = 2.0
INPUT_AMP = 1.5       # generate_test_signal amplitude (pre-gain)
FREQ = 440.0
SAMPLE_RATE = 44100.0
EXP_CLAMP = 50.0

# (mu, ex, kg, kp, kvb) — Koren fits, same as TUBE_PRESETS in main.cpp.
TUBE1 = dict(mu=60.0, ex=1.35, kg=460.0, kp=300.0, kvb=300.0)   # 12AT7
TUBE2 = dict(mu=100.0, ex=1.4, kg=1060.0, kp=600.0, kvb=300.0)  # 12AX7
GRID = dict(vct=0.5, gg=2e-3, cg=10.0, eps=1.3, igo=8e-8)

# Analysis window: 44 cycles of 440 Hz is exactly 0.1 s = 4410 samples at
# 44.1 kHz, so a rectangular FFT window is coherent on both sides.
N_CYCLES = 44
N_HARMONICS = 10

# --- Tolerances ---
TOL_OP_VOLTS = 0.05          # abs disagreement per DC node voltage (V)
TOL_TRANSFER_MEAN = 1.0      # mean abs transfer-curve deviation (V)
TOL_TRANSFER_MAX = 3.0       # max abs transfer-curve deviation (V)
TOL_HARMONIC_DB = 1.5        # per-harmonic dBc deviation (harmonics > -60 dBc)
TOL_THD_REL = 0.10           # relative THD disagreement
HARMONIC_COMPARE_FLOOR_DB = -60.0


def anode_expr(plate, grid, cathode, p):
    """Koren anode current as an ngspice B-source expression, mirroring
    TubeModel::anode_current including the exp clamp and vak <= 0 cutoff."""
    vpk = f"v({plate},{cathode})"
    vgk = f"v({grid},{cathode})"
    veff = f"(1/{p['mu']}+({vgk}+{GRID['vct']})/sqrt({p['kvb']}+{vpk}*{vpk}))"
    e1 = f"(({vpk}/{p['kp']})*ln(1+exp(min({p['kp']}*{veff},{EXP_CLAMP}))))"
    return f"u({vpk})*2*pow(max({e1},0),{p['ex']})/{p['kg']}"


def grid_expr(grid, cathode):
    """Softplus grid current, mirroring TubeModel::grid_current."""
    vgk = f"v({grid},{cathode})"
    g = GRID
    soft = f"(ln(1+exp(min({g['cg']}*{vgk},{EXP_CLAMP})))/{g['cg']})"
    return f"{g['gg']}*pow({soft},{g['eps']})+{g['igo']}"


def build_netlist(workdir, t_stop, t_max_step):
    tran_file = workdir / "spice_tran.txt"
    op_file = workdir / "spice_op.txt"
    netlist = f"""* Two-stage triode preamp - behavioral Koren models (mirror of main.cpp)
VIN in 0 DC 0 SIN(0 {INPUT_AMP * GAIN} {FREQ})
VDD vdd 0 DC {BIAS_V}
RG in g1 {RG}
RK1 k1 0 {RK1}
CK k1 0 {CK}
RA1 vdd p1 {RA1}
* Stage 1 (12AT7): grid and anode currents
BIG1 g1 k1 I={grid_expr('g1', 'k1')}
BIP1 p1 k1 I={anode_expr('p1', 'g1', 'k1', TUBE1)}
* Stage 2 (12AX7) cathode follower, grid DC-coupled to p1, anode at B+
RK2 k2 0 {RK2}
BIG2 p1 k2 I={grid_expr('p1', 'k2')}
BIP2 vdd k2 I={anode_expr('vdd', 'p1', 'k2', TUBE2)}

.options reltol=1e-5 abstol=1e-12 vntol=1e-9
.control
set filetype=ascii
op
echo OP-RESULTS > {op_file}
print v(g1)-v(k1) v(p1)-v(k1) v(p1)-v(k2) v(vdd)-v(k2) >> {op_file}
tran {t_max_step} {t_stop} 0 {t_max_step}
wrdata {tran_file} v(k2) v(in)
quit
.endc
.end
"""
    path = workdir / "preamp.cir"
    path.write_text(netlist)
    return path, tran_file, op_file


def run_cpp(binary, workdir):
    result = subprocess.run(
        [str(Path(binary).resolve())], cwd=workdir,
        capture_output=True, text=True, check=True,
    )
    match = re.search(
        r"DC-OP vgk1=(\S+) vak1=(\S+) vgk2=(\S+) vak2=(\S+)", result.stdout
    )
    if not match:
        raise SystemExit("Error: no DC-OP line in simulator output; rebuild the binary.")
    dc_op = np.array([float(g) for g in match.groups()])
    df = pd.read_csv(workdir / "results.csv")
    return dc_op, df


def run_ngspice(workdir, netlist):
    result = subprocess.run(
        ["ngspice", "-b", str(netlist)], cwd=workdir,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(f"Error: ngspice exited with {result.returncode}")
    return result


def parse_op(op_file):
    """The .control print writes lines like 'v(g1)-v(k1) = -1.234e+00'."""
    values = re.findall(r"=\s*(-?[\d.eE+-]+)", op_file.read_text())
    if len(values) < 4:
        raise SystemExit(f"Error: could not parse 4 op voltages from {op_file}")
    return np.array([float(v) for v in values[:4]])


def parse_tran(tran_file):
    """wrdata ascii format: one row per point, columns repeat (time, value)
    per saved vector: t v(k2) t v(in)."""
    data = np.loadtxt(tran_file)
    return data[:, 0], data[:, 1], data[:, 3]


def steady_state_window(t, vk2, vin):
    """Interpolate the (variable-step) SPICE transient onto the simulator's
    uniform sample grid over the final N_CYCLES cycles."""
    duration = N_CYCLES / FREQ
    n = int(round(duration * SAMPLE_RATE))
    grid = t[-1] - duration + np.arange(n) / SAMPLE_RATE
    out = -np.interp(grid, t, vk2)     # C++ output convention: -(cathode V)
    drive = np.interp(grid, t, vin)
    return drive, out


def cpp_window(df):
    n = int(round(N_CYCLES / FREQ * SAMPLE_RATE))
    tail = df.iloc[-n:]
    return tail["input"].to_numpy() * GAIN, tail["output"].to_numpy()


def harmonic_levels(y):
    """Coherent rectangular-window FFT: dBc of harmonics 2..N plus THD (%).
    Assumes y spans exactly N_CYCLES cycles."""
    spectrum = np.abs(np.fft.rfft(y - np.mean(y)))
    fundamental = spectrum[N_CYCLES]
    ks = np.arange(2, N_HARMONICS + 1)
    mags = spectrum[ks * N_CYCLES]
    levels_db = 20 * np.log10(mags / fundamental)
    thd = 100.0 * np.sqrt(np.sum(mags**2)) / fundamental
    return dict(zip(ks.tolist(), levels_db.tolist())), thd


def binned_transfer(drive, out, edges):
    idx = np.clip(np.searchsorted(edges, drive, side="right") - 1, 0, len(edges) - 2)
    counts = np.bincount(idx, minlength=len(edges) - 1)
    sums = np.bincount(idx, weights=out, minlength=len(edges) - 1)
    means = np.full(len(edges) - 1, np.nan)
    means[counts > 0] = sums[counts > 0] / counts[counts > 0]
    return means


class Report:
    def __init__(self):
        self.failures = 0

    def check(self, label, value, tol, fmt="{:.4g}"):
        ok = value <= tol
        status = "PASS" if ok else "FAIL"
        if not ok:
            self.failures += 1
        print(f"  [{status}] {label}: {fmt.format(value)} (tol {fmt.format(tol)})")
        return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="build/SimpleTubeSim",
                        help="Path to the compiled simulator")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the working directory with netlist and raw data")
    args = parser.parse_args()

    if shutil.which("ngspice") is None:
        raise SystemExit("Error: ngspice not found on PATH.")
    if not Path(args.binary).exists():
        raise SystemExit(f"Error: {args.binary} not found; build the project first.")

    workdir = Path(tempfile.mkdtemp(prefix="tube_validate_"))
    print(f"Working directory: {workdir}")

    # Run long enough for the cathode bypass cap (5 tau ~= 0.103 s) to settle
    # before the final analysis window.
    t_stop = 0.25
    netlist, tran_file, op_file = build_netlist(workdir, t_stop, 1.0 / SAMPLE_RATE / 8)

    print("Running C++ simulator...")
    dc_cpp, df = run_cpp(args.binary, workdir)
    print("Running ngspice...")
    run_ngspice(workdir, netlist)

    report = Report()

    # --- 1. DC operating point ---
    dc_spice = parse_op(op_file)
    print("\nDC operating point (C++ vs ngspice):")
    for name, c, s in zip(("vgk1", "vak1", "vgk2", "vak2"), dc_cpp, dc_spice):
        print(f"    {name}: {c:+.4f} V vs {s:+.4f} V")
        report.check(f"{name} |diff|", abs(c - s), TOL_OP_VOLTS, "{:.4f} V")

    # --- 2. Steady-state transfer curve ---
    drive_c, out_c = cpp_window(df)
    t, vk2, vin = parse_tran(tran_file)
    drive_s, out_s = steady_state_window(t, vk2, vin)

    edges = np.linspace(-INPUT_AMP * GAIN, INPUT_AMP * GAIN, 81)
    curve_c = binned_transfer(drive_c, out_c, edges)
    curve_s = binned_transfer(drive_s, out_s, edges)
    both = ~np.isnan(curve_c) & ~np.isnan(curve_s)
    deviation = np.abs(curve_c[both] - curve_s[both])
    print("\nSteady-state transfer curve (80 bins over the drive range):")
    report.check("mean |deviation|", float(np.mean(deviation)), TOL_TRANSFER_MEAN, "{:.3f} V")
    report.check("max |deviation|", float(np.max(deviation)), TOL_TRANSFER_MAX, "{:.3f} V")

    # --- 3. Harmonics and THD ---
    harm_c, thd_c = harmonic_levels(out_c)
    harm_s, thd_s = harmonic_levels(out_s)
    print(f"\nHarmonics (C++ vs ngspice), THD {thd_c:.2f}% vs {thd_s:.2f}%:")
    for k in harm_c:
        c, s = harm_c[k], harm_s[k]
        if max(c, s) < HARMONIC_COMPARE_FLOOR_DB:
            print(f"    H{k}: {c:.1f} vs {s:.1f} dBc (below compare floor, skipped)")
            continue
        print(f"    H{k}: {c:.1f} vs {s:.1f} dBc")
        report.check(f"H{k} |diff|", abs(c - s), TOL_HARMONIC_DB, "{:.2f} dB")
    report.check("THD relative diff", abs(thd_c - thd_s) / thd_s, TOL_THD_REL, "{:.3f}")

    if args.keep:
        print(f"\nKept working directory: {workdir}")
    else:
        shutil.rmtree(workdir, ignore_errors=True)

    if report.failures:
        print(f"\n{report.failures} check(s) FAILED")
        return 1
    print("\nAll checks passed: K-method solution matches ngspice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
