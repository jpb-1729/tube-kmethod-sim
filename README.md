# tube-kmethod-sim

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A C++ simulation of a two-stage vacuum tube preamplifier circuit using the
**K-method**, a discrete-time state-space technique for simulating nonlinear
analog circuits sample-by-sample. Tube behavior is modeled with the
Koren tube equations, and the resulting nonlinear system is solved at each
sample with a damped Newton-Raphson iteration.

A 440 Hz sine driven into clipping, showing the odd/even harmonic content
introduced by the tube stages:

![Example input/output waveforms and output spectrum](assets/example_output.png)

## How it works

- The linear part of the circuit (resistors, the cathode bypass cap) is
  expressed as a discrete-time state-space system derived via the bilinear
  transform.
- The two tubes contribute nonlinear grid and anode currents (Koren model)
  that couple back into the circuit's node voltages.
- At every sample, a 4x4 nonlinear system for the tube node voltages is
  solved with Newton's method (finite-difference Jacobian, adaptive damping,
  LU decomposition with a pseudo-inverse fallback for ill-conditioned steps).
- `main()` builds a 440 Hz test tone, runs it through the amplifier, and
  writes the input/output waveforms to `results.csv`.

## Requirements

- CMake >= 3.16
- A C++17 compiler (GCC or Clang tested)
- [Eigen3](https://eigen.tuxfamily.org/) (>= 3.3), discoverable via
  `find_package(Eigen3)` (e.g. `brew install eigen` on macOS)
- Python 3 with the packages in `requirements.txt`, only needed for plotting
  results

## Building and running

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
./SimpleTubeSim
```

This prints input/output RMS levels and writes `results.csv` (columns:
`time,input,output`) to the working directory.

## Plotting results

```bash
pip install -r requirements.txt
python3 plot_results.py
```

Reads `results.csv`, plots the input waveform, output waveform, and the
output's power spectral density, and saves the figure to `results.png`.

## Configuration

Amplifier parameters live in `TubeAmpConfig` in `main.cpp`: circuit values
(plate/grid/cathode resistors, cathode bypass cap), gain, bias voltage,
sample rate, and Newton solver tolerance/iteration limits. Tube presets
(`12AX7`, `12AU7`, `12AT7`) are defined in `TUBE_PRESETS` and can be assigned
to `tube1_params` / `tube2_params`.

## License

MIT — see [LICENSE](LICENSE).
