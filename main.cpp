// improved_tube_amp.cpp
//
// Performance and style-improved version of the tube amplifier simulation
//
// Key improvements:
// 1. Better const correctness and organization
// 2. Eliminated magic numbers with named constants
// 3. Optimized Newton solver with better initial guesses
// 4. Reduced memory allocations
// 5. Added basic error handling
// 6. Cleaner code structure

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <map>
#include <stdexcept>
#include <utility>
#include <chrono>
#include <Eigen/Dense>

// Constants
namespace Constants {
    // Settling is governed by the slowest pole of the linear subsystem, so the
    // trim is expressed in state time constants rather than as a fixed
    // duration. Five time constants leaves under 1% of the initial offset.
    constexpr double TRANSIENT_TIME_CONSTANTS = 5.0;
    // Hard cap on the trim so that a short input still produces some output.
    constexpr double MAX_TRANSIENT_RATIO = 0.5;
    // The DC operating point is solved once from a coarse guess, so it gets a
    // far more generous iteration budget than the per-sample solve.
    constexpr int DC_SOLVE_MAX_ITERATIONS = 200;
    constexpr double EXP_CLAMP_LIMIT = 50.0;
    constexpr double DAMPING_REDUCTION_FACTOR = 0.5;
    constexpr double DAMPING_GROWTH_FACTOR = 1.1;
    // Koren's (1 + sign(E1)) term for E1 > 0.
    constexpr double TUBE_IA_SCALE_FACTOR = 2.0;
    constexpr size_t MAX_CONVERGENCE_WARNINGS = 10;
    // Anti-aliasing/interpolation filter for oversampled runs: ~90 dB Kaiser
    // design with the transition band from 0.9x to 1.0x the base-rate
    // Nyquist, which sets the tap count per unit of oversampling factor.
    constexpr double OS_FILTER_KAISER_BETA = 8.96;
    constexpr size_t OS_FILTER_TAPS_PER_FACTOR = 115;
    constexpr double OS_FILTER_CUTOFF_BASE = 0.475;  // cycles/sample at base rate
}

// Type aliases
using Vec4d = Eigen::Vector4d;
using Mat4d = Eigen::Matrix4d;
using Vec1d = Eigen::Matrix<double, 1, 1>;
using Mat1d = Eigen::Matrix<double, 1, 1>;

// =============================================================================
//                            TUBE MODEL
// =============================================================================

struct TubeModelParams {
    double mu;
    double ex;
    double kg;
    double kp;
    double kvb;
    double vct = 0.5;
    double gg = 2e-3;
    // Sharpness of the grid-conduction knee. The model approximates
    // max(0, vgk) with a softplus whose knee is ~1/cg volts wide, so small
    // values leak large grid currents at negative bias where a real triode
    // draws only nanoamps. Sharp enough here that igo sets the cutoff-region
    // grid current, while conduction at positive vgk is left intact.
    double cg = 10.0;
    double epsilon = 1.3;
    double igo = 8e-8;

    TubeModelParams(double mu_, double ex_, double kg_, double kp_, double kvb_)
        : mu(mu_), ex(ex_), kg(kg_), kp(kp_), kvb(kvb_) {}
};

// Norman Koren's published triode fits (mu, Ex, Kg1, Kp, Kvb) from
// "Improved vacuum-tube models for SPICE simulations".
const std::map<std::string, TubeModelParams> TUBE_PRESETS = {
    {"12AX7", TubeModelParams(100.0, 1.4, 1060, 600, 300)},
    {"12AU7", TubeModelParams(21.5, 1.3, 1180, 84, 300)},
    {"12AT7", TubeModelParams(60.0, 1.35, 460, 300, 300)},
};

class TubeModel {
public:
    explicit TubeModel(const TubeModelParams& params) : p_(params) {
        if (p_.mu <= 0) {
            throw std::invalid_argument("Tube mu parameter must be positive");
        }
    }

    double grid_current(double vgk) const {
        const double exp_term = std::clamp(p_.cg * vgk, 
                                         -Constants::EXP_CLAMP_LIMIT, 
                                         Constants::EXP_CLAMP_LIMIT);
        return p_.gg * std::pow(std::log1p(std::exp(exp_term)) / p_.cg, p_.epsilon) + p_.igo;
    }

    double anode_current(double vak, double vgk) const {
        if (vak <= 0) return 0.0;
        
        const double veff = 1.0 / p_.mu + (vgk + p_.vct) / std::sqrt(p_.kvb + vak * vak);
        const double exp_term = std::clamp(p_.kp * veff, 
                                         -Constants::EXP_CLAMP_LIMIT, 
                                         Constants::EXP_CLAMP_LIMIT);
        const double ip = (vak / p_.kp) * std::log1p(std::exp(exp_term));
        
        return std::max(0.0, Constants::TUBE_IA_SCALE_FACTOR *
                            std::pow(ip, p_.ex) / p_.kg);
    }

    // d(grid_current)/d(vgk). The softplus knee differentiates to a logistic
    // sigmoid; epsilon > 1 keeps the power-term derivative finite at s -> 0.
    // Where the exp clamp saturates, the implemented softplus is constant, so
    // its exact derivative is zero (far outside the operating region).
    double grid_current_dvgk(double vgk) const {
        const double raw = p_.cg * vgk;
        if (std::abs(raw) >= Constants::EXP_CLAMP_LIMIT) return 0.0;
        const double s = std::log1p(std::exp(raw)) / p_.cg;
        const double sigmoid = 1.0 / (1.0 + std::exp(-raw));
        return p_.gg * p_.epsilon * std::pow(s, p_.epsilon - 1.0) * sigmoid;
    }

    // {d(anode_current)/d(vak), d(anode_current)/d(vgk)} by the chain rule
    // through ip = (vak/kp) * softplus(kp * veff).
    std::pair<double, double> anode_current_derivatives(double vak, double vgk) const {
        if (vak <= 0) return {0.0, 0.0};

        const double root = std::sqrt(p_.kvb + vak * vak);
        const double veff = 1.0 / p_.mu + (vgk + p_.vct) / root;
        const double raw = p_.kp * veff;
        const double t = std::clamp(raw,
                                    -Constants::EXP_CLAMP_LIMIT,
                                    Constants::EXP_CLAMP_LIMIT);
        const double sp = std::log1p(std::exp(t));
        // Where the clamp saturates, sp is constant wrt the voltages and only
        // the (vak/kp) prefactor of ip still varies.
        const double sigmoid = (std::abs(raw) < Constants::EXP_CLAMP_LIMIT)
            ? 1.0 / (1.0 + std::exp(-t)) : 0.0;
        const double ip = (vak / p_.kp) * sp;
        if (ip <= 0) return {0.0, 0.0};

        const double dia_dip = Constants::TUBE_IA_SCALE_FACTOR * p_.ex *
                               std::pow(ip, p_.ex - 1.0) / p_.kg;
        const double dveff_dvak = -(vgk + p_.vct) * vak / (root * root * root);
        const double dip_dvak = sp / p_.kp + vak * sigmoid * dveff_dvak;
        const double dip_dvgk = vak * sigmoid / root;
        return {dia_dip * dip_dvak, dia_dip * dip_dvgk};
    }

private:
    const TubeModelParams p_;
};

// =============================================================================
//                      RESAMPLING (oversampling support)
// =============================================================================
//
// The nonlinear loop runs at oversample_factor times the output rate so that
// clipping harmonics fall below the internal Nyquist instead of aliasing into
// the audio band. One linear-phase Kaiser-windowed-sinc lowpass serves as
// both the interpolation and the anti-aliasing (decimation) filter; its group
// delay is an integer number of internal samples and is compensated exactly,
// keeping the decimated output sample-aligned with the base-rate input.

namespace resample {

double bessel_i0(double x) {
    double sum = 1.0, term = 1.0;
    for (int k = 1; k < 64; ++k) {
        term *= (x * x) / (4.0 * k * k);
        sum += term;
        if (term < sum * 1e-16) break;
    }
    return sum;
}

// Odd-length, unity-DC-gain lowpass. cutoff is in cycles per sample at the
// rate the filter runs at; 0.5 is that rate's Nyquist.
std::vector<double> kaiser_lowpass(size_t taps, double cutoff, double beta) {
    const int mid = static_cast<int>(taps - 1) / 2;
    std::vector<double> h(taps);
    double sum = 0.0;
    const double norm = bessel_i0(beta);
    for (size_t n = 0; n < taps; ++n) {
        const int m = static_cast<int>(n) - mid;
        const double sinc = (m == 0)
            ? 2.0 * cutoff
            : std::sin(2.0 * M_PI * cutoff * m) / (M_PI * m);
        const double r = static_cast<double>(m) / mid;
        h[n] = sinc * bessel_i0(beta * std::sqrt(1.0 - r * r)) / norm;
        sum += h[n];
    }
    for (double& v : h) v /= sum;
    return h;
}

// Zero-stuff by factor and lowpass, exploiting the stuffed zeros (each input
// sample scatters through the kernel once). Gain factor restores the level.
std::vector<double> upsample_fir(const std::vector<double>& in, int factor,
                                 const std::vector<double>& h) {
    const int mid = static_cast<int>(h.size() - 1) / 2;
    const long n_out = static_cast<long>(in.size()) * factor;
    std::vector<double> out(n_out, 0.0);
    for (size_t j = 0; j < in.size(); ++j) {
        const double x = in[j] * factor;
        const long base = static_cast<long>(j) * factor + mid;
        for (size_t k = 0; k < h.size(); ++k) {
            const long i = base - static_cast<long>(k);
            if (i >= 0 && i < n_out) out[i] += h[k] * x;
        }
    }
    return out;
}

// Lowpass below the base-rate Nyquist and keep every factor-th sample,
// computing only the samples that survive decimation.
std::vector<double> decimate_fir(const std::vector<double>& in, int factor,
                                 const std::vector<double>& h) {
    const int mid = static_cast<int>(h.size() - 1) / 2;
    const long n_in = static_cast<long>(in.size());
    std::vector<double> out(in.size() / factor, 0.0);
    for (size_t i = 0; i < out.size(); ++i) {
        const long center = static_cast<long>(i) * factor + mid;
        double acc = 0.0;
        for (size_t k = 0; k < h.size(); ++k) {
            const long idx = center - static_cast<long>(k);
            if (idx >= 0 && idx < n_in) acc += h[k] * in[idx];
        }
        out[i] = acc;
    }
    return out;
}

}  // namespace resample

// =============================================================================
//                            TUBE AMPLIFIER CONFIGURATION
// =============================================================================

struct TubeAmpConfig {
    // Circuit parameters
    double ra1 = 100e3;
    // Grid stopper. Real preamps put tens of kilohms here (or see an
    // equivalent source impedance), which is what makes grid conduction
    // clamp the positive half-cycle; 1k would barely limit it.
    double rg = 68e3;
    double rk1 = 0.820e3;
    double rk2 = 50e3;
    double ck = 25e-6;
    
    // Operating parameters
    double sample_rate = 44100.0;
    // The nonlinear solve runs at sample_rate * oversample_factor; the output
    // is anti-alias filtered back down to sample_rate. 1 disables resampling.
    int oversample_factor = 8;
    double gain = 2.0;
    double bias_voltage = 325.0;
    
    // Tube models
    TubeModelParams tube1_params = TUBE_PRESETS.at("12AX7");
    TubeModelParams tube2_params = TUBE_PRESETS.at("12AX7");
    
    // Solver parameters
    double newton_tolerance = 1e-4;
    int newton_max_iterations = 22;
    
    void validate() const {
        if (sample_rate <= 0) {
            throw std::invalid_argument("Sample rate must be positive");
        }
        if (oversample_factor < 1) {
            throw std::invalid_argument("Oversample factor must be >= 1");
        }
        if (newton_tolerance <= 0) {
            throw std::invalid_argument("Newton tolerance must be positive");
        }
        // The state matrix is -1/(ck*rk1) and is inverted for the DC solve.
        if (ck <= 0 || rk1 <= 0) {
            throw std::invalid_argument("Cathode capacitor and resistor must be positive");
        }
    }
};

// =============================================================================
//                            TUBE AMPLIFIER
// =============================================================================

class TubeAmplifier {
public:
    explicit TubeAmplifier(const TubeAmpConfig& config)
        : cfg_(config),
          tube1_(cfg_.tube1_params),
          tube2_(cfg_.tube2_params),
          dt_(1.0 / (cfg_.sample_rate * std::max(1, cfg_.oversample_factor))),
          a_(2.0 / dt_),
          convergence_warning_count_(0),
          output_offset_(0) {

        cfg_.validate();
        if (cfg_.oversample_factor > 1) {
            size_t taps = Constants::OS_FILTER_TAPS_PER_FACTOR *
                          static_cast<size_t>(cfg_.oversample_factor) + 1;
            if (taps % 2 == 0) ++taps;
            kernel_ = resample::kaiser_lowpass(
                taps,
                Constants::OS_FILTER_CUTOFF_BASE / cfg_.oversample_factor,
                Constants::OS_FILTER_KAISER_BETA);
        }
        initialize_state_space();
        solve_dc_operating_point();
    }

    std::vector<double> process(const std::vector<double>& x_in) {
        if (x_in.empty()) return {};
        const int factor = cfg_.oversample_factor;
        const std::vector<double> x_int = (factor > 1)
            ? resample::upsample_fir(x_in, factor, kernel_)
            : x_in;
        const size_t N = x_int.size();

        // Pre-allocate all working memory
        Eigen::Matrix<double, 2, Eigen::Dynamic> u(2, N);
        Eigen::Matrix<double, 1, Eigen::Dynamic> x_state(1, N);
        Eigen::Matrix<double, 4, Eigen::Dynamic> i_tube(4, N);
        Eigen::Matrix<double, 4, Eigen::Dynamic> v_nodes(4, N);

        // Start from the quiescent operating point, so sample 0 is already a
        // self-consistent solution rather than an unsolved guess.
        prepare_input(x_int, u);
        x_state.setZero();
        v_nodes.col(0) = v_dc_;
        x_state.col(0) = x_dc_;
        i_tube.col(0) = i_dc_;

        // Main processing loop
        process_samples(u, x_state, i_tube, v_nodes, N);

        // Extract and prepare output
        return extract_output(i_tube, N);
    }

    // Index into the ORIGINAL input where the returned output starts (the
    // transient trim), so callers can align input and output sample-exactly.
    size_t output_offset() const { return output_offset_; }

private:
    void initialize_state_space() {
        A_(0,0) = -1.0 / (cfg_.ck * cfg_.rk1);
        B_.setZero();
        C_ << 1.0 / cfg_.ck, 1.0 / cfg_.ck, 0.0, 0.0;
        D_ << -1.0, -1.0, 0.0, 0.0;
        E_ << 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0;
        F_ << -cfg_.rg, 0.0, 0.0, 0.0,
              0.0, -cfg_.ra1, -cfg_.ra1, 0.0,
              0.0, -cfg_.ra1, -cfg_.ra1 - cfg_.rk2, -cfg_.rk2,
              0.0, 0.0, -cfg_.rk2, -cfg_.rk2;
        
        a_eye_(0,0) = a_;
        H_ = (a_eye_ - A_).inverse();
        
        // Pre-compute matrices for efficiency
        DH_ = D_ * H_;
        K_ = DH_ * C_ + F_;

        // Steady-state form of the same system. Setting dx/dt = 0 in
        // dx/dt = A x + B u + C i gives x = -A^-1 (B u + C i), so
        // v = D x + E u + F i becomes
        //     v = (E - D A^-1 B) u + (F - D A^-1 C) i,
        // which has the same v = p + K i shape the per-sample solver expects.
        A_inv_ = A_.inverse();
        K_dc_ = F_ - D_ * A_inv_ * C_;
    }

    // Solve for the quiescent operating point (no input signal, supply
    // applied) so the simulation starts in steady state instead of ramping
    // into it over several cathode-bypass time constants.
    void solve_dc_operating_point() {
        const Eigen::Vector2d u_dc(0.0, cfg_.bias_voltage);
        const Vec4d p_dc = (E_ - D_ * A_inv_ * B_) * u_dc;

        const auto [v_dc, converged] = newton_solve(K_dc_, p_dc, initial_voltage_guess(),
                                                    Constants::DC_SOLVE_MAX_ITERATIONS);
        if (!converged) {
            throw std::runtime_error(
                "Failed to solve for the DC operating point; check circuit and tube parameters");
        }

        v_dc_ = v_dc;
        i_dc_ = compute_tube_currents(v_dc_);
        x_dc_ = -A_inv_ * (B_ * u_dc + C_ * i_dc_);

        // Machine-readable operating point, consumed by validate_spice.py to
        // cross-check this solver against ngspice on the same circuit.
        std::cout << std::setprecision(12)
                  << "DC-OP vgk1=" << v_dc_(0) << " vak1=" << v_dc_(1)
                  << " vgk2=" << v_dc_(2) << " vak2=" << v_dc_(3)
                  << " ig1=" << i_dc_(0) << " ia1=" << i_dc_(1)
                  << " ig2=" << i_dc_(2) << " ia2=" << i_dc_(3)
                  << std::setprecision(6) << "\n";
    }

    void prepare_input(const std::vector<double>& x_in, 
                      Eigen::Matrix<double, 2, Eigen::Dynamic>& u) const {
        for(size_t i = 0; i < x_in.size(); ++i) {
            u(0, i) = x_in[i] * cfg_.gain;
            u(1, i) = cfg_.bias_voltage;
        }
    }

    Vec4d initial_voltage_guess() const {
        // Rough operating point for a small-signal triode; only used to seed
        // the one-off DC solve, which then seeds every per-sample solve.
        return Vec4d(-1.5, 150.0, -1.5, 150.0);
    }

    void process_samples(const Eigen::Matrix<double, 2, Eigen::Dynamic>& u,
                        Eigen::Matrix<double, 1, Eigen::Dynamic>& x_state,
                        Eigen::Matrix<double, 4, Eigen::Dynamic>& i_tube,
                        Eigen::Matrix<double, 4, Eigen::Dynamic>& v_nodes,
                        size_t N) {
        for (size_t n = 1; n < N; ++n) {
            process_single_sample(n, u, x_state, i_tube, v_nodes);
        }
    }

    void process_single_sample(size_t n,
                              const Eigen::Matrix<double, 2, Eigen::Dynamic>& u,
                              Eigen::Matrix<double, 1, Eigen::Dynamic>& x_state,
                              Eigen::Matrix<double, 4, Eigen::Dynamic>& i_tube,
                              Eigen::Matrix<double, 4, Eigen::Dynamic>& v_nodes) {
        const Vec1d x_prev = x_state.col(n - 1);
        const Vec4d i_prev = i_tube.col(n - 1);
        const Eigen::Vector2d u_curr = u.col(n);
        const Eigen::Vector2d u_prev = u.col(n - 1);

        // K-method state calculation
        const Vec1d state_part = (a_eye_ + A_) * x_prev + B_ * (u_curr + u_prev) + C_ * i_prev;
        const Vec4d p = DH_ * state_part + E_ * u_curr;
        
        // Solve nonlinear system with previous solution as initial guess
        auto [v_solved, converged] = newton_solve(K_, p, v_nodes.col(n - 1),
                                                  cfg_.newton_max_iterations);
        v_nodes.col(n) = v_solved;

        if (!converged) {
            handle_convergence_failure(n);
        }

        // Update currents and state
        i_tube.col(n) = compute_tube_currents(v_nodes.col(n));
        const Vec4d i_curr = i_tube.col(n);
        x_state.col(n) = H_ * ((a_eye_ + A_) * x_prev + B_ * (u_curr + u_prev) + 
                               C_ * (i_curr + i_prev));
    }

    Vec4d compute_tube_currents(const Vec4d& v) const {
        return Vec4d(
            tube1_.grid_current(v(0)),
            tube1_.anode_current(v(1), v(0)),
            tube2_.grid_current(v(2)),
            tube2_.anode_current(v(3), v(2))
        );
    }
    
    // Analytic Jacobian of the tube currents. Each tube couples only to its
    // own grid and anode voltages, so just 6 of the 16 entries are nonzero.
    Mat4d compute_jacobian(const Vec4d& v) const {
        Mat4d J = Mat4d::Zero();
        J(0, 0) = tube1_.grid_current_dvgk(v(0));
        const auto [dia1_dvak, dia1_dvgk] = tube1_.anode_current_derivatives(v(1), v(0));
        J(1, 0) = dia1_dvgk;
        J(1, 1) = dia1_dvak;
        J(2, 2) = tube2_.grid_current_dvgk(v(2));
        const auto [dia2_dvak, dia2_dvgk] = tube2_.anode_current_derivatives(v(3), v(2));
        J(3, 2) = dia2_dvgk;
        J(3, 3) = dia2_dvak;
        return J;
    }

    // Solves v = p + K i(v). K is a parameter so that the same routine serves
    // both the per-sample solve (K_) and the DC operating-point solve (K_dc_).
    std::pair<Vec4d, bool> newton_solve(const Mat4d& K, const Vec4d& p,
                                        const Vec4d& v_guess, int max_iterations) const {
        Vec4d v = v_guess;
        double damping = 1.0;

        for (int iter = 0; iter < max_iterations; ++iter) {
            const Vec4d i_tube = compute_tube_currents(v);
            const Vec4d residual = p - v + K * i_tube;
            
            // Check convergence
            if (residual.lpNorm<1>() < cfg_.newton_tolerance) {
                return {v, true};
            }

            // Compute step
            const Mat4d Jg = K * compute_jacobian(v) - Mat4d::Identity();
            Vec4d delta;
            
            // Try LU decomposition first
            Eigen::FullPivLU<Mat4d> lu(Jg);
            if (lu.isInvertible()) {
                delta = lu.solve(residual);
            } else {
                // Fallback for ill-conditioned systems
                delta = Jg.completeOrthogonalDecomposition().solve(residual);
                damping = Constants::DAMPING_REDUCTION_FACTOR;
            }
            
            // Update with damping
            v -= damping * delta;
            
            // Adaptive damping
            damping = std::min(1.0, damping * Constants::DAMPING_GROWTH_FACTOR);
        }
        
        return {v, false};
    }

    void handle_convergence_failure(size_t sample_index) {
        if (convergence_warning_count_ < Constants::MAX_CONVERGENCE_WARNINGS) {
            std::cerr << "Warning: Newton solver failed to converge at sample " 
                     << sample_index << std::endl;
            convergence_warning_count_++;
        }
    }

    std::vector<double> extract_output(const Eigen::Matrix<double, 4, Eigen::Dynamic>& i_tube,
                                       size_t n_internal) {
        // Output is the voltage the second stage's total cathode current
        // develops across rk2. Both currents must be summed before scaling by
        // the resistance: scaling only one of them adds a voltage to a current.
        std::vector<double> y_int(n_internal);
        for (size_t i = 0; i < n_internal; ++i) {
            y_int[i] = -(i_tube(3, i) + i_tube(2, i)) * cfg_.rk2; // Inverted
        }

        std::vector<double> y = (cfg_.oversample_factor > 1)
            ? resample::decimate_fir(y_int, cfg_.oversample_factor, kernel_)
            : std::move(y_int);

        // Trim the bias-shift transient from the head and the filter-edge
        // samples (whose convolution window hangs past the data) from the
        // tail; the head edge is buried inside the much longer transient.
        const size_t head = calculate_transient_samples(y.size());
        const size_t tail = filter_edge_samples();
        output_offset_ = head;
        if (head + tail >= y.size()) {
            return {};
        }
        return std::vector<double>(y.begin() + head, y.end() - tail);
    }

    // Decimated samples at each end whose filter window extends past the
    // available data (both the interpolation and the decimation filter
    // contribute half a kernel of group delay each).
    size_t filter_edge_samples() const {
        if (cfg_.oversample_factor <= 1) return 0;
        const size_t half = (kernel_.size() - 1) / 2;
        return 2 * half / static_cast<size_t>(cfg_.oversample_factor) + 1;
    }

    // Slowest time constant of the linear subsystem (here the cathode bypass
    // network), read off the state matrix rather than hardcoded.
    double state_time_constant() const {
        return 1.0 / std::abs(A_(0, 0));
    }

    // The DC initialization removes the cold-start transient, but applying a
    // signal still shifts the average cathode current, so the bypass cap needs
    // several time constants to reach its new steady state.
    size_t calculate_transient_samples(size_t total_samples) const {
        const size_t settling_based = static_cast<size_t>(
            Constants::TRANSIENT_TIME_CONSTANTS * state_time_constant() * cfg_.sample_rate
        );
        const size_t cap = static_cast<size_t>(
            total_samples * Constants::MAX_TRANSIENT_RATIO
        );

        if (settling_based > cap) {
            std::cerr << "Warning: input is only " << total_samples / cfg_.sample_rate
                      << " s, too short to settle over "
                      << Constants::TRANSIENT_TIME_CONSTANTS << " time constants ("
                      << Constants::TRANSIENT_TIME_CONSTANTS * state_time_constant()
                      << " s); the output still contains a bias-shift transient\n";
        }
        return std::min(settling_based, cap);
    }

    // Member variables
    const TubeAmpConfig cfg_;
    const TubeModel tube1_;
    const TubeModel tube2_;
    const double dt_;
    const double a_;
    size_t convergence_warning_count_;
    size_t output_offset_;
    std::vector<double> kernel_;  // shared interpolation/anti-alias filter
    
    // State-space matrices
    Mat1d A_, a_eye_, H_;
    Eigen::Matrix<double, 1, 2> B_;
    Eigen::Matrix<double, 1, 4> C_;
    Eigen::Matrix<double, 4, 1> D_, DH_;
    Eigen::Matrix<double, 4, 2> E_;
    Mat4d F_, K_;

    // DC operating point (quiescent state the simulation starts from)
    Mat1d A_inv_;
    Mat4d K_dc_;
    Vec4d v_dc_, i_dc_;
    Vec1d x_dc_;
};

// =============================================================================
//                             UTILITY FUNCTIONS
// =============================================================================

std::vector<double> generate_test_signal(double freq, double amp, double dur, double sr) {
    const size_t n_samples = static_cast<size_t>(sr * dur);
    std::vector<double> signal(n_samples);
    const double phase_step = 2.0 * M_PI * freq / sr;
    
    for (size_t i = 0; i < n_samples; ++i) {
        signal[i] = amp * std::sin(static_cast<double>(i) * phase_step);
    }
    return signal;
}

double calculate_rms(const std::vector<double>& x) {
    if (x.empty()) return 0.0;
    const double sum_sq = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
    return std::sqrt(sum_sq / x.size());
}

void save_results_to_csv(const std::vector<double>& x_in,
                        const std::vector<double>& y_out,
                        double sample_rate,
                        const std::string& filename = "results.csv") {
    if (x_in.size() != y_out.size()) {
        throw std::invalid_argument("save_results_to_csv: x_in and y_out must be the same size");
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + filename);
    }
    
    file << "time,input,output\n";
    const double time_step = 1.0 / sample_rate;
    
    for (size_t i = 0; i < y_out.size(); ++i) {
        file << std::fixed << std::setprecision(8) 
             << i * time_step << ","
             << x_in[i] << "," 
             << y_out[i] << "\n";
    }
}

// =============================================================================
//                                MAIN
// =============================================================================

int main() {
    try {
        std::cout << "Starting Improved C++ Tube Amplifier Simulation...\n";
        
        // Configure amplifier
        TubeAmpConfig cfg;
        cfg.gain = 2.0;  // enough input drive to push both stages into clipping
        cfg.bias_voltage = 300.0;
        
        // Use preset
        auto preset_it = TUBE_PRESETS.find("12AT7");
        if (preset_it != TUBE_PRESETS.end()) {
            cfg.tube1_params = preset_it->second;
        } else {
            throw std::runtime_error("Unknown tube preset");
        }
        
        // Generate test signal. The duration must comfortably exceed the
        // cathode-bypass settling time (~0.1 s) so that enough steady-state
        // output survives the transient trim.
        const auto x_in = generate_test_signal(440.0, 1.5, 0.5, cfg.sample_rate);
        
        // Process
        TubeAmplifier amp(cfg);
        std::cout << "Processing " << x_in.size() << " samples ("
                  << cfg.oversample_factor << "x oversampled)...\n";
        const auto t_start = std::chrono::steady_clock::now();
        const auto y_raw = amp.process(x_in);
        const double elapsed_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        const double audio_s = x_in.size() / cfg.sample_rate;
        std::cout << "Processed in " << elapsed_s * 1000.0 << " ms ("
                  << audio_s / elapsed_s << "x real-time)\n";

        if (y_raw.empty()) {
            throw std::runtime_error("Input too short: nothing survived the transient trim");
        }

        // Align the input with the trimmed output
        const size_t offset = amp.output_offset();
        std::vector<double> x_in_trimmed(x_in.begin() + offset,
                                         x_in.begin() + offset + y_raw.size());
        
        // Save results
        std::cout << "Saving results to results.csv...\n";
        save_results_to_csv(x_in_trimmed, y_raw, cfg.sample_rate);
        
        // Print statistics
        std::cout << "\n--- Simulation Complete ---\n";
        std::cout << "Input RMS:  " << calculate_rms(x_in_trimmed) << "\n";
        std::cout << "Output RMS: " << calculate_rms(y_raw) << "\n";
        std::cout << "\nResults saved to results.csv\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}