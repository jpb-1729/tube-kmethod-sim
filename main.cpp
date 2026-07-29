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
#include <Eigen/Dense>

// Constants
namespace Constants {
    constexpr double TRANSIENT_DURATION_SEC = 0.05;
    constexpr double MAX_TRANSIENT_RATIO = 0.1;
    constexpr double EXP_CLAMP_LIMIT = 50.0;
    constexpr double JACOBIAN_STEP_SIZE = 1e-7;
    constexpr double DAMPING_REDUCTION_FACTOR = 0.5;
    constexpr double DAMPING_GROWTH_FACTOR = 1.1;
    constexpr double TUBE_IP_EXPONENT = 1.4;
    constexpr double TUBE_IA_SCALE_FACTOR = 2.0;
    constexpr double RMS_EPSILON = 1e-9;
    constexpr size_t MAX_CONVERGENCE_WARNINGS = 10;
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
    double kg;
    double kp;
    double kvb = 300.0;
    double vct = 0.5;
    double gg = 2e-3;
    double cg = 1.0;
    double epsilon = 1.3;
    double igo = 8e-8;
    
    TubeModelParams(double mu_, double kg_, double kp_) 
        : mu(mu_), kg(kg_), kp(kp_) {}
};

const std::map<std::string, TubeModelParams> TUBE_PRESETS = {
    {"12AX7", TubeModelParams(100, 1060, 600)},
    {"12AU7", TubeModelParams(20, 1200, 800)},
    {"12AT7", TubeModelParams(60, 1100, 700)},
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
                            std::pow(ip, Constants::TUBE_IP_EXPONENT) / p_.kg);
    }

private:
    const TubeModelParams p_;
};

// =============================================================================
//                            TUBE AMPLIFIER CONFIGURATION
// =============================================================================

struct TubeAmpConfig {
    // Circuit parameters
    double ra1 = 100e3;
    double rg = 1e3;
    double rk1 = 0.820e3;
    double rk2 = 50e3;
    double ck = 25e-6;
    
    // Operating parameters
    double sample_rate = 44100.0;
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
        if (newton_tolerance <= 0) {
            throw std::invalid_argument("Newton tolerance must be positive");
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
          dt_(1.0 / cfg_.sample_rate), 
          a_(2.0 / dt_),
          convergence_warning_count_(0) {
        
        cfg_.validate();
        initialize_state_space();
    }

    std::vector<double> process(const std::vector<double>& x_in) {
        const size_t N = x_in.size();
        if (N == 0) return {};

        // Pre-allocate all working memory
        Eigen::Matrix<double, 2, Eigen::Dynamic> u(2, N);
        Eigen::Matrix<double, 1, Eigen::Dynamic> x_state(1, N);
        Eigen::Matrix<double, 4, Eigen::Dynamic> i_tube(4, N);
        Eigen::Matrix<double, 4, Eigen::Dynamic> v_nodes(4, N);
        
        // Initialize
        prepare_input(x_in, u);
        x_state.setZero();
        initialize_voltage_guess(v_nodes);
        i_tube.col(0) = compute_tube_currents(v_nodes.col(0));

        // Main processing loop
        process_samples(u, x_state, i_tube, v_nodes, N);

        // Extract and prepare output
        return extract_output(i_tube, N);
    }

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
    }

    void prepare_input(const std::vector<double>& x_in, 
                      Eigen::Matrix<double, 2, Eigen::Dynamic>& u) const {
        for(size_t i = 0; i < x_in.size(); ++i) {
            u(0, i) = x_in[i] * cfg_.gain;
            u(1, i) = cfg_.bias_voltage;
        }
    }

    void initialize_voltage_guess(Eigen::Matrix<double, 4, Eigen::Dynamic>& v_nodes) const {
        // Better initial guess based on typical operating points
        v_nodes.col(0) << -1.5, 150.0, -1.5, 150.0;
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
        auto [v_solved, converged] = newton_solve(p, v_nodes.col(n - 1));
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
    
    Mat4d compute_jacobian(const Vec4d& v) const {
        Mat4d J = Mat4d::Zero();
        const double h = Constants::JACOBIAN_STEP_SIZE;
        
        // Central difference approximation
        for (int i = 0; i < 4; ++i) {
            Vec4d v_plus = v;
            Vec4d v_minus = v;
            v_plus(i) += h;
            v_minus(i) -= h;
            J.col(i) = (compute_tube_currents(v_plus) - compute_tube_currents(v_minus)) / (2.0 * h);
        }
        return J;
    }

    std::pair<Vec4d, bool> newton_solve(const Vec4d& p, const Vec4d& v_guess) const {
        Vec4d v = v_guess;
        double damping = 1.0;
        
        for (int iter = 0; iter < cfg_.newton_max_iterations; ++iter) {
            const Vec4d i_tube = compute_tube_currents(v);
            const Vec4d residual = p - v + K_ * i_tube;
            
            // Check convergence
            if (residual.lpNorm<1>() < cfg_.newton_tolerance) {
                return {v, true};
            }

            // Compute step
            const Mat4d Jg = K_ * compute_jacobian(v) - Mat4d::Identity();
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
                                      size_t N) const {
        // Calculate output signal
        std::vector<double> y(N);
        for (size_t i = 0; i < N; ++i) {
            y[i] = -(i_tube(3, i) + i_tube(2, i) * cfg_.rk2); // Inverted
        }
        
        // Remove transient
        const size_t transient_samples = calculate_transient_samples(N);
        return std::vector<double>(y.begin() + transient_samples, y.end());
    }

    size_t calculate_transient_samples(size_t total_samples) const {
        const size_t duration_based = static_cast<size_t>(
            Constants::TRANSIENT_DURATION_SEC * cfg_.sample_rate
        );
        const size_t ratio_based = static_cast<size_t>(
            total_samples * Constants::MAX_TRANSIENT_RATIO
        );
        return std::min(duration_based, ratio_based);
    }

    // Member variables
    const TubeAmpConfig cfg_;
    const TubeModel tube1_;
    const TubeModel tube2_;
    const double dt_;
    const double a_;
    mutable size_t convergence_warning_count_;
    
    // State-space matrices
    Mat1d A_, a_eye_, H_;
    Eigen::Matrix<double, 1, 2> B_;
    Eigen::Matrix<double, 1, 4> C_;
    Eigen::Matrix<double, 4, 1> D_, DH_;
    Eigen::Matrix<double, 4, 2> E_;
    Mat4d F_, K_;
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

std::vector<double> normalize_audio(const std::vector<double>& x, double peak_level = 0.5) {
    if (x.empty()) return {};
    
    const double mean = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    double peak = 0.0;
    
    for (const double val : x) {
        peak = std::max(peak, std::abs(val - mean));
    }
    
    std::vector<double> y = x;
    if (peak > Constants::RMS_EPSILON) {
        const double scale = peak_level / peak;
        for (auto& val : y) {
            val = (val - mean) * scale;
        }
    }
    return y;
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
        cfg.gain = 1.5;
        cfg.bias_voltage = 300.0;
        
        // Use preset
        auto preset_it = TUBE_PRESETS.find("12AT7");
        if (preset_it != TUBE_PRESETS.end()) {
            cfg.tube1_params = preset_it->second;
        } else {
            throw std::runtime_error("Unknown tube preset");
        }
        
        // Generate test signal
        const auto x_in = generate_test_signal(440.0, 1.5, .1, cfg.sample_rate);
        
        // Process
        TubeAmplifier amp(cfg);
        std::cout << "Processing " << x_in.size() << " samples...\n";
        const auto y_raw = amp.process(x_in);
        
        // Prepare for output
        const size_t transient_samples = x_in.size() - y_raw.size();
        std::vector<double> x_in_trimmed(x_in.begin() + transient_samples, x_in.end());
        
        // Save results
        std::cout << "Saving results to results.csv...\n";
        save_results_to_csv(x_in_trimmed, y_raw, cfg.sample_rate);
        
        // Print statistics
        std::cout << "\n--- Simulation Complete ---\n";
        std::cout << "Input RMS:  " << calculate_rms(x_in) << "\n";
        std::cout << "Output RMS: " << calculate_rms(y_raw) << "\n";
        std::cout << "\nResults saved to results.csv\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}