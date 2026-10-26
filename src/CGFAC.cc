#include <cassert>
#include <cmath>
#include <complex>
#include <random>
#include <vector>

class CGFAC {

  double eta_;

  int npar_;

  double l2reg_;

  double dropout_p_;

  double momentum_;

  std::mt19937 rgen_;

  int niter_;

  double eta1_;

  double eta0_;

  bool etaDecay_;

  std::vector<std::complex<double>> velocity_;

public:
  CGFAC(double eta, double eta1, bool etaDecay = true, double l2reg = 0,
        double momentum = 0, double dropout_p = 0,
        unsigned int seed = std::mt19937::default_seed)
      : eta_(eta), npar_(-1), l2reg_(l2reg), dropout_p_(dropout_p),
        momentum_(momentum), rgen_(seed), niter_(0), eta1_(eta1), eta0_(eta),
        etaDecay_(etaDecay) {}

  void SetNpar(int npar) {
    assert(npar > 0);
    npar_ = npar;
    velocity_.assign(npar_, std::complex<double>(0.0, 0.0));
  }

  double GetEta() { return eta_; }

  void SetEta(double eta) {
    eta_ = eta;
    eta0_ = eta;
  }

  void SetEtaAfterRestart(double eta) { eta_ = eta; }

  int GetNiter() { return niter_; }

  void SetNiter(int niter) { niter_ = niter; }

  void SetMomentum(double momentum) { momentum_ = momentum; }

  void SetDropout(double dropout_p) { dropout_p_ = dropout_p; }

  void SetL2Reg(double l2reg) { l2reg_ = l2reg; }

  void Update(const std::vector<std::complex<double>> &direction,
              std::vector<std::complex<double>> &pars) {
    assert(npar_ > 0);
    assert(direction.size() == static_cast<std::size_t>(npar_));
    assert(pars.size() == static_cast<std::size_t>(npar_));

    if (velocity_.size() != static_cast<std::size_t>(npar_)) {
      velocity_.assign(npar_, std::complex<double>(0.0, 0.0));
    }

    double eta = eta_;
    if (etaDecay_ && eta1_ > 0.0) {
      eta = eta0_ / std::sqrt(1.0 + static_cast<double>(niter_) / eta1_);
      eta_ = eta;
    }

    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < npar_; i++) {
      if (distribution(rgen_) <= dropout_p_) {
        continue;
      }

      const std::complex<double> regularized_direction =
          direction[i] + l2reg_ * pars[i];

      velocity_[i] = momentum_ * velocity_[i] - eta * regularized_direction;
      pars[i] += velocity_[i];
    }

    niter_ += 1;
  }
};
