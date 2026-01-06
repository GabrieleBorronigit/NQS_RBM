#include <iostream>
// #include <Eigen/Core>
// #include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <complex>
#include <random>
#include <vector>

class Sgd {

  // decay constant
  double eta_; // prima era complesso, ha senso?non credo

  int npar_;

  double l2reg_;

  double dropout_p_;

  double momentum_;

  std::mt19937 rgen_;

  double decay_factor_;

  int niter_;

  double eta1_;
  double eta0;

  bool etaDecay_;

public:
  Sgd(double eta, double eta1, bool etaDecay = true, double l2reg = 0,
      double momentum = 0, double dropout_p = 0)
      : eta_(eta), eta1_(eta1), etaDecay_(etaDecay), l2reg_(l2reg),
        dropout_p_(dropout_p), momentum_(momentum) {
    npar_ = -1;
    eta0 = eta_;
    decay_factor_ = 1;
    niter_ = 0;
  }

  void SetNpar(int npar) { npar_ = npar; }
  double GetEta() { return eta_; }
  void SetEta(double eta) { eta_ = eta; }
  double GetNiter() { return niter_; }
  void SetNiter(int niter) { niter_ = niter; }
  void Update(const std::vector<std::complex<double>> &grad,
              std::vector<std::complex<double>> &pars) {
    assert(npar_ > 0);
    if (etaDecay_ == true) {
      eta_ =
          eta0 *
          (1 / std::sqrt(
                   1 + niter_ / eta1_)); // riduzione dinamica del learning rate

      /*
 //test: riduzione esponenziale del learning rate
double warmup=200; //eta rimane costante per le prime 200 iterazioni
double tau=100000; //cost decadimento

      // Warmup iniziale a learning rate costante
     if(niter_>=warmup) {
        eta_=eta_ * std::exp(-(niter_ - warmup) / tau);
        std::cout<<"\n\n current eta: "<<eta_<<"\n\n\n"<<std::endl;
    }*/
    }

    // eta_*=decay_factor_;
    double norm = normalize(grad);
    if (norm == 0 || std::isnan(norm)) {
      std::cerr << "❌ Gradiente nullo o NaN, salto update" << std::endl;
      return;
    }

    double gradnorm = 1. / norm; // normalizzo il gradiente
    // per ora non normalizzo
    std::uniform_real_distribution<double> distribution(0, 1); // cmq non la uso

    for (int i = 0; i < npar_; i++) {
      if (distribution(rgen_) >
          dropout_p_) { // dropout: eseguo l'update dei pesi con probabilità 1-p
        pars[i] = (1. - momentum_) * pars[i] -
                  (grad[i] + l2reg_ * pars[i]) * eta_ * gradnorm;
      }
    }

    niter_ += 1;
  }

  // void SetDecayFactor(double decay_factor){
  //   assert(decay_factor<1);
  //   decay_factor_=decay_factor;
  // }

  // void Reset(){
  // }

  double normalize(const std::vector<std::complex<double>> &grad) {
    double sum = 0.0;
    for (const auto &val : grad) {
      sum += std::norm(val); // std::norm(val) è |val|^2
    }
    return std::sqrt(sum); // Poi fai la radice
  }
};
