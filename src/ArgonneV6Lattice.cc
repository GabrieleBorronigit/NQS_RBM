// ArgonneV6Lattice: variante di ArgonneV6 con il potenziale "regolarizzato
// per il reticolo" (Nuclear Lattice EFT, vedi referenze sotto) al posto di
// AV6' valutato punto per punto.
//
// Motivazione (verificata numericamente in sessione, diagonalizzazione
// esatta e Lanczos sparso): valutare AV6' punto per punto su un reticolo
// con cinetica a soli primi vicini NON ha un limite continuo definito. A
// box fisso e ben convergente (12 fm), l'energia del deutone passa da
// +1.75 MeV (dx=1.33 fm) a -15.43 MeV (dx=0.44 fm) in modo monotono e
// senza segno di appiattirsi: AV6' varia di centinaia di MeV su scale di
// ~0.3-0.5 fm, piu' fine di qualunque dx praticabile.
//
// Rimedio standard in Nuclear Lattice EFT: si "smeara" (convolve) il
// potenziale con un regolatore gaussiano di larghezza b ~ dx, cosi' che la
// sua variazione piu' fine risolvibile sia alla scala del reticolo, poi si
// ricalibra l'intensita' complessiva (un solo fattore "scale") per
// riprodurre l'energia di legame fisica del deutone (-2.2246 MeV) SU QUEL
// reticolo specifico -- esattamente come si rifittano i coefficienti a
// basso-energia (LEC) ad ogni passo reticolare.
//
// Referenze:
//  - G.P. Lepage, "How to Renormalize the Schrodinger Equation",
//    lezioni all'8th Jorge Andre Swieca Summer School (1997),
//    arXiv:nucl-th/9706029.
//  - B. Borasoy, E. Epelbaum, H. Krebs, D. Lee, U.-G. Meissner,
//    "Lattice simulations for light nuclei: Chiral effective field theory
//    at leading order", Eur. Phys. J. A 31, 105 (2007),
//    arXiv:nucl-th/0611087.
//  - N. Klein, D. Lee, W. Liu, U.-G. Meissner, "Regularization Methods for
//    Nuclear Lattice Effective Field Theory", Phys. Lett. B 747 (2015) 511,
//    arXiv:1505.07000.
//  - D. Lee, "Lattice simulations for few- and many-body systems" (review),
//    Prog. Part. Nucl. Phys. 63 (2009) 117-154.
//
// La calibrazione (FitScale) usa una diagonalizzazione sparsa (Lanczos)
// della stessa Hamiltoniana di FindConn, eseguita UNA volta nel costruttore.

#pragma once

#include "ArgonneV6.cc"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <complex>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>

class ArgonneV6Lattice : public ArgonneV6 {
  double smear_b_;
  double scale_;

public:
  ArgonneV6Lattice(int nspins, int Mag, double traplength = 4.,
                    double alpha_ne = -2.0, double hbar = 197.327053,
                    double mass = (938.27231 + 939.56563) / 2,
                    double smear_b_factor = 1.0,
                    double E_target = -2.2245758106075337)
      : ArgonneV6(nspins, Mag, traplength, alpha_ne, hbar, mass) {
    int l = (int)std::lround(std::cbrt((double)nspins));
    double dx = traplength / l;
    smear_b_ = smear_b_factor * dx;
    scale_ = 1.0;

    if (IsMaster())
      std::cout << "# ArgonneV6Lattice: calibrazione (b=" << smear_b_
                << " fm, l=" << l << ", traplength=" << traplength
                << " fm)..." << std::endl;
    scale_ = FitScale(nspins, E_target);
    if (IsMaster()) {
      double echeck = GroundEnergy(nspins);
      std::cout << "# ArgonneV6Lattice: scale_fit=" << scale_
                << "  E_check=" << echeck
                << " MeV (target " << E_target << " MeV)" << std::endl;
    }
  }

  // Override: AV6' smussato (gaussiana di larghezza smear_b_) e riscalato
  // (scale_, calibrato nel costruttore). FindConn (ereditato, non
  // sovrascritto) chiama AV6operators su *this: grazie a "virtual" nella
  // classe base, questo override viene usato automaticamente.
  virtual const std::vector<double> AV6operators(double r) override {
    return Smear(r);
  }

private:
  std::vector<double> Smear(double r0) {
    if (smear_b_ <= 0.0) {
      std::vector<double> v = ArgonneV6::AV6operators(r0);
      for (double &x : v) x *= scale_;
      return v;
    }
    const int n = 41;
    double lo = std::max(1.0e-4, r0 - 4.0 * smear_b_);
    double hi = r0 + 4.0 * smear_b_;
    double dr = (hi - lo) / (n - 1);
    std::vector<double> acc(6, 0.0);
    double wsum = 0.0;
    for (int i = 0; i < n; i++) {
      double rr = lo + i * dr;
      double d = rr - r0;
      double w = std::exp(-(d * d) / (2.0 * smear_b_ * smear_b_));
      std::vector<double> v = ArgonneV6::AV6operators(rr);
      for (int k = 0; k < 6; k++)
        acc[k] += w * v[k];
      wsum += w;
    }
    for (int k = 0; k < 6; k++)
      acc[k] = scale_ * acc[k] / wsum;
    return acc;
  }

  // ---------------- calibrazione: Lanczos sparso su FindConn ----------------

  Eigen::SparseMatrix<std::complex<double>> BuildSparseH(int nv) {
    auto idxOf = [&](int xp, int sp, int sn) { return (xp * 2 + sp) * 2 + sn; };
    std::vector<Eigen::Triplet<std::complex<double>>> trips;
    trips.reserve((size_t)4 * nv * 15);
    std::vector<std::vector<int>> flipsh;
    std::vector<std::complex<double>> mel;

    for (int xp = 0; xp < nv; ++xp) {
      const int xn = nv - 1 - xp;
      for (int sp = 0; sp < 2; ++sp) {
        for (int sn = 0; sn < 2; ++sn) {
          const int p = xp + sp * nv;
          const int n = xn + (2 + sn) * nv;
          std::vector<int> state(4 * nv, -1);
          state[p] = 1;
          state[n] = 1;
          std::vector<int> occ = {p, n};
          flipsh.clear();
          mel.clear();
          this->FindConn(state, occ, flipsh, mel, 1, nv);
          const int row = idxOf(xp, sp, sn);
          for (size_t k = 0; k < flipsh.size(); ++k) {
            if (k > 0 && flipsh[k].empty())
              continue;
            std::vector<int> occ2 = {p, n};
            for (int s : flipsh[k]) {
              auto it = std::find(occ2.begin(), occ2.end(), s);
              if (it != occ2.end())
                occ2.erase(it);
              else
                occ2.push_back(s);
            }
            int ps = -1, ns = -1;
            for (int s : occ2) {
              if (s < 2 * nv) ps = s; else ns = s;
            }
            const int xp2 = ps % nv, sp2 = ps / nv;
            const int nloc = ns - 2 * nv;
            const int sn2 = nloc / nv;
            trips.emplace_back(row, idxOf(xp2, sp2, sn2), mel[k]);
          }
        }
      }
    }
    Eigen::SparseMatrix<std::complex<double>> H(4 * nv, 4 * nv);
    H.setFromTriplets(trips.begin(), trips.end());
    return H;
  }

  double LowestEigenvalue(const Eigen::SparseMatrix<std::complex<double>> &H,
                           int D, int m = 150) {
    using Cplx = std::complex<double>;
    using VecC = Eigen::VectorXcd;
    if (D <= 50) {
      Eigen::MatrixXcd Hd = Eigen::MatrixXcd(H);
      return Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd>(
                 Hd, Eigen::EigenvaluesOnly)
          .eigenvalues()(0);
    }
    m = std::min(m, D);
    std::mt19937_64 rng(12345);
    std::normal_distribution<double> nd(0.0, 1.0);
    VecC v = VecC::Zero(D);
    for (int i = 0; i < D; i++) v(i) = Cplx(nd(rng), nd(rng));
    v.normalize();

    std::vector<VecC> V;
    V.push_back(v);
    Eigen::VectorXd alpha(m), beta(m);
    VecC vprev = VecC::Zero(D);
    double betaPrev = 0.0;

    for (int j = 0; j < m; j++) {
      VecC w = H * V[j];
      w -= betaPrev * vprev;
      Cplx a = V[j].dot(w);
      alpha(j) = a.real();
      w -= alpha(j) * V[j];
      for (int k = 0; k <= j; k++) {
        Cplx c = V[k].dot(w);
        w -= c * V[k];
      }
      double b = w.norm();
      beta(j) = b;
      if (j + 1 < m) {
        if (b < 1e-12) { m = j + 1; break; }
        VecC vnext = w / b;
        vprev = V[j];
        betaPrev = b;
        V.push_back(vnext);
      }
    }
    int mm = (int)V.size();
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(mm, mm);
    for (int i = 0; i < mm; i++) T(i, i) = alpha(i);
    for (int i = 0; i + 1 < mm; i++) { T(i, i + 1) = beta(i); T(i + 1, i) = beta(i); }
    Eigen::VectorXd ev =
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(T, Eigen::EigenvaluesOnly)
            .eigenvalues();
    return ev(0);
  }

  double GroundEnergy(int nv) {
    auto H = BuildSparseH(nv);
    return LowestEigenvalue(H, 4 * nv);
  }

  double FitScale(int nv, double E_target) {
    auto energyAt = [&](double s) {
      scale_ = s;
      return GroundEnergy(nv);
    };
    double lo = 1e-3, hi = 80.0;
    double flo = energyAt(lo) - E_target;
    double fhi = energyAt(hi) - E_target;
    int tries = 0;
    while (flo <= 0 && tries < 12) { lo /= 2.0; flo = energyAt(lo) - E_target; tries++; }
    tries = 0;
    while (fhi >= 0 && tries < 12) { hi *= 2.0; fhi = energyAt(hi) - E_target; tries++; }
    if (flo <= 0 || fhi >= 0) {
      std::cerr << "# ArgonneV6Lattice::FitScale: bracket fallito ("
                << "f(" << lo << ")=" << flo << ", f(" << hi << ")=" << fhi
                << ")" << std::endl;
      return 1.0;
    }
    for (int it = 0; it < 60; it++) {
      double mid = 0.5 * (lo + hi);
      double fm = energyAt(mid) - E_target;
      if (std::fabs(fm) < 1e-6) return mid;
      if ((fm > 0) == (flo > 0)) { lo = mid; flo = fm; } else { hi = mid; fhi = fm; }
    }
    return 0.5 * (lo + hi);
  }
};
