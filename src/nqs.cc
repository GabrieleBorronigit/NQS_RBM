/*
############################ COPYRIGHT NOTICE ##################################

Code provided by G. Carleo and M. Troyer, written by G. Carleo, December 2016.

Permission is granted for anyone to copy, use, modify, or distribute the
accompanying programs and documents for any purpose, provided this copyright
notice is retained and prominently displayed, along with a complete citation of
the published version of the paper:
 ______________________________________________________________________________
| G. Carleo, and M. Troyer                                                     |
| Solving the quantum many-body problem with artificial neural-networks        |
|______________________________________________________________________________|

The programs and documents are distributed without any warranty, express or
implied.

These programs were written for research purposes only, and are meant to
demonstrate and reproduce the main results obtained in the paper.

All use of these programs is entirely at the user's own risk.

################################################################################
*/

#include "mpi.h"
#include <cassert>
#include <cmath>
#include <complex>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

class Nqs {

  // Neural-network weights
  std::vector<std::vector<std::complex<double>>> W_;

  // Neural-network visible bias
  std::vector<std::complex<double>> a_;
  std::vector<std::complex<double>> acopy_;

  // Neural-network hidden bias
  std::vector<std::complex<double>> b_;

  // Number of hidden units
  int nh_;

  // Number of visible units
  int nv_;
  int nvtot_;    // nv*ncanali (2 senza isospin, 4 con)
  bool isospin_; // flag per usare o meno l isospin

  // Total number of parameters
  int npar_;

  // per la generazione dei parametri iniziali
  double sigma_;

  // look-up tables
  std::vector<std::complex<double>> Lt_;

  // Useful quantities for safe computation of ln(cosh(x))
  const double log2_;

  // learning rate e iterazioni dopo il restart, da passare a variational
  double etaRestart_;
  int niterRestart_;

public:
  // costruttore del main vecchio
  Nqs(std::string filename, int loadtype = 0, double eta = 0, int niter = 0,
      bool isospin = false)
      : log2_(std::log(2.)), etaRestart_(eta), niterRestart_(niter),
        isospin_(isospin) {

    // loadtype sceglie il tipo di funzione da usare per generare i parametri,
    // di default tutti random
    if (loadtype == 0)
      LoadRandomParameters(filename);
    else if (loadtype ==
             1) // carica tutti i nodi visibili e genera i nascosti random
      LoadNvParametersFromFile(filename);
    else if (loadtype == 2) // carica tutti i nodi nascosti e visibili
      LoadParametersFromFile(filename);
    else if (loadtype == 3) // carica i parametri e aggiunge nodi nascosti
                            // random
      LoadParametersFromFileAddNh(filename);

    npar_ = (nvtot_) + nh_ + (nvtot_)*nh_;
  }

  // costruttore nuovo
  Nqs(int nv, int nh, double sigma, int loadtype, bool isospin,
      std::string restart = "none", int nhadd = 0)
      : log2_(std::log(2.)) { // stesso ma con gli input da file
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    nv_ = nv;
    nh_ = nh;
    sigma_ = sigma;
    isospin_ = isospin;
    b_.resize(nh_);
    // aggiungo i canali per l isospin, metto la flag per poter utilizzare il
    // codice sia per atomi che per nuclei. metto una varabile extra per fare un
    // if solo a monte di tutto il codice, altrimenti ogni volta dovriei
    // controllare la flag
    nvtot_ = 2 * nv;
    if (isospin_ == true)
      nvtot_ = 4 * nv_;

    a_.resize(nvtot_);
    W_.resize(nvtot_, std::vector<std::complex<double>>(nh_));
    npar_ = (nvtot_) + nh_ + (nvtot_)*nh_;

    // tutti random
    if (loadtype == 0) {
      std::mt19937 gen_;
      gen_.seed(std::time(nullptr));
      std::normal_distribution<double> distribution(0, sigma_);

      for (int i = 0; i < nvtot_; i++) {
        a_[i] = std::complex<double>(distribution(gen_), distribution(gen_));
      }

      for (int j = 0; j < nh_; j++) {
        b_[j] = std::complex<double>(distribution(gen_), distribution(gen_));
      }

      for (int i = 0; i < nvtot_; i++) {
        for (int j = 0; j < nh_; j++) {
          W_[i][j] =
              std::complex<double>(distribution(gen_), distribution(gen_));
        }
      }

      if (rank == 0) {
        std::cout << "# NQS w/ Weights loaded with random values " << std::endl;
        if (isospin_ == false)
          std::cout << "# N_visible = " << nv_ << "(x2)  N_hidden = " << nh_
                    << std::endl;
        else if (isospin_ == true)
          std::cout << "# N_visible = " << nv_ << "(x4)  N_hidden = " << nh_
                    << std::endl;
      }

    } // tutti random

    else if (loadtype ==
             1) // carica tutti i nodi visibili e genera i nascosti random
      LoadNvParametersFromFile(restart);
    else if (loadtype == 2) // carica tutti i nodi nascosti e visibili
      LoadParametersFromFile(restart);
    else if (loadtype == 3) // carica i parametri e aggiunge nodi nascosti
                            // random
      LoadParametersFromFileAddNh(restart);
  }
  // computes the logarithm of the wave-function; non ci sono i ln2
  inline std::complex<double> LogVal(const std::vector<int> &state) const {

    std::complex<double> rbm(0., 0.);

    for (int v = 0; v < nvtot_; v++) {
      rbm += a_[v] * double(state[v]);
    }

    for (int h = 0; h < nh_; h++) {

      std::complex<double> thetah = b_[h];

      for (int v = 0; v < nvtot_; v++) {
        thetah += double(state[v]) * (W_[v][h]);
      }

      rbm += Nqs::lncosh(thetah);
    }

    return rbm;
  }

  // computes the logarithm of Psi(state')/Psi(state)
  // where state' is a state with a certain number of flipped spins
  // the vector "flips" contains the sites to be flipped
  // look-up tables are used to speed-up the calculation
  inline std::complex<double> LogPoP(const std::vector<int> &state,
                                     const std::vector<int> &flips) const {

    if (flips.size() == 0) {
      return 0.;
    }
    for (const auto &flip : flips) {
      if (flip < 0 || flip >= state.size()) {
        std::cerr << "❌ Invalid flip index in PoP: " << flip
                  << " (state size = " << state.size() << ")" << std::endl;
      }
      if (flip < 0 || flip >= a_.size()) {
        std::cerr << "❌ Invalid a_ index in PoP: " << flip
                  << " (a_.size() = " << a_.size() << ")" << std::endl;
      }
      if (flip < 0 || flip >= W_.size()) {
        std::cerr << "❌ Invalid W_ index in PoP: " << flip
                  << " (W_.size() = " << W_.size() << ")" << std::endl;
      }
    }

    std::complex<double> logpop(0., 0.);

    // Change due to the visible bias
    /* std::cout<<"flips={";
     for(const auto &flip : flips){
     std::cout<<flip<<"\t";
     }*/
    // std::cout<<"}"<<std::endl;
    for (const auto &flip : flips) {
      // std::cout<<"a[flip]="<<a_[flip]<<"\t
      // state[flip]="<<state[flip]<<std::endl;
      logpop -= a_[flip] * 2. * double(state[flip]);
    }

    // Change due to the interaction weights
    for (int h = 0; h < nh_; h++) {
      const std::complex<double> thetah = Lt_[h];
      std::complex<double> thetahp = thetah;

      for (const auto &flip : flips) {
        thetahp -= 2. * double(state[flip]) * (W_[flip][h]);
      }
      logpop += (Nqs::lncosh(thetahp) - Nqs::lncosh(thetah));
    }
    if (std::isnan(logpop.real()) || std::isnan(logpop.imag())) {
      std::cerr << "Warning! NaN in logpop!!!!!!" << std::endl;
      exit(1);
    }

    return logpop;
  }

  inline std::complex<double> PoP(const std::vector<int> &state,
                                  const std::vector<int> &flips) const {

    return std::exp(LogPoP(state, flips));
  }

  // initialization of the look-up tables
  void InitLt(const std::vector<int> &state) {
    Lt_.resize(nh_);

    for (int h = 0; h < nh_; h++) {
      Lt_[h] = b_[h];
      for (int v = 0; v < nvtot_; v++) {
        Lt_[h] += double(state[v]) * (W_[v][h]);
      }
    }
  }

  // updates the look-up tables after spin flips
  // the vector "flips" contains the indices of sites to be flipped
  void UpdateLt(const std::vector<int> &state, const std::vector<int> &flips) {
    if (flips.size() == 0) {
      return;
    }

    for (int h = 0; h < nh_; h++) {
      for (const auto &flip : flips) { // looks up elements in vector flips
        Lt_[h] -= 2. * double(state[flip]) * W_[flip][h];
      }
    }
  }

  // loads the parameters of the wave-function from a given file -> now it
  // generates initial parameters too
  void LoadRandomParameters(std::string filename) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::ifstream fin(filename.c_str());

    if (!fin.good()) {
      if (rank == 0) {
        std::cerr << "# Error : Cannot load from file " << filename
                  << " : file not found." << std::endl;
        std::abort();
      }
    }

    fin >> nv_;
    fin >> nh_;
    fin >> sigma_;
    if (isospin_ == false)
      nvtot_ = 2 * nv_;
    else if (isospin_ == true)
      nvtot_ = 4 * nv_;
    if (!fin.good() || nv_ < 0 || nh_ < 0) {
      if (rank == 0) {
        std::cerr << "# Trying to load from an invalid file.";
        std::cerr << std::endl;
        std::abort();
      }
    }

    if (!fin.good()) {
      if (rank == 0) {
        std::cerr << "# Trying to load from an invalid file.";
        std::cerr << std::endl;
        std::abort();
      }
    }

    a_.resize(nvtot_);
    acopy_.resize(nvtot_);
    b_.resize(nh_);
    W_.resize(nvtot_, std::vector<std::complex<double>>(nh_));

    std::mt19937 gen_;
    gen_.seed(std::time(nullptr));
    std::normal_distribution<double> distribution(0, sigma_);

    for (int i = 0; i < nvtot_; i++) {
      a_[i] = std::complex<double>(distribution(gen_), distribution(gen_));
    }
    // a_=acopy_;
    // a_.insert(a_.end(), acopy_.begin(), acopy_.end());

    for (int j = 0; j < nh_; j++) {
      b_[j] = std::complex<double>(distribution(gen_), distribution(gen_));
    }

    for (int i = 0; i < nvtot_; i++) {
      for (int j = 0; j < nh_; j++) {
        W_[i][j] = std::complex<double>(distribution(gen_), distribution(gen_));
      }
    }
    if (rank == 0) {
      std::cout << "# NQS w/ Spin loaded with random values " << filename
                << std::endl;
      std::cout << "# N_visible = " << nv_ << "(x2)  N_hidden = " << nh_
                << std::endl;
    }
  }
  /*

  void LoadParametersFromFile2(std::string filename) {
      std::ifstream fin(filename.c_str());

      if (!fin.good()) {
          std::cerr << "# Error : Cannot load from file " << filename << " :
  file not found." << std::endl; std::abort();
      }

      fin >> nv_;
      fin >> nh_;

      if (!fin.good() || nv_ < 0 || nh_ < 0) {
          std::cerr << "# Trying to load from an invalid file." << std::endl;
          std::abort();
      }

      // Ridimensiona 'a_' per 2 * nv_ e per 'b_' e 'W_'.
      a_.resize(2 * nv_);
      acopy_.resize(nv_);
      b_.resize(nh_);
      W_.resize(2 * nv_, std::vector<std::complex<double>>(nh_));

      // Legge direttamente 2 * nv_ valori per 'a_'.
      for (int i = 0; i < 2 * nv_; i++) {
          fin >> a_[i];
      }

      // Imposta 'b_' e 'W_' a zero **sempre**.
      std::fill(b_.begin(), b_.end(), 0.0001);  // Imposta tutti gli elementi di
  'b_' a zero.

      for (int i = 0; i < 2 * nv_; i++) {
          for (int j = 0; j < nh_; j++) {
              W_[i][j] = {0.0001, 0.0001};  // Imposta ogni elemento di 'W_' a
  zero.
          }
      }

      if (!fin.good()) {
          std::cerr << "# Trying to load from an invalid file." << std::endl;
          std::abort();
      }

      std::cout << "# NQS w/ Spin loaded from file " << filename << std::endl;
      std::cout << "# N_visible = " << nv_ << "(x2)  N_hidden = " << nh_ <<
  std::endl;
  }*/

  void LoadNvParametersFromFile(std::string filename) {
    std::ifstream fin(filename.c_str());

    if (!fin.good()) {
      std::cerr << "# Error : Cannot load from file " << filename
                << " : file not found." << std::endl;
      std::abort();
    }

    fin >> nv_;
    fin >> nh_;
    if (!fin.good() || nv_ < 0 || nh_ < 0) {
      std::cerr << "# Trying to load from an invalid file." << std::endl;
      std::abort();
    }
    if (isospin_ == false)
      nvtot_ = 2 * nv_;
    else if (isospin_ == true)
      nvtot_ = 4 * nv_;
    // Ridimensiona 'a_' per 2 * nv_ e 'b_', 'W_'.
    b_.resize(nh_);
    a_.resize(nvtot_);
    W_.resize(nvtot_, std::vector<std::complex<double>>(nh_));

    // Inizializzazione del generatore casuale
    std::mt19937 gen(std::time(nullptr));
    std::normal_distribution<double> distribution(0.0,
                                                  0.0001); // μ = 0, σ = 0.0001

    // Inizializza 'a_' con valori letti dal file

    for (int i = 0; i < nvtot_; i++) {
      fin >> a_[i];
    }
    /*for(int i=0;i<nv_;i++){
     fin>>acopy_[i];
   }
   a_=acopy_;
   a_.insert(a_.end(), acopy_.begin(), acopy_.end());*/

    // Inizializza 'b_' con valori casuali gaussiani
    for (int j = 0; j < nh_; j++) {
      b_[j] = std::complex<double>(distribution(gen), distribution(gen));
    }

    // Inizializza 'W_' con valori casuali gaussiani
    for (int i = 0; i < nvtot_; i++) {
      for (int j = 0; j < nh_; j++) {
        W_[i][j] = std::complex<double>(distribution(gen), distribution(gen));
      }
    }

    if (!fin.good()) {
      std::cerr << "# Trying to load from an invalid file." << std::endl;
      std::abort();
    }

    std::cout << "# NQS w/ Spin loaded from file " << filename << std::endl;
    std::cout << "# N_visible = " << nv_
              << "(x2/4 with isospin)  N_hidden = " << nh_ << std::endl;
  }

  // loads the parameters of the wave-function from a given file
  void LoadParametersFromFile(std::string filename) {

    std::ifstream fin(filename.c_str());

    if (!fin.good()) {
      std::cerr << "# Error : Cannot load from file " << filename
                << " : file not found." << std::endl;
      std::abort();
    }

    fin >> nv_;
    fin >> nh_;
    fin >> etaRestart_;
    fin >> niterRestart_;
    if (isospin_ == false)
      nvtot_ = 2 * nv_;
    else if (isospin_ == true)
      nvtot_ = 4 * nv_;

    if (!fin.good() || nv_ < 0 || nh_ < 0) {
      std::cerr << "# Trying to load from an invalid file.";
      std::cerr << std::endl;
      std::abort();
    }

    a_.resize(nvtot_);
    b_.resize(nh_);
    W_.resize(nvtot_, std::vector<std::complex<double>>(nh_));

    for (int i = 0; i < nvtot_; i++) {
      fin >> a_[i];
    }

    for (int j = 0; j < nh_; j++) {
      fin >> b_[j];
    }

    for (int i = 0; i < nvtot_; i++) {
      for (int j = 0; j < nh_; j++) {
        fin >> W_[i][j];
      }
    }

    if (!fin.good()) {
      std::cerr << "# Trying to load from an invalid file.";
      std::cerr << std::endl;
      std::abort();
    }

    std::cout << "# NQS w/ Spin loaded from file " << filename << std::endl;
    std::cout << "# N_visible = " << nv_ << "(x2)  N_hidden = " << nh_
              << std::endl;
  }

  double GetEtaAfterRestart() { return etaRestart_; }
  int GetNiterAfterRestart() { return niterRestart_; }

  void LoadParametersFromFileAddNh(
      std::string filename) { // reads parameters from file, and adds nh random
                              // initialized hidden units
    // il file deve avere nelle prime due righe nv ed nh della run precedente,
    // come terza riga il numero di nh da aggiungere e come quarta poi lerning
    // rate ed iterazioni

    std::ifstream fin(filename.c_str());

    if (!fin.good()) {
      std::cerr << "# Error : Cannot load from file " << filename
                << " : file not found." << std::endl;
      std::abort();
    }
    int nh = 0, nhadd = 0;
    fin >> nv_;
    fin >> nh;
    fin >> nhadd;
    fin >> sigma_;
    nh_ = nh + nhadd;
    if (isospin_ == false)
      nvtot_ = 2 * nv_;
    else if (isospin_ == true)
      nvtot_ = 4 * nv_;
    // Inizializzazione del generatore casuale
    std::mt19937 gen(std::time(nullptr));
    std::normal_distribution<double> distribution(0.0, sigma_);

    if (!fin.good() || nv_ < 0 || nh_ < 0 || nhadd <= 0) {
      std::cerr << "# Trying to load from an invalid file.";
      std::cerr << std::endl;
      std::abort();
    }

    a_.resize(nvtot_);
    b_.resize(nh_);
    W_.resize(nvtot_, std::vector<std::complex<double>>(nh_));

    for (int i = 0; i < nvtot_; i++) {
      fin >> a_[i];
    }

    // aggiunge i pesi gia noti
    for (int j = 0; j < nh; j++) {
      fin >> b_[j];
    }
    // completa con pesi random
    for (int j = nh; j < nh_; j++) {
      b_[j] = std::complex<double>(distribution(gen), distribution(gen));
    }
    // same
    for (int i = 0; i < nvtot_; i++) {
      for (int j = 0; j < nh; j++) {
        fin >> W_[i][j];
      }
      for (int j = nh; j < nh_; j++) {
        W_[i][j] = std::complex<double>(distribution(gen), distribution(gen));
      }
    }

    if (!fin.good()) {
      std::cerr << "# Trying to load from an invalid file.";
      std::cerr << std::endl;
      std::abort();
    }

    std::cout << "# NQS w/ Spin loaded from file " << filename << std::endl;
    std::cout << "# N_visible = " << nv_ << "(x2)  N_hidden = " << nh << "+"
              << nhadd << " w/random weights" << std::endl;
  }

  // used for variational MC:
  std::vector<std::complex<double>> GetParameters() {
    // carica in pars i parametri nel seguente ordine:
    //  tutti gli a_-->tutti i b_--> tutti gli elementi di matrice Wij,
    //  DEVE SUCCEDERE OGNI VOLTA CHE PASSO I PARAMETRI!!!
    std::vector<std::complex<double>> pars;
    pars.resize(npar_);
    int k = 0;
    for (; k < nvtot_; k++) {
      pars[k] = a_[k];
    }

    for (int p = 0; p < nh_; p++) {
      pars[k] = b_[p];
      k++;
    }

    for (int i = 0; i < nvtot_; i++) {
      for (int j = 0; j < nh_; j++) {
        pars[k] = W_[i][j];
        k++;
      }
    }

    return pars;
  }

  void SetParameters(const std::vector<std::complex<double>> &pars) {
    // a-->b-->Wij ok!
    int k = 0;
    for (; k < nvtot_; k++) {
      a_[k] = pars[k];
    }
    for (int p = 0; p < nh_; p++) {
      b_[p] = pars[k];
      k++;
    }
    for (int i = 0; i < nvtot_; i++) {
      for (int j = 0; j < nh_; j++) {
        W_[i][j] = pars[k];
        k++;
      }
    }
  }

  std::vector<std::complex<double>> DerLog(const std::vector<int> &state) {
    // a-->b-->Wij ok!

    std::vector<std::complex<double>> der;
    der.resize(npar_);

    int k = 0;
    for (; k < nvtot_; k++) { // non so se l avevo toccato io o meno
      der[k] = state[k];      // d (log(psi))/da_i
    }

    std::vector<std::complex<double>> tanhThetas; // d(log(psi))/d(b_j)

    tanhThetas.resize(nh_);
    tanh(Lt_, tanhThetas);
    for (int p = 0; p < nh_; p++) {
      der[k] = tanhThetas[p];
      k++;
    }

    for (int i = 0; i < nvtot_;
         i++) { // d(log(psi))/d(W_ij) sarebbe una matrice ma la "appiattisce"
      for (int j = 0; j < nh_; j++) {
        der[k] = tanhThetas[j] * double(state[i]); // tanh(t_j)S_i
        k++; // riempie il vettore mettendo in fila i vettori riga delle derlogs
             // di Wij //[tanh(t_0)S0, t_1S0....t0S1, t1S1,t2S1,....t0Sn,..tnsn]
      }
    }
    return der;
  }

  // tanh for complex vectors
  void tanh(const std::vector<std::complex<double>> &x,
            std::vector<std::complex<double>> &y) {
    for (int i = 0; i < x.size(); i++) {
      y[i] = std::tanh(x[i]);
    }
  }

  // ln(cos(x)) for real argument
  // for large values of x we use the asymptotic expansion
  inline double lncosh(double x) const {
    const double xp = std::abs(x);
    if (xp <= 20.) {
      return std::log(std::cosh(xp));
    } else {
      return xp - log2_;
    }
  }

  // ln(cos(x)) for complex argument
  // the modulus is computed by means of the previously defined function
  // for real argument
  inline std::complex<double> lncosh(std::complex<double> x) const {
    const double xr = x.real();
    const double xi = x.imag();

    std::complex<double> res = Nqs::lncosh(xr);
    res += std::log(
        std::complex<double>(std::cos(xi), std::tanh(xr) * std::sin(xi)));

    return res;
  }

  /*
  inline std::complex<double> lncosh(std::complex<double> x) const {
  double xr = std::real(x);
  double xi = std::imag(x);

  if (std::abs(xr) > 20.0) {
    // Comportamento asintotico per evitare overflow o errori numerici
    return std::complex<double>(std::abs(xr) - log2_, 0.0);
  } else {
    return std::log(std::cosh(x));
  }}

*/
  // total number of spins
  // equal to the number of visible units
  inline int Nspins() const { return nv_; }

  inline int NHspins() const { return nh_; }

  int GetNpar() const { return npar_; }

  int GetNvtot() const { return nvtot_; }
  bool Isospin() const { return isospin_; }
};
