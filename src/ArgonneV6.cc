/*
############################ COPYRIGHT NOTICE ##################################

Code provided by L.Lazzarino, written by L.Lazzarino , October 2021.

Permission is granted for anyone to copy, use, modify, or distribute the
accompanying programs and documents for any purpose, provided this copyright
notice is retained and prominently displayed, along with a complete citation of
the published version of the paper:
 ______________________________________________________________________________
| L.Lazzarino                                                                  |
| Solving the quantum many-body problem with artificial neural-networks        |
|______________________________________________________________________________|

The programs and documents are distributed without any warranty, express or
implied.

These programs were written for research purposes only, and are meant to
demonstrate and reproduce the main results obtained in the paper.

All use of these programs is entirely at the user's own risk.

################################################################################
*/

#include "eigen3/Eigen/Dense"
#include <complex>
#include <iostream>
#include <vector>

class ArgonneV6 {

  // dimension of space
  int dim_ = 3;

  // dimension of the cubic lattice
  int l_;

  // number of spins
  const int nspins_;

  // number of particles
  const double Mag_;

  int connectedcomponents_;

  // width of the trap
  const double traplength_;
  double dx_;

  // constants
  const double hbar_;
  const double mass_;
  double KEconst_;

  // pre-computed quantities
  std::vector<std::complex<double>> mel_;
  std::vector<std::vector<int>> flipsh_;
  std::vector<std::vector<int>> nn_; // near neighbours

  double alpha_ee; // coupling constant Ve-e interaction
  double alpha_ne; // coupling constant of Vn-e
  int COM; // position of the center of mass (fissata nel centro della scatola)

  // Costanti fisiche per il calcolo degli operatori Vi(r)
  static constexpr double hc = 197.327053;
  static constexpr double mpi0 = 134.9739;
  static constexpr double mpic = 139.5675;
  static constexpr double mp = 938.27231;
  static constexpr double mn = 939.56563;
  static constexpr double alpha = 1.0 / 137.035989;
  static constexpr double mup = 2.7928474;
  static constexpr double mun = -1.9130427;
  Eigen::Matrix4d spincoupling_;
  std::vector<Eigen::Matrix2cd> sigma_;

public:
  ArgonneV6(int nspins, int Mag, double traplength = 4., double alpha_ne = -2.0,
            double hbar = hc, double mass = (mp + mn) / 2)
      : nspins_(nspins), Mag_(Mag), traplength_(traplength), alpha_ne(alpha_ne),
        hbar_(hbar), mass_(mass) {
    std::cout << "hbar=hc=" << hbar << "\t mass=" << mass << std::endl;
    connectedcomponents_ =
        1 + 6 * Mag_; // devo mettere hbar e mass per calcolare bene l energia
                      // cinetica? penso di si

    l_ = cbrt(nspins_); // lato= radice cubica del numero di spin("volume")
    dx_ = traplength_ / l_;
    std::cout << "dx=" << dx_ << std::endl;
    KEconst_ = -std::pow(hbar_, 2) / (2 * mass_ * std::pow(dx_, 2));
    // KEconst_=0.0;
    InitLattice();
    Init();
    InitPauli();        // crea matrici di pauli
    SpincouplingInit(); // salva la matrice di accoppiamento degli spin
  }
  double spincoupling(int bra, int ket) { return spincoupling_(bra, ket); }
  void InitLattice() {
    if ((l_ * l_ * l_) != nspins_) {
      std::cerr << "# Error , the number of spins is not compabitle with a "
                   "cubic lattice "
                << std::endl;
      std::abort();
    }
  }

  int getDimensionSpace() { return dim_; }

  int getMagnetizazion() { return Mag_; }
  double GetTraplength() { return traplength_; }

  void Init() { // parte indep dalla configurazione

    alpha_ne = -2.; // coupling constant Vn-e (numero di protoni nel nucleo)
    alpha_ee = 1.;
    COM = (l_ * l_ * l_ - 1) / 2.; // vero se dispari

    mel_.resize(connectedcomponents_);
    flipsh_.resize(connectedcomponents_);

    // no spins flipped = no particle moves:
    mel_[0] =
        KEconst_ * (-2) * dim_ * Mag_; // dim_== dof, laplaciano discreto dell
    std::cout << "mel_[0]=" << mel_[0] << std::endl;
    // energia cinetica= -2*dim per ogni particella. mel[0] ha la traccia di T
    // per tutte le particelle
    std::cout << "dx_ = " << dx_ << " fm, KEconst_ = " << KEconst_ << " MeV"
              << std::endl;

    std::cout << "-----------Keconst=" << KEconst_ << "--------------"
              << std::endl;
    flipsh_[0] = std::vector<int>();
    std::cout << "Kinetic only: " << KEconst_ * (-2) * dim_ * Mag_ << std::endl;

    for (int i = 1; i < connectedcomponents_; i++) {
      mel_[i] = KEconst_ * (1); // termini offdiagonal di T, quelli accessibili
                                // con un salto della particella
    }
    std::cout << "# Deuteron in a box (" << l_ << "x" << l_ << "x" << l_
              << ") and COM in " << "(" << int(int(COM / l_) / l_) << ","
              << int(COM / l_) % l_ << "," << COM % l_ << ")" << std::endl;
  }

  int MinFlips() const { // mantains magnetization
    return 2;
  }

  // Calcolo coordinate di una particella s
  std::vector<int> CalculateCoo(int s) {
    // s encoded as x*l_*l_ + y*l_ + z
    int z = s % l_;
    int y = (s / l_) % l_;
    int x = s / (l_ * l_);
    return std::vector<int>{x, y, z};
    // ritorna (x, y, z)
  }

  void StampaCoo(int s) {
    int z = s % l_;
    int y = (s / l_) % l_;
    int x = s / (l_ * l_);
    std::cout << "(" << x << "," << y << "," << z << ")" << std::endl;
  }

  double Distance(int s1, int s2) {
    std::vector<int> COOs1 = CalculateCoo(s1);
    std::vector<int> COOs2 = CalculateCoo(s2);
    double a = pow(COOs1[0] - COOs2[0], 2);
    double b = pow(COOs1[1] - COOs2[1], 2);
    double c = pow(COOs1[2] - COOs2[2], 2);
    double dist = sqrt(a + b + c);
    return dist;
  }
  std::vector<int> FindNeighbours(int s) {

    int dim = dim_;
    bool pbc_ = false;
    std::vector<int> n_;
    n_.resize(dim * 2); // 1d -> 2, 2d -> 4, 3d -> 6
    // x- , x+
    n_[0] = pbc_ ? PbcH(s - 1, s) : H(s - 1, s);
    n_[1] = pbc_ ? PbcH(s + 1, s) : H(s + 1, s);

    // y- , y+
    n_[2] = pbc_ ? PbcV(s - l_, s) : V(s - l_, s);
    n_[3] = pbc_ ? PbcV(s + l_, s) : V(s + l_, s);

    // z- , z+
    n_[4] = pbc_ ? PbcI(s - l_ * l_) : I(s - l_ * l_);
    n_[5] = pbc_ ? PbcI(s + l_ * l_) : I(s + l_ * l_);

    return n_; // -1 se fuori dalla box
  }

  std::vector<int> FindNeighboursGlobal(int s, int nv) {
    int channel = (int)(s / nv); // capisco a quale canale appartiene il sito
    int offset = channel * nv;   // blocco isospinico
    int local = s % nv;          // indice nel blocco locale

    // trova vicini nel blocco locale
    std::vector<int> nn_local = FindNeighbours(local);

    // riporta agli indici globali
    for (auto &n : nn_local)
      if (n >= 0)
        n += offset;

    return nn_local;
  }

  // Small functions to find neighbours
  // Horizontal Pbc
  int PbcH(int nn, int s) {
    if (s % l_ == 0 && nn == (s - 1))
      return (s + l_ - 1);
    else if ((s + 1) % l_ == 0 && nn == (s + 1))
      return (s - l_ + 1);
    else
      return nn;
  }

  // Depht (towards and back) Pbc
  int PbcI(int nn) {
    if (nn >= nspins_)
      return (nn - nspins_);
    else if (nn < 0)
      return (nspins_ + nn);
    else
      return nn;
  }

  // Vertical Pbc
  int PbcV(int nn, int s) {
    if (nn >= nspins_)
      return (nn - l_ * l_);
    else if (nn < 0)
      return (l_ * l_ + nn);
    else if (int(nn / (l_ * l_)) == int(s / (l_ * l_)))
      return nn;
    else if (nn > s)
      return nn - l_ * l_;
    else
      return nn + l_ * l_;
  }

  // Horizontal without pbc
  int H(int nn, int s) {
    if (s % l_ == 0 && nn == (s - 1))
      return (-1);
    else if ((s + 1) % l_ == 0 && nn == (s + 1))
      return (-1);
    else
      return nn;
  }

  // Depht (towards and back) without pbc
  int I(int nn) {
    if (nn >= nspins_)
      return -1;
    else if (nn < 0)
      return -1;
    else
      return nn;
  }

  // Vertical without pbc
  int V(int nn, int s) {
    if (nn >= nspins_)
      return -1;
    else if (nn < 0)
      return -1;
    else if (int(nn / (l_ * l_)) == int(s / (l_ * l_)))
      return nn;
    else
      return (-1);
  }

  void FindConn(const std::vector<int> &state,
                const std::vector<int> &SpinIndexAll,
                std::vector<std::vector<int>> &flipsh,
                std::vector<std::complex<double>> &mel, int Nup, int nv) {

    mel.resize(connectedcomponents_);
    flipsh.resize(connectedcomponents_);

    // double rab_special = std::sqrt(3.0 / 16.0) *   dx_; // distanza minima
    // tra due particles nel reticolo
    double rab_special = 0.0;

    // dentro init comincia gia a calcolare alcuni elementi di matrice
    // (diagonali dell energia cinetica e quelli offdiagonal) li copio
    nn_.resize(Mag_);
    mel = mel_;
    flipsh = flipsh_;
    //  std::cout<<"mel[0]="<<mel[0]<<std::endl;
    int index = 1;

    // 1) Vicini GLOBALI "raw" (solo bordi, NO occupazione) + diagonale T
    std::vector<std::vector<int>> nn_raw(Mag_);
    for (int p = 0; p < Mag_; ++p) {
      nn_raw[p] = FindNeighboursGlobal(SpinIndexAll[p], nv); // 6 vicini

      // conta SOLO i bordi "positivi": +x,+y,+z  → indici 1,3,5 in
      // FindNeighbours
      int missing_pos = 0;
      if (nn_raw[p][1] < 0)
        ++missing_pos; // +x  (I: s + l_*l_)
      if (nn_raw[p][3] < 0)
        ++missing_pos; // +y  (V: s + l_)
      if (nn_raw[p][5] < 0)
        ++missing_pos; // +z  (H: s + 1)

      // aggiorna UNA VOLTA per particella, non dentro il for(d)
      mel[0] += KEconst_ * (-missing_pos);
    }

    // filtro l’OCCUPAZIONE sul canale giusto, partendo da nn_raw
    for (int p = 0; p < Mag_; ++p) {
      nn_[p] = nn_raw[p];
      if (SpinIndexAll[p] < nv ||
          (SpinIndexAll[p] >= 2 * nv && SpinIndexAll[p] < 3 * nv)) {
        for (int j = 0; j < (int)nn_[p].size(); ++j)
          for (int k = 0; k < Nup; ++k)
            if (nn_[p][j] == SpinIndexAll[k])
              nn_[p][j] = -1;
      } else {
        for (int j = 0; j < (int)nn_[p].size(); ++j)
          for (int k = Nup; k < Mag_; ++k)
            if (nn_[p][j] == SpinIndexAll[k])
              nn_[p][j] = -1;
      }
    }

    //  OFF-DIAG cinetici 1..6*Mag: KEconst oppure 0
    for (int p = 0; p < Mag_; ++p) {
      for (int j = 0; j < (int)nn_[p].size(); ++j) {
        if (nn_[p][j] >= 0) {
          flipsh[index] = {SpinIndexAll[p], nn_[p][j]};
          mel[index] = KEconst_;
        } else {
          flipsh[index].clear();
          mel[index] = 0.0;
        }
        ++index;
      }
    }

    if (index != 1 + 6 * Mag_) {
      std::cerr << "❌ index=" << index << " atteso=" << 1 + 6 * Mag_ << "\n";
      std::exit(1);
    }

    //<i|H|j> i è lo stato attuale (p, n, up,down) mentre j è una diversa
    // combinazione di spin/isospin. devo sommare s tutti i j possibili

    // calcolo distanza tra i nucleoni
    // distanza e direzione
    double rab = Distance(SpinIndexAll[0] % nv, SpinIndexAll[1] % nv) * dx_;
    assert(rab <= std::sqrt(3) * traplength_);
    if (rab < rab_special)
      rab = rab_special;
    // std::cout << "site 1:" << SpinIndexAll[0] % nv
    //           << ", site 2=" << SpinIndexAll[1] % nv << std::endl;
    std::vector<double> r_hat =
        UnitVector(SpinIndexAll[0] % nv, SpinIndexAll[1] % nv);
    double checknorm = 0.0;
    for (int i = 0; i < 3; i++) {
      checknorm += std::pow(r_hat[i], 2);

      if (!std::isfinite(r_hat[i])) {
        std::cerr << "nan in r_hat" << std::endl;
        exit(1);
      }
    }

    assert(std::fabs(checknorm - 1.0) < 1e-7 || checknorm == 0);
    // std::cout<<"rab="<<rab<<std::endl;
    //  calcolo gli operatori radiali
    std::vector<double> V_i = AV6operators(rab);
    for (int i = 0; i < V_i.size(); i++) {
      if (!std::isfinite(std::real(V_i[i])) ||
          !std::isfinite(std::imag(V_i[i]))) {
        std::cerr << "⚠️ NaN in radial operator V[" << i << "], rab=" << rab
                  << std::endl;
      }
    }
    // calcolo la matrice dell'S12 per questo raggio (normalizzato)
    Eigen::Matrix4cd S12 = Compute_S12(r_hat);
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        if (!std::isfinite(std::real(S12(i, j))) ||
            !std::isfinite(std::imag(S12(i, j)))) {
          std::cerr << "naN in S12 operator" << std::endl;
          exit(1);
        }
      }
    }

    // deuterio = isospin opposti
    int t1, t2, t4;
    // spinindexall è ordinato, nel primo posto cie sempre un protone e nel
    // secondo un neutrone
    t1 = +1;
    t2 = -1;
    int s1, s2;               // spin nello stato corrente
    if (SpinIndexAll[0] < nv) // protone up
      s1 = +1;
    else
      s1 = -1;                    // protone down
    if (SpinIndexAll[1] < 3 * nv) // neutrone up
      s2 = +1;
    else // neutrone down
      s2 = -1;
    int braT = MatrixIndex(t1, t2);
    int braS = MatrixIndex(s1, s2);
    // loop sulle combinazioni di isospin per lo stato di arrivo
    for (int t3 = -1; t3 <= 1; t3 += 2) {
      t4 = -t3;                       // isospin flip (deuterio)
      int ketT = MatrixIndex(t3, t4); // indicizzo lo stato di arrivo
      // loop sulle combinazioni possibili degli spin per lo stato di arrivo
      for (int s3 = -1; s3 <= 1; s3 += 2) {
        for (int s4 = -1; s4 <= 1; s4 += 2) {
          int ketS = MatrixIndex(s3, s4);
          // std::cout<<"going from ("<<s1<<","<<t1<<")" <<"to -->
          // ("<<s3<<","<<t3<<")"<<std::endl;
          // std::cout<<"("<<s2<<","<<t2<<")" <<"to -->
          // ("<<s4<<","<<t4<<")"<<std::endl;
          std::complex<double> Vij;
          std::complex<double> V1 = V_i[0] + V_i[1] * spincoupling_(braS, ketS);
          if (!std::isfinite(std::real(V1)) || !std::isfinite(std::imag(V1))) {
            std::cerr << "⚠️ NaN in V1, rab=" << rab << std::endl;
            Vij = {0.0, 0.0};
          }
          std::complex<double> V2 = V_i[2] * spincoupling_(braT, ketT);
          if (!std::isfinite(std::real(V2)) || !std::isfinite(std::imag(V2))) {
            std::cerr << "⚠️ NaN in V2, rab=" << rab << std::endl;
            Vij = {0.0, 0.0};
          }
          std::complex<double> V3 =
              V_i[3] * spincoupling_(braS, ketS) * spincoupling_(braT, ketT);
          if (!std::isfinite(std::real(V3)) || !std::isfinite(std::imag(V3))) {
            std::cerr << "⚠️ NaN in V3, rab=" << rab << std::endl;
            Vij = {0.0, 0.0};
          }
          std::complex<double> V4 = V_i[4] * S12(braS, ketS);
          if (!std::isfinite(std::real(V4)) || !std::isfinite(std::imag(V4))) {
            std::cerr << "⚠️ NaN in V4, rab=" << rab << std::endl;
            Vij = {0.0, 0.0};
          }
          std::complex<double> V5 =
              V_i[5] * S12(braS, ketS) * spincoupling_(braT, ketT);
          if (!std::isfinite(std::real(V5)) || !std::isfinite(std::imag(V5))) {
            std::cerr << "⚠️ NaN in V5, rab=" << rab << std::endl;
            Vij = {0.0, 0.0};
          }
          Vij = V1 + V2 + V3 + V4 + V5;

          //    std::cout<<"VIJ="<<Vij.real()<<std::endl;
          // std::cout<<"parto da: ("<<SpinIndexAll[0]<<" ,
          // "<<SpinIndexAll[1]<<")"<<std::endl<<std::endl<<std::endl<<std::endl;
          if (ketT == braT && ketS == braS) {
            // se succede è un termine diagonale!-> va in mel_[0] e non aggiorna
            // flipsh
            mel[0] += Vij;

            // metti potenziale a zero e vedi cosa succede, dovrei avere f d
            // onda di due particelle libere(non interagenti)

          } else {
            // calcolo elemento di matrice, poi decido dove metterlo in syncro
            // con il flips relativo (offdiagonal) (devo ricostruire la mossa
            // fatta per passarla a pop)
            if (braT == ketT &&
                braS != ketS) { // protone e/o neutrone hanno cambiato lo spin,
                                // devo capire chi dei due e come
              // std::cout<<"cambio spin ma isospin no"<<std::endl;
              if (s1 == s3 && s2 != s4) { // neutrone ha cambiato spin
                //    std::cout<<"neutrone cambia spin"<<std::endl;
                int flip = +1; // se parte da up e va in down shifta di +nv
                               // viceversa di -nv
                if (s2 == -1)
                  flip = -1;
                // ricorda che spinindexall è ordinato, quindi in 0 sempre
                // protone e in 1 sempre neutrone
                flipsh.push_back(
                    {SpinIndexAll[1], SpinIndexAll[1] + (flip * nv)});
                mel.push_back(Vij);
              }

              else if (s1 != s3 && s2 == s4) { // solo protone ha cambiato spin
                // std::cout<<"protone cambia spin"<<std::endl;
                int flip = +1; // se parte da up e va in down shifta di +nv
                               // viceversa di -nv
                if (s1 == -1)
                  flip = -1;
                // ricorda che spinindexall è ordinato, quindi in 0 sempre
                // protone e in 1 sempre neutrone
                flipsh.push_back(
                    {SpinIndexAll[0], SpinIndexAll[0] + (flip * nv)});
                mel.push_back(Vij);

              } else if (s1 != s3 &&
                         s2 != s4) { // entrambi hanno flippato lo spin
                // std::cout<<"entrambi cambiano spin"<<std::endl;
                int flip1 = +1;
                if (s1 == -1)
                  flip1 = -1;

                int flip2 = +1;
                if (s2 == -1)
                  flip2 = -1;

                flipsh.push_back(
                    {SpinIndexAll[0], SpinIndexAll[0] + (flip1 * nv),
                     SpinIndexAll[1], SpinIndexAll[1] + (flip2 * nv)});
                mel.push_back(Vij);
              }

            }

            else if (braS == ketS &&
                     braT != ketT) { // spin invariati ma protone e neutrone
                                     // scambiati. es: protone spin up->
                                     // neutrone spinup neutrone spin down->
                                     // protone spin down
              // se un neutrone diventa protone, anche l altro deve scambiarsi
              // per forza... non servono controlli, shifto entrambi di 2nv
              // std::cout<<"spin non cambiano ma isospin si-> flip
              // pn"<<std::endl;

              flipsh.push_back({SpinIndexAll[0], SpinIndexAll[0] + (2 * nv),
                                SpinIndexAll[1], SpinIndexAll[1] - (2 * nv)});
              mel.push_back(Vij);

            }

            else if (braS != ketS &&
                     braT != ketT) { // cambio di spin e isospin es protone spin
                                     // up-> neutrone spin down
              //  std::cout<<"cambiano sia spin che isospin"<<std::endl;
              if (s1 != s3 && s2 != s4) { // scambio protone e neutrone,
                                          // entrambi cambiano spin
                int flip1 = +1; // al solito se parte da up devo shiftare di un
                                // +nv extra
                if (s1 == -1)
                  flip1 = -1;
                // std::cout<<"scmabio pn ed entrambi flippano lo
                // spin"<<std::endl;
                int flip2 = +1;
                if (s2 == -1)
                  flip2 = -1;
                flipsh.push_back(
                    {SpinIndexAll[0], SpinIndexAll[0] + (2 * nv + flip1 * nv),
                     SpinIndexAll[1], SpinIndexAll[1] - (2 * nv - flip2 * nv)});
                mel.push_back(Vij);
              }

              else if (s1 != s3 &&
                       s2 == s4) { // flip protone e neutrona, ma solo il
                                   // protone cambia segno. es: (p su, n giu)->
                                   // (n giu, p su)

                int flip = 1;
                if (s1 == -1)
                  flip = -1;
                // std::cout<<"protone spin down-> neutrone spin up, neutrone
                // spin up-> protone spin up"<<std::endl;
                flipsh.push_back({SpinIndexAll[0],
                                  SpinIndexAll[0] + (2 * nv + flip * nv),
                                  SpinIndexAll[1], SpinIndexAll[1] - (2 * nv)});

                mel.push_back(Vij);

              } else if (s1 == s3 && s2 != s4) {
                // std::cout<<"protone spin up-> neutrone spin up, neutrone spin
                // up-> protone spin down"<<std::endl;
                int flip = 1;
                if (s2 == -1)
                  flip = -1;

                flipsh.push_back({SpinIndexAll[0], SpinIndexAll[0] + (2 * nv),
                                  SpinIndexAll[1],
                                  SpinIndexAll[1] - ((2 - flip) * nv)});

                mel.push_back(Vij);
              }
            }
          }
        }
      }
    }

    if (mel.size() != flipsh.size()) {
      std::cerr << "❌ ERROR: mel and flipsh size mismatch: " << mel.size()
                << " vs " << flipsh.size() << std::endl;
      std::exit(1);
    }
    for (size_t k = 0; k < mel.size(); ++k) {
      if (std::isnan(mel[k].real()) || std::isnan(mel[k].imag())) {
        std::cerr << "❌ mel[" << k << "] è NaN in findconn()\n";
        std::exit(1);
      }
    }

    // std::cout<<"fine findconn"<<std::endl;
  }

  inline int MatrixIndex(int a, int b) {
    int ia = (a == 1 ? 1 : 0);
    int ib = (b == 1 ? 1 : 0);
    return 2 * ia + ib; // restituisce 0..3 //indiciza gli spin
  }

  void SpincouplingInit() {
    // inizializza la matrice dei si dot sj, usata per accoppiare sia spin che
    // isospin
    spincoupling_ = Eigen::Matrix4d::Zero();
    for (int s1 = 0; s1 <= 1; s1++) {
      for (int s2 = 0; s2 <= 1; s2++) {
        for (int s3 = 0; s3 <= 1; s3++) {
          for (int s4 = 0; s4 <= 1; s4++) {
            int bra = 2 * s1 + s2;
            int ket = 2 * s3 + s4;
            // Mappa (s1, s2) → indice nella base computazionale (4 stati
            // totali)
            //
            // Stati di spin (s1, s2):
            //   (0,0)  → |00⟩  ≡ indice 0
            //   (0,1)  → |01⟩  ≡ indice 1
            //   (1,0)  → |10⟩  ≡ indice 2
            //   (1,1)  → |11⟩  ≡ indice 3

            double factor = 0.0;
            if (s1 == s4 && s2 == s3)
              factor += 2;
            if (s1 == s3 && s2 == s4)
              factor -= 1;

            spincoupling_(bra, ket) += factor;
          }
        }
      }
    }
  }

  // calcola il versore dal sito 1 al sito 2
  std::vector<double> UnitVector(int s1, int s2) {
    std::vector<double> r_hat(dim_, 0.0);
    double norm = 0.0;
    std::vector<int> COOs1 = CalculateCoo(s1);
    std::vector<int> COOs2 = CalculateCoo(s2);
    if (s1 == s2)
      return r_hat;
    for (int i = 0; i < dim_; i++) {
      r_hat[i] = static_cast<double>(COOs2[i] - COOs1[i]);
      norm += pow(r_hat[i], 2);
    }

    for (int i = 0; i < dim_; i++) {
      r_hat[i] /= sqrt(norm);
    }

    return r_hat;
  }
  void InitPauli() {

    sigma_.resize(3);

    sigma_[0] << 0.0, 1.0, 1.0, 0.0; // σ_x

    sigma_[1] << 0.0, std::complex<double>(0, -1), std::complex<double>(0, 1),
        0.0; // σ_y

    sigma_[2] << 1.0, 0.0, 0.0, -1.0; // σ_z
  }

  // fa il prodotto tensoriale tra due matrici 2x2(pauli)
  Eigen::Matrix4cd kron(const Eigen::Matrix2cd &A, const Eigen::Matrix2cd &B) {
    Eigen::Matrix4cd result;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        result.block<2, 2>(2 * i, 2 * j) = A(i, j) * B;
      }
    }

    return result;
  }

  // calcola la matrice 4x4 di s12 in un dato r (versore)
  Eigen::Matrix4cd Compute_S12(const std::vector<double> &r) {

    if (r[0] * r[0] + r[1] * r[1] + r[2] * r[2] == 0) {
      Eigen::Matrix4cd null = Eigen::Matrix4cd::Zero();
      return null;
    }
    Eigen::Matrix4cd first_term = Eigen::Matrix4cd::Zero();
    Eigen::Matrix4cd second_term = Eigen::Matrix4cd::Zero();

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        Eigen::Matrix4cd sigma1_i =
            kron(sigma_[i], Eigen::Matrix2cd::Identity());
        Eigen::Matrix4cd sigma2_j =
            kron(Eigen::Matrix2cd::Identity(), sigma_[j]);
        first_term += 3.0 * r[i] * r[j] * (sigma1_i * sigma2_j);
      }
    }

    for (int i = 0; i < 3; i++) {
      Eigen::Matrix4cd sigma1_i = kron(sigma_[i], Eigen::Matrix2cd::Identity());
      Eigen::Matrix4cd sigma2_i = kron(Eigen::Matrix2cd::Identity(), sigma_[i]);
      second_term += sigma1_i * sigma2_i;
    }

    Eigen::Matrix4cd S12 = first_term - second_term;
    return S12;
  }

  // calcolo degli operatori Vi(r) (dal codice fortran)

  inline double yc(double t, double x) { return std::exp(-t) / x; }
  inline double yt(double t, double x) {
    return (1.0 + 3.0 / t + 3.0 / (t * t)) * std::exp(-t) / x;
  }
  inline double yls(double t, double x) {
    return -(1.0 + t) * std::exp(-t) / (x * x * x);
  }
  inline double yl2(double t, double x) {
    return (1.0 + 2.0 / t) * std::exp(-t) / (x * x * x);
  }

  const std::vector<double> AV6operators(double r) {

    std::vector<double> vnn;

    // Costanti
    double mpi = (mpi0 + 2.0 * mpic) / 3.0; // pioni
    double mu0 = mpi0 / hc;
    double muc = mpic / hc;
    double mu = mpi / hc;
    double fsq = 0.075; // costanti che regolano le interazioni
    double cpi = 2.1;
    double rws = 0.5;
    double aiws = 5.0;

    double x = mu * r;
    double x0 = mu0 * r;
    double xc = muc * r;

    double ypi0, ypic, tpi0, tpic, ypi, tpi;

    if (r <= 1e-4) { // se le particelle sono troppo vicine usa un espansione
                     // dei termini di yukawa (altrimenti e^-r esplode)
      double tpi_tmp = 3.0 * cpi * cpi * r / (mu * mu * mu);
      ypi0 = std::pow(mpi0 / mpic, 2) * (mpi0 / 3.0) * cpi * r / mu0;
      tpi0 = 3 * cpi * ypi0 / (mu0 * mu0);
      ypic = (mpic / 3.0) * cpi * r / muc;
      tpic = 3 * cpi * ypic / (muc * muc);
      tpi = tpi_tmp;
    } else { // se sono lontati usa i termini di yukawa completi
      double rcut = 1.0 - std::exp(-cpi * r * r);
      ypi = std::exp(-x) * rcut / x;
      tpi = (1.0 + (3.0 + 3.0 / x) / x) * ypi * rcut;
      ypi0 =
          std::pow(mpi0 / mpic, 2) * (mpi0 / 3.0) * std::exp(-x0) * rcut / x0;
      tpi0 = (1.0 + (3.0 + 3.0 / x0) / x0) * ypi0 * rcut;
      ypic = (mpic / 3.0) * std::exp(-xc) * rcut / xc;
      tpic = (1.0 + (3.0 + 3.0 / xc) / xc) * ypic * rcut;
    }

    ypi0 = fsq * ypi0; // le prime 4 moltiplicano le funzioni di yukawa per l
                       // intensità dell interazione, regolata da fsq
    ypic = fsq * ypic;
    tpi0 = fsq * tpi0;
    tpic = fsq * tpic;
    double tpi2 = tpi * tpi; // funzione tensioriale media ^2
    double ws =
        1 /
        (1 + exp((r - rws) * aiws)); // fattore woods saxon per il corto raggio
    double ws0 = 1 / (1 + exp(-rws * aiws));
    double wsp = ws * (1 + aiws * exp(-rws * aiws) * ws0 *
                               r); // derivate e correzioni di forma
    double wsx = ws * x;
    double wsx2 = wsx * x;
    double dypi00 =
        pow((mpi0 / mpic), 2) * (mpi0 / 3) * cpi / mu0; // derivate delle yukawa
    double dypic0 = (mpic / 3) * cpi / muc;
    double ypi0p = ypi0 - fsq * dypi00 * ws * r / ws0; // correzione sulle
                                                       // yukawa
    double ypicp = ypic - fsq * dypic0 * ws * r / ws0;
    ypi = (ypi0 + 2 * ypic) / 3; // media pesata sulle componenti del pione
    tpi = (tpi0 + 2 * tpic) / 3;
    // Questo blocco è la tabella dei potenziali parziali:
    // ogni riga definisce l’energia potenziale per una combinazione di spin e
    // isospin,
    //  con parametri calibrati sperimentalmente.
    double p11pp = -7.62701 * tpi2 + 1815.4920 * wsp + 1847.8059 * wsx2 + ypi0p;
    double p11np = -7.62701 * tpi2 + 1813.5315 * wsp + 1847.8059 * wsx2 -
                   ypi0p + 2 * ypicp;
    double p11nn = -7.62701 * tpi2 + 1811.5710 * wsp + 1847.8059 * wsx2 + ypi0p;
    double pt1pp = 1.07985 * tpi2 - 190.0949 * wsx - 811.2040 * wsx2 + tpi0;
    double pt1np =
        1.07985 * tpi2 - 190.0949 * wsx - 811.2040 * wsx2 - tpi0 + 2 * tpic;
    double pt1nn = 1.07985 * tpi2 - 190.0949 * wsx - 811.2040 * wsx2 + tpi0;
    double pls1 = -.62697 * tpi2 - 570.5571 * wsp + 819.1222 * wsx2;
    double pl211 = 0.06709 * tpi2 + 342.0669 * wsp - 615.2339 * wsx2;
    double pls21 = 0.74129 * tpi2 + 9.3418 * wsp - 376.4384 * wsx2;
    double p10 =
        -8.62770 * tpi2 + 2605.2682 * wsp + 441.9733 * wsx2 - ypi0p - 2 * ypicp;
    double pt0 =
        1.485601 * tpi2 - 1126.8359 * wsx + 370.1324 * wsx2 - tpi0 - 2 * tpic;
    double pls0 = 0.10180 * tpi2 + 86.0658 * wsp - 356.5175 * wsx2;
    double pl210 = -.13201 * tpi2 + 253.4350 * wsp - 1.0076 * wsx2;
    double pls20 = 0.07357 * tpi2 - 217.5791 * wsp + 18.3935 * wsx2;
    double p01pp = -11.27028 * tpi2 + 3346.6874 * wsp - 3 * ypi0p;
    double p01np =
        -10.66788 * tpi2 + 3126.5542 * wsp - 3 * (-ypi0p + 2 * ypicp);
    double p01nn = -11.27028 * tpi2 + 3342.7664 * wsp - 3 * ypi0p;
    double pl201 = 0.12472 * tpi2 + 16.7780 * wsp;
    double p00 = -2.09971 * tpi2 + 1204.4301 * wsp - 3 * (-ypi0p - 2 * ypicp);
    double pl200 = -.31452 * tpi2 + 217.4559 * wsp;
    double p11 = (p11pp + p11nn + p11np) / 3;
    double p11cd = (0.5 * (p11pp + p11nn) - p11np) / 6;
    double p11cs = (p11pp - p11nn) / 4;
    double pt1 = (pt1pp + pt1nn + pt1np) / 3;
    double pt1cd = (0.5 * (pt1pp + pt1nn) - pt1np) / 6;
    double pt1cs = (pt1pp - pt1nn) / 4;
    double p01 = (p01pp + p01nn + p01np) / 3;
    double p01cd = (0.5 * (p01pp + p01nn) - p01np) / 6;
    double p01cs = (p01pp - p01nn) / 4;

    p10 = p10 - .3 * pls0;

    vnn.push_back(0.0625 * (9 * p11 + 3 * p10 + 3 * p01 + p00));
    vnn.push_back(0.0625 * (3 * p11 - 3 * p10 + p01 - p00));
    vnn.push_back(0.0625 * (3 * p11 + p10 - 3 * p01 - p00));
    vnn.push_back(0.0625 * (p11 - p10 - p01 + p00));
    vnn.push_back(0.25 * (3 * pt1 + pt0));
    vnn.push_back(0.25 * (pt1 - pt0));

    return vnn;
  }
};
