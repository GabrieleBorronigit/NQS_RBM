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

// Transverse-field Ising model in 1d
class Coulomb3d {

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
  int N;           // position of the nucleus

public:
  Coulomb3d(int nspins, int Mag, double traplength = 4., double alpha_ne = -2.0,
            double hbar = 1., double mass = 1.)
      : nspins_(nspins), Mag_(Mag), traplength_(traplength), alpha_ne(alpha_ne),
        hbar_(hbar), mass_(mass) {
    connectedcomponents_ = 1 + 6 * Mag_;

    l_ = cbrt(nspins_); // lato= radice cubica del numero di spin("volume")
    dx_ = traplength_ / l_;
    KEconst_ = -std::pow(hbar_, 2) / (2 * mass_ * std::pow(dx_, 2));

    InitLattice();
    Init();
  }

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

  void Init() { // parte indep dalla configurazione

    alpha_ne = -2.; // coupling constant Vn-e (numero di protoni nel nucleo)
    alpha_ee = 1.;
    N = (l_ * l_ * l_ - 1) / 2.; // vero se dispari

    mel_.resize(connectedcomponents_);
    flipsh_.resize(connectedcomponents_);

    // no spins flipped = no particle moves:
    mel_[0] =
        KEconst_ * (-2) * dim_ * Mag_; // dim_== dof, laplaciano discreto dell
    // energia cinetica= -2*dim per ogni particella. mel[0] ha la traccia di T
    // per tutte le particelle
    flipsh_[0] = std::vector<int>();

    for (int i = 1; i < connectedcomponents_; i++) {
      mel_[i] = KEconst_ * (1); // termini offdiagonal di T, quelli accessibili
                                // con un salto della particella
    }

    /*boundCondMel_.resize(connectedcomponents_-1); //considera boundary diversi
    il muro e un'altra particella	//es {0,0} boundCondMel_[0]= KEconst_ *
    (-1);			//es C{-1,0}
    boundCondMel_[connectedcomponents_-2]= KEconst_ * (-1);	//es C{-1,-1}
    for(int n=1;n<connectedcomponents_-2;n++){
      boundCondMel_[n] = 0.;
    }*/

    std::cout << "# Helium in a box (" << l_ << "x" << l_ << "x" << l_
              << ") and nucleus in " << "(" << int(int(N / l_) / l_) << ","
              << int(N / l_) % l_ << "," << N % l_ << ")" << std::endl;
  }

  // Finds the non-zero matrix elements of the hamiltonian
  // on the given state
  // i.e. all the state' such that <state'|H|state> = mel(state') \neq 0
  // state' is encoded as the sequence of spin flips to be performed on state
  // stateIndex is now usorted. The first Nup elements are spins up.
  void FindConn(const std::vector<int> &stateIndex,
                std::vector<std::vector<int>> &flipsh,
                std::vector<std::complex<double>> &mel, int Nup) {

    mel.resize(connectedcomponents_);
    flipsh.resize(connectedcomponents_);
    double rab_special = std::sqrt(3.0 / 16.0) *
                         dx_; // distanza minima tra due particles nel reticolo
    // assigning pre-computed matrix elements
    mel = mel_;
    flipsh = flipsh_;
    nn_.resize(Mag_);

    // Vn-e interaction dependent on distance e-n
    for (int i = 0; i < Mag_; i++) {
      double rab = Distance(stateIndex[i], N) *
                   dx_; // traplenght=1. -> dx=1/l_ -> rab= distance*dx
      if (rab != 0) {
        mel[0] += double(alpha_ne) / rab;
      } else {
        mel[0] +=
            alpha_ne / rab_special; // no divisioni per zero a distanza nulla
      }

      /*
         for(int j=i+1;j<Mag_;j++){
           double rab = Distance(stateIndex[j],stateIndex[i])*dx_;
         //traplenght=1. -> dx=1/l_ -> rab= distance*dx if(rab==0) { mel[0] +=
         alpha_ee / rab_special;
           }
           else {
           mel[0] += alpha_ee /rab;
           }
         }*/
    }
    // Ve-e interaction dependent on distance e-e
    for (int i = 0; i < Mag_; i++) {
      for (int j = i + 1; j < Mag_; j++) {
        double rab = Distance(stateIndex[j], stateIndex[i]) /
                     double(l_); // traplenght=1. -> dx=1/l_ -> rab= distance*dx
        mel[0] += double(alpha_ee) / rab;
      }
    }

    for (int i = 0; i < Mag_; i++) {
      nn_[i] = FindNeighbours(stateIndex[i]); // find Neighbours of the particle
      if (i < Nup) {
        for (int j = 0; j < nn_[i].size(); j++) {
          for (int k = 0; k < Nup; k++) {
            if (nn_[i][j] == stateIndex[k]) {
              nn_[i][j] = -1;
            }
          }
        }
      } else {
        for (int j = 0; j < nn_[i].size(); j++) {
          for (int k = Nup; k < Mag_; k++) {
            if (nn_[i][j] == stateIndex[k]) {
              nn_[i][j] = -1;
            }
          }
        }
      }
      /*std::vector<int> CooPart=CalculateCoo(stateIndex[i]);
      for(int j=0; j<CooPart.size(); j++){
      //CooPart.size()==3 if(CooPart[j]==0 || CooPart[j]==l_-1){ mel[0]+=
      KEconst_ * (-1);
        }
      }*/
      std::vector<int> CooPart =
          CalculateCoo(stateIndex[i]); // Calcola le coordinate della particella
      int boundaryCount = 0;

      for (int j = 0; j < CooPart.size(); j++) { // CooPart.size() == 3
        if (CooPart[j] == 0 || CooPart[j] == l_ - 1) {
          boundaryCount++;
        }
      }

      // Assegna il coefficiente in base al numero di confini toccati
      if (boundaryCount == 3) {
        mel[0] += KEconst_ * (-3); // Vertici
      } else if (boundaryCount == 2) {
        mel[0] += KEconst_ * (-2); // Spigoli
      } else if (boundaryCount == 1) {
        mel[0] += KEconst_ * (-1); // Facce
      }
    }

    int index = 1;
    for (int i = 0; i < Mag_; i++) {
      for (int k = 0; k < 6; k++) { // 6= nn_[i].size()
        if (i < Nup) {
          if (nn_[i][k] >=
              0) { // nn contiene i k=0,..,5 primi vicini per la particella i
            flipsh[index] = std::vector<int>{stateIndex[i], nn_[i][k]};
            /*std::cout<<stateIndex[i]<<"  "<<vicino<<std::endl;
            std::cout<<nn_[i][k]<<std::endl;*/
            index++;
          } else {
            mel[index] = 0.;
            flipsh[index] = std::vector<int>();
            index++;
          }
        } else {
          if (nn_[i][k] >= 0) {
            flipsh[index] =
                std::vector<int>{stateIndex[i] + nspins_, nn_[i][k] + nspins_};
            /*std::cout<<stateIndex[i]<<"  "<<vicino<<std::endl;
            std::cout<<nn_[i][k]<<std::endl;*/
            index++;
          } else {
            mel[index] = 0.;
            flipsh[index] = std::vector<int>();
            index++;
          }
        }
      }
    }

    // implementing boundary conditions by turning mel to 0 and emptying flipsh
    /*std::vector<int> partPosD;
    partPosD.resize(connectedcomponents_-1); //andranno da 0 a 2*Mag_
    partPosD={0,0,0,0} std::vector<int> boundCond;
    boundCond.resize(connectedcomponents_-1);// boundCond={0,0,0,0}

    boundCond[0]=-1;
    boundCond[connectedcomponents_-2]=nspins_;
    for(int n=1;n<connectedcomponents_-2;n++){
      int v = n/2 + (n%2) * 2 - 1; //particella vicina
      boundCond[n]= stateIndex[v];
    }	//es partPosD={0,0,0,0}	es boundCond={-1,stateIndex[1],stateIndex[1],10}

    for(int n=0;n<connectedcomponents_-1;n++){
      int d = n/2; //raddoppio posizioni particelle	//n=0 n=1 -> d=0  //n=2
    n=3 -> d=1 partPosD[n]=stateIndex[d];

      if(std::abs(partPosD[n]-boundCond[n]) == 1){
        mel[0] += boundCondMel_[n];
        mel[n+1] = 0;
        flipsh[n+1]=std::vector<int>();
      }
      else{
        flipsh[n+1]=std::vector<int>({stateIndex[n/2], stateIndex[n/2] + (n%2) *
    2 - 1});
      }
    }*/
  }

  int MinFlips() const { // mantains magnetization
    return 2;
  }

  // Calcolo coordinate di una particella s
  std::vector<int> CalculateCoo(
      int s) { // ritornava (z,y,x), l ho corretta e ora ritorna (x, y, z)
    return std::vector<int>{int(s % l_), int(s / l_) % l_, int((s / l_) / l_)};
  }

  void StampaCoo(int s) {
    std::cout << "(" << s % l_ << "," << int(s / l_) % l_ << ","
              << int(int(s / l_) / l_) << ")" << std::endl;
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

  std::vector<int>
  FindNeighbours(int s) { // PBC if pbc_=true; else spins in a box. If there's
                          // not a near slot -1.

    int dim = dim_;
    bool pbc_ = false;

    std::vector<int> n_;
    n_.resize(dim * 2); // 1d -> 2, 2d -> 4, 3d -> 6

    n_[0] = (pbc_) ? PbcI(s - l_ * l_) : I(s - l_ * l_); // ingoing	x coo
    n_[1] = (pbc_) ? PbcI(s + l_ * l_) : I(s + l_ * l_); // outgoing  	x coo
    n_[2] = (pbc_) ? PbcV(s - l_, s) : V(s - l_, s);     // up		y coo
    n_[3] = (pbc_) ? PbcV(s + l_, s) : V(s + l_, s);     // down 		y coo
    n_[4] = (pbc_) ? PbcH(s - 1, s) : H(s - 1, s);       // sx		z coo
    n_[5] = (pbc_) ? PbcH(s + 1, s) : H(s + 1, s);       // dx 		z coo

    return n_; // vec contiene gli indici lineari delle posizioni acessibili, e
               // -1 per quelle inaccessibili (fuori dalla box)
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
                const std::vector<int> &IsospinIndexAll,
                std::vector<std::vector<int>> &flipsh,
                std::vector<std::complex<double>> &mel, int Nup) {

    mel.resize(connectedcomponents_);
    flipsh.resize(connectedcomponents_);
    double rab_special = std::sqrt(3.0 / 16.0) *
                         dx_; // distanza minima tra due particles nel reticolo
    // assigning pre-computed matrix elements
    mel = mel_;
    flipsh = flipsh_;
    nn_.resize(Mag_);

    // Vn-e interaction dependent on distance e-n (nel nucleo non c'è)

    for (int i = 0; i < Mag_; i++) {
      // repulsione coulombiana
      for (int j = i + 1; j < Mag_; j++) {
        double rab = Distance(SpinIndexAll[j], SpinIndexAll[i]) *
                     dx_; // traplenght=1. -> dx=1/l_ -> rab= distance*dx
        if (rab == 0) {
          mel[0] += alpha_ee / rab_special;
        } else {
          mel[0] += alpha_ee / rab;
        }
      }
    }

    for (int i = 0; i < Mag_; i++) {
      nn_[i] =
          FindNeighbours(SpinIndexAll[i]); // find Neighbours of the particle
      if (i < Nup) {
        for (int j = 0; j < nn_[i].size(); j++) {
          for (int k = 0; k < Nup; k++) {
            if (nn_[i][j] == SpinIndexAll[k]) {
              nn_[i][j] = -1;
            }
          }
        }
      } else {
        for (int j = 0; j < nn_[i].size(); j++) {
          for (int k = Nup; k < Mag_; k++) {
            if (nn_[i][j] == SpinIndexAll[k]) {
              nn_[i][j] = -1;
            }
          }
        }
      }

      std::vector<int> CooPart = CalculateCoo(
          SpinIndexAll[i]); // Calcola le coordinate della particella
      int boundaryCount = 0;

      for (int j = 0; j < CooPart.size(); j++) { // CooPart.size() == 3
        if (CooPart[j] == 0 || CooPart[j] == l_ - 1) {
          boundaryCount++;
        }
      }

      // Assegna il coefficiente in base al numero di confini toccati
      if (boundaryCount == 3) {
        mel[0] += KEconst_ * (-3); // Vertici
      } else if (boundaryCount == 2) {
        mel[0] += KEconst_ * (-2); // Spigoli
      } else if (boundaryCount == 1) {
        mel[0] += KEconst_ * (-1); // Facce
      }
    }

    int index = 1;
    for (int i = 0; i < Mag_; i++) {
      for (int k = 0; k < 6; k++) { // 6= nn_[i].size()
        if (i < Nup) {
          if (nn_[i][k] >= 0) {
            flipsh[index] = std::vector<int>{SpinIndexAll[i], nn_[i][k]};
            /*std::cout<<stateIndex[i]<<"  "<<vicino<<std::endl;
            std::cout<<nn_[i][k]<<std::endl;*/
            index++;
          } else {
            mel[index] = 0.;
            flipsh[index] = std::vector<int>();
            index++;
          }
        } else {
          if (nn_[i][k] >= 0) {
            flipsh[index] = std::vector<int>{SpinIndexAll[i] + nspins_,
                                             nn_[i][k] + nspins_};
            /*std::cout<<stateIndex[i]<<"  "<<vicino<<std::endl;
            std::cout<<nn_[i][k]<<std::endl;*/
            index++;
          } else {
            mel[index] = 0.;
            flipsh[index] = std::vector<int>();
            index++;
          }
        }
      }
    }
  }
};

//////////////////////cestino
/*    flipsh.resize(connectedcomponents_);

    //assigning pre-computed matrix elements
    mel=mel_;
        flipsh=flipsh_;

        for(int i=0;i<Mag_;i++){ //particella
                for(int j=0;j<2;j++){ //direzione
                        flipsh[2*i+j+1]=std::vector<int>({stateIndex[i],stateIndex[i]+2*j-1});
                }
        }

        //implementing boundary conditions by turning mel to 0 and emptying
   flipsh std::vector<int> partPosD; partPosD.resize(connectedcomponents_-1);
        std::vector<int> boundCond;
        boundCond.resize(connectedcomponents_-1);

        for(int i=0;i<Mag_;i++){ //particella
                for(int j=0;j<2;j++){ //direzione
                        partPosD[2*i+j+1]= stateIndex[i];

                        int posPartVicina = i+2*j-1;

                        if(posPartVicina==-1){
                                boundCond[2*i+j+1]=-1;
                        }
                        else if(posPartVicina==Mag_){
                                boundCond[2*i+j+1]=nspins_;
                        }
                        else{
                                boundCond[2*i+j+1]= stateIndex[posPartVicina];
                        }

                        if(std::abs( partPosD[2*i+j+1]-boundCond[2*i+j+1] ) == 1
   ){ mel[2*i+j+1] = 0; flipsh[2*i+j+1].resize(0);
                        }
                }
    }*/

///////////////////////////////////////
