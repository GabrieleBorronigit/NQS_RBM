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

#include <complex>
#include <iostream>
#include <vector>

// Anti-ferromagnetic Heisenberg model in 3d
class Heisenberg3d {

  // number of spins
  const int nspins_;

  // size of the cubic lattice
  const int l_;

  // option to use periodic boundary conditions
  const bool pbc_;

  // coupling constant
  const double jz_;

  // Nearest neighbors on the cubic lattice
  std::vector<std::vector<int>> nn_;

  std::vector<std::vector<int>> bonds_;

public:
  Heisenberg3d(int nspins, double jz, bool pbc = true)
      : nspins_(nspins), pbc_(pbc), jz_(jz), l_(std::sqrt(nspins_)) {
    InitLattice();
    std::cout << "# Using the 3d Heisenberg model with J_z = " << jz_
              << std::endl;
  }

  void InitLattice() {
    if (l_ * l_ * l_ != nspins_) {
      std::cerr << "# Error , the number of spins is not compabitle with a "
                   "cubic lattice "
                << std::endl;
      std::abort();
    }

    nn_.resize(nspins_, std::vector<int>(6)); // matrix [nspins x 6]

    // Defining the nearest-neighbors
    for (int i = 0; i < nspins_;
         i++) { // w/ PBC if pbc_=true; else we have a 3D box. (return -1 when
                // there's not a near spin)
      nn_[i][0] = (pbc_) ? PbcH(i - 1, i) : H(i - 1, i);       // sx
      nn_[i][1] = (pbc_) ? PbcH(i + 1, i) : H(i + 1, i);       // dx
      nn_[i][2] = (pbc_) ? PbcV(i - l_, i) : V(i - l_, i);     // up
      nn_[i][3] = (pbc_) ? PbcV(i + l_, i) : V(i + l_, i);     // down
      nn_[i][4] = (pbc_) ? PbcI(i - l_ * l_) : I(i - l_ * l_); // outgoing
      nn_[i][5] = (pbc_) ? PbcI(i + l_ * l_) : I(i + l_ * l_); // ingoing
    }

    for (int i = 0; i < nspins_; i++) {
      for (int k = 0; k < 6; k++) {
        int j = nn_[i][k];
        if (i < j) {
          bonds_.push_back(std::vector<int>{i, j});
        }
      }
    } // saved the interactions(counted once) between near spins.
  }

  // Finds the non-zero matrix elements of the hamiltonian
  // on the given state
  // i.e. all the state' such that <state'|H|state> = mel(state') \neq 0
  // state' is encoded as the sequence of spin flips to be performed on state
  void FindConn(const std::vector<int> &state,
                std::vector<std::vector<int>> &flipsh,
                std::vector<std::complex<double>> &mel) {

    mel.resize(1);
    flipsh.resize(1);

    // computing interaction part Sz*Sz
    mel[0] = 0.;

    for (int i = 0; i < bonds_.size(); i++) {
      mel[0] += double(state[bonds_[i][0]] * state[bonds_[i][1]]);
    }

    mel[0] *= jz_;

    // Looks for possible spin flips
    for (int i = 0; i < bonds_.size(); i++) { //???
      const int si = bonds_[i][0];
      const int sj = bonds_[i][1];

      if (state[si] != state[sj]) {
        mel.push_back(-2);
        flipsh.push_back(std::vector<int>({si, sj}));
      }
    }
  }

  int MinFlips() const { return 2; }

  // Small functions to set up the lattice
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
};
