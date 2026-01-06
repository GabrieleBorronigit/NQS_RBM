#include "eigen3/Eigen/Eigenvalues"
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <random>
// #define MKL_Complex16 std::complex<double>
#include <stdio.h>
#include <stdlib.h>
/*#include "mkl_lapacke.h"
#include "mkl.h"
#include "mkl_types.h"*/
using namespace std;
using Eigen::MatrixXcd;

class SR {

  // decay constant
  double eta_;

  int npar_;

  double l2reg_;

  double dropout_p_;

  double momentum_;

  std::mt19937 rgen_;

  double decay_factor_;

  int niter_;

  double eta1_;

  bool etaDecay_;

public:
  SR(double eta, double eta1, bool etaDecay = true, double l2reg = 0,
     double momentum = 0, double dropout_p = 0)
      : eta_(eta), eta1_(eta1), etaDecay_(etaDecay), l2reg_(l2reg),
        dropout_p_(dropout_p), momentum_(momentum) {
    npar_ = -1;
    decay_factor_ = 1;
    niter_ = 0;

    // Eigen::setNbThreads(6);
  }

  void SetNpar(int npar) { npar_ = npar; }

  void Update(std::vector<std::complex<double>> &grad,
              std::vector<std::vector<std::complex<double>>> &QFisher0,
              std::vector<std::complex<double>> &pars) {
    assert(npar_ > 0);

    MatrixXcd QFisher(npar_, npar_);
    for (int i = 0; i < npar_; i++) // cols
      QFisher.col(i) =
          Eigen::Map<Eigen::VectorXcd>(QFisher0[i].data(), npar_); // rows
    MatrixXcd pinv = QFisher.completeOrthogonalDecomposition().pseudoInverse();

    Eigen::VectorXcd eigengrad =
        Eigen::Map<Eigen::VectorXcd>(grad.data(), npar_);

    Eigen::VectorXcd SRgrad = pinv * eigengrad;

    double eta;
    if (etaDecay_ == true) {
      eta = eta_ * (1 / std::sqrt(1 + niter_ / eta1_));
    }

    std::uniform_real_distribution<double> distribution(0, 1); // cmq non la uso
    for (int i = 0; i < npar_; i++) {
      if (distribution(rgen_) > dropout_p_) {
        pars[i] = (1. - momentum_) * pars[i] -
                  (SRgrad(i) + l2reg_ * pars[i]) * eta; //*gradnorm
      }
    }
    // for(int i=10;i<npar_;i++){
    //   if(distribution(rgen_)>dropout_p_){
    //     pars[i]=(1.-momentum_)*pars[i] - (SRgrad(i)+l2reg_*pars[i]) * eta_
    //     * 2.5; //*gradnorm
    //   }
    // }

    // if(niter_==25){
    //   eta_=eta_*0.7;
    // }
    niter_ += 1;
  }

  void Update2(std::vector<std::complex<double>> &grad, MatrixXcd &QFisher0,
               std::vector<std::complex<double>> &pars) {
    assert(npar_ > 0);

    MatrixXcd pinv = QFisher0.completeOrthogonalDecomposition().pseudoInverse();

    Eigen::VectorXcd eigengrad =
        Eigen::Map<Eigen::VectorXcd>(grad.data(), npar_);

    Eigen::VectorXcd SRgrad = pinv * eigengrad;

    std::uniform_real_distribution<double> distribution(0, 1); // cmq non la uso
    for (int i = 0; i < npar_; i++) {
      if (distribution(rgen_) > dropout_p_) {
        pars[i] = (1. - momentum_) * pars[i] -
                  (SRgrad(i) + l2reg_ * pars[i]) * eta_; //*gradnorm
      }
    }
    // for(int i=10;i<npar_;i++){
    //   if(distribution(rgen_)>dropout_p_){
    //     pars[i]=(1.-momentum_)*pars[i] - (SRgrad(i)+l2reg_*pars[i]) * eta_
    //     * 2.5; //*gradnorm
    //   }
    // }

    // if(niter_==50){
    //   eta_=eta_/100;
    // }

    if (etaDecay_ == true) {
      eta_ = eta_ * (1 / std::sqrt(1 + niter_ / eta1_));
    }

    niter_ += 1;
  }

  /*  void Update3(std::vector<std::complex<double>> & grad,
                  std::complex<double>* QFisher0,
                  std::vector<std::complex<double>> & pars){
      assert(npar_>0);


      int m = npar_, n = npar_, lda = npar_, ldu = npar_, ldvt = npar_, info;
      double s[m];  //il minore fra M e N
      complex<double> u[ldu*m], vt[ldvt*n];

      // Compute SVD
      info = LAPACKE_zgesdd( LAPACK_ROW_MAJOR, 'A', m, n, QFisher0, lda, s, u,
       ldu, vt, ldvt );
      // Check for convergence
      if( info > 0 ) {
              printf( "The algorithm computing SVD failed to converge.\n" );
              exit( 1 );
      }

      for (int i = 0; i < n; i++) {
        double ss;
        if(s[i] > 1.0e-9)
          ss=1.0/s[i];
        else
          ss=s[i];
        cblas_zdscal(n,ss,&u[i],n);
      }

      //for (int i = 0; i < n; i++) {
      //  cblas_zdscal(n,s[i],&u[i],n);
      //}

      complex<double>  pinv[n*m];
      complex<double> alpha[1] = {{1.,0.}};
      complex<double> beta[1] = {{0.,0.}};

      cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasConjTrans, n, n, m, alpha,
    vt, n, u, m, beta, pinv, m);

      complex<double>* SRgrad = new complex<double>[npar_*npar_];
      std::copy(grad.begin(), grad.end(), SRgrad);
      //double* SRgrad = &grad[0];

      cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, n, 1, alpha, pinv,
    n, SRgrad, 1, beta, SRgrad, 1);

      double eta;
      if(etaDecay_==true){
        eta=eta_*(1/std::sqrt(1+niter_/eta1_));
      }

      std::uniform_real_distribution<double> distribution(0,1); //cmq non la uso
      for(int i=0;i<npar_;i++){
        if(distribution(rgen_)>dropout_p_){
          pars[i]=(1.-momentum_)*pars[i] - (SRgrad[i]+l2reg_*pars[i]) * eta; //
    *gradnorm
        }
      }
      //for(int i=10;i<npar_;i++){
      //  if(distribution(rgen_)>dropout_p_){
      //    pars[i]=(1.-momentum_)*pars[i] - (SRgrad(i)+l2reg_*pars[i]) * eta_
    * 2.5; //   *gradnorm
      //  }
      //}

      //if(niter_==25){
      //  eta_=eta_*0.7;
      //}
      niter_+=1;
      delete[] SRgrad;
    }*/

  // void SetDecayFactor(double decay_factor){
  //   assert(decay_factor<1);
  //   decay_factor_=decay_factor;
  // }

  // void Reset(){
  // }
  double GetEta() { return eta_; }
  int GetNiter() { return niter_; }
};
