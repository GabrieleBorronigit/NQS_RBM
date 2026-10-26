#include <iostream>
// #include <Eigen/Dense>
// #include <Eigen/IterativeLinearSolvers>
#include <Eigen/Eigenvalues>
#include "mpi.h"
#include <complex>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

using namespace std;
using namespace Eigen;

template <class Hamiltonian, class Nqs, class Sampler, class Optimizer>
class Variational {

  Hamiltonian &ham_;
  Nqs &nqs_;
  Sampler &sampler_;
  Optimizer opt_;

  int npar_;
  vector<vector<int>> configSamp_;
  vector<vector<int>> indexSamp_;
  vector<vector<complex<double>>> derLogs_;
  vector<double> elocs_;
  double elocmean_;
  double sigmaElocmean_;
  double loss_;
  int nsweeps_;
  int sweepfactor_;
  double mean_radius_;
  vector<complex<double>> grad_;
  vector<vector<complex<double>>> QFisher0_;
  // MatrixXcd QFisher0_;
  // complex<double>* QFisher0_; per lapack-> costruttore, distruttore,
  // calcQFisher e UpdateParametersSR e include di SR

  ofstream out_;

  int Iter0_;

public:
  Variational(Sampler &sampler, Optimizer &opt)
      : ham_(sampler.GetHamiltonian()), nqs_(sampler.GetNqs()),
        sampler_(sampler), opt_(opt) {

    npar_ = nqs_.GetNpar();

    grad_.resize(npar_);
    opt_.SetNpar(npar_);
    Iter0_ = 0;
    sweepfactor_ = 1;
    mean_radius_ = 0.0;
    loss_ = 0.0;
    // QFisher0_ = new complex<double>[npar_*npar_];
  }

  ~Variational() {
    // delete[] QFisher0_;
    out_.close();
  }

  // Campionamento per il TRAINING (SGD): il sampler accumula on-the-fly
  // sum_k O_k e sum_k E_k O_k, cosi' NON teniamo in memoria l'intera matrice
  // nsweeps x npar ne' copiamo history inutili. Ricava solo elocs_/elocmean_.
  void Sample() {
    sampler_.SetStoreFull(false);
    sampler_.Sample(nsweeps_, 0.6, sweepfactor_);
    elocs_.resize(nsweeps_);
    realPart(sampler_.GetSamplingEnergies(), elocs_);
    elocmean_ =
        accumulate(elocs_.begin(), elocs_.end(), 0.0) / elocs_.size();
  }

  // Campionamento COMPLETO (pretraining e SR): serve la matrice per-campione
  // (derLogs_) e le history delle configurazioni. Comportamento originale.
  void SampleFull() {
    sampler_.SetStoreFull(true);
    sampler_.Sample(nsweeps_, 0.6, sweepfactor_);
    configSamp_ = sampler_.GetSamplingConfig();
    indexSamp_ = sampler_.GetSamplingIndex();
    elocs_.resize(nsweeps_);
    realPart(sampler_.GetSamplingEnergies(), elocs_);
    elocmean_ =
        accumulate(elocs_.begin(), elocs_.end(), 0.0) / elocs_.size();
    derLogs_ = sampler_.GetDerLogs();
  }

  void CalcSigmaElocmean(int bins = 50) {
    int blocksize = floor(double(elocs_.size()) / double(bins));
    double enmean = 0;
    double enmeansq = 0;
    for (int i = 0; i < bins; i++) {
      double eblock = 0;
      for (int j = i * blocksize; j < (i + 1) * blocksize; j++) {
        eblock += elocs_[j];
      }
      eblock /= double(blocksize);
      double delta = eblock - enmean;
      enmean += delta / double(i + 1);
      double delta2 = eblock - enmean;
      enmeansq += delta * delta2;
    }
    enmeansq /= (double(bins - 1)); // diviso n-1 ?
    // enmean; //estimated average
    sigmaElocmean_ = sqrt(enmeansq / double(bins)); // estimated error
  }

  // void CalcDerLogs(){
  //   derLogs_.resize(nsweeps_,vector<complex<double>> (npar_));
  //   for(int i=0;i<nsweeps_;i++){
  //     derLogs_[i] = nqs_.DerLog(configSamp_[i]); //sulla riga cresce il
  //     numero di sweep, i sale sulla y
  //   }
  // }

  // THIS ONE USES C++ VECTORS AND EIGEN
  void CalcQuantumFisher() {
    QFisher0_.resize(npar_, vector<complex<double>>(npar_));
    vector<complex<double>> meanDerlog(npar_, {0., 0.});
    for (int i = 0; i < npar_; i++) {
      for (int k = 0; k < nsweeps_; k++) {
        meanDerlog[i] += derLogs_[k][i] / double(nsweeps_);
      }
    }
    for (int i = 0; i < npar_; i++) {
      for (int j = 0; j < npar_; j++) {
        QFisher0_[i][j] = -conj(meanDerlog[j]) * meanDerlog[i];
        for (int k = 0; k < nsweeps_; k++) {
          QFisher0_[i][j] +=
              conj(derLogs_[k][j]) * derLogs_[k][i] / double(nsweeps_);
        }
      }
    }
  }

  /*  THIS ONE USES ONLY EIGEN AND PAIRS WITH SR.UPGRADE2
  void CalcQuantumFisher(){
      QFisher0_.resize(npar_,npar_);
      vector<complex<double>> meanDerlog(npar_,(0.,0.));
      for(int i=0;i<npar_;i++){
        for(int k=0;k<nsweeps_;k++){
          meanDerlog[i] += derLogs_[k][i] / double(nsweeps_);
        }
      }
      for(int i=0;i<npar_;i++){
        for(int j=0;j<npar_;j++){
          QFisher0_(i,j)= - conj(meanDerlog[i]) * meanDerlog[j];
          for(int k=0;k<nsweeps_;k++){
            QFisher0_(i,j) += conj(derLogs_[k][i])*derLogs_[k][j] /
  double(nsweeps_);
          }
        }
      }
    } */

  // THIS ONE USES MKL AND LAPACKE, THUS POINTERS OF COMPLEX ROW-MAJOR, AND
  // PAIRS WITH SR.UPGRADE3
  /*  void CalcQuantumFisher(){
      vector<complex<double>> meanDerlog(npar_,(0.,0.));
      for(int i=0;i<npar_;i++){
        for(int k=0;k<nsweeps_;k++){
          meanDerlog[i] += derLogs_[k][i] / double(nsweeps_);
        }
      }
      for(int i=0;i<npar_;i++){
        for(int j=0;j<npar_;j++){
          QFisher0_[i*npar_+j] = - conj(meanDerlog[i]) * meanDerlog[j];
          for(int k=0;k<nsweeps_;k++){
            QFisher0_[i*npar_+j] += conj(derLogs_[k][i])*derLogs_[k][j] /
    double(nsweeps_);
          }
        }
      }
    } */

  inline complex<double> PhiTarget(vector<int> stateIndex) { // 2p1d GS
    int nv = nqs_.Nspins();
    if (stateIndex.size() == 1) { // una particella: GS della buca
      return sin(complex<double>(M_PI * (stateIndex[0] + 0.5) / nv, 0));
    }
    // due fermioni identici: determinante di Slater dei modi n=1 e n=2
    // (prima i due termini erano identici e phi risultava sempre 0)
    complex<double> phi =
        1 / sqrt(2) *
        (sin(complex<double>(M_PI * (stateIndex[0] + 0.5) / nv, 0)) *
             sin(complex<double>(2 * M_PI * (stateIndex[1] + 0.5) / nv, 0)) -
         sin(complex<double>(2 * M_PI * (stateIndex[0] + 0.5) / nv, 0)) *
             sin(complex<double>(M_PI * (stateIndex[1] + 0.5) / nv, 0)));
    return phi;
  }

  void CalcExactLossFunction() { // exact loss function for 1particle
    int nv;
    nv = nqs_.Nspins();
    for (int i = 0; i < nv; i++) {
      vector<int> state(nv, 0);
      state[i] = 1;
      vector<int> stateIndex(1, i);
      loss_ += norm(exp(nqs_.LogVal(state)) - PhiTarget(stateIndex));
    }
  }

  void PretrainingGradient() {
    // CalcDerLogs();
    for (int i = 0; i < npar_; i++)
      grad_[i] = 0.;

    for (int i = 0; i < nsweeps_; i++) {
      complex<double> targetDistance =
          1. - PhiTarget(indexSamp_[i]) / exp(nqs_.LogVal(configSamp_[i]));
      for (int j = 0; j < npar_; j++) {
        grad_[j] +=
            targetDistance * conj(derLogs_[i][j]) * 2. / double(nsweeps_);
      }
    }
  }

  // Gradiente stocastico dagli accumulatori del sampler. Identico a
  //   grad[j] = (2/N) sum_i (E_i - <E>) conj(O_i[j])
  // riscritto come  (2/N) ( sumEDer[j] - <E> conj(sumDer[j]) ),  dove
  // sumEDer accumula E_k*conj(O_k) con E_k COMPLESSA,
  // con E_i reali: matematicamente equivalente, ma senza tenere in RAM la
  // matrice nsweeps x npar.
  void TrainingGradient() {
    const long N = sampler_.GetNAccum();
    const auto &sumDer = sampler_.GetSumDer();
    const auto &sumEDer = sampler_.GetSumEDer();
    if (N <= 0 || (int)sumDer.size() != npar_) {
      std::cerr << "TrainingGradient: accumulatori vuoti/incoerenti (N=" << N
                << ", size=" << sumDer.size() << ")" << std::endl;
      for (int j = 0; j < npar_; j++)
        grad_[j] = 0.;
      return;
    }
    const double invN = 1.0 / double(N);
    for (int j = 0; j < npar_; j++) {
      if (std::isnan(sumEDer[j].real()) || std::isnan(sumDer[j].real())) {
        std::cerr << "NAN in accumulatori gradiente! par " << j << std::endl;
      }
      grad_[j] =
          2.0 * invN * (sumEDer[j] - elocmean_ * conj(sumDer[j]));
    }
  }

  void RunPretraining(int nsweeps, int niter) {
    nsweeps_ = nsweeps;
    SetOutputFilePretraining();
    for (int i = 0; i < niter; i++) {
      SampleFull(); // pretraining: serve la matrice per-campione

      PretrainingGradient();

      CalcSigmaElocmean();
      CalcExactLossFunction();
      PrintStats(i);

      UpdateParameters();
      for (const auto &p : nqs_.GetParameters()) {
        if (std::isnan(p.real()) || std::isnan(p.imag())) {
          std::cerr << "❌ NaN in parameters after update!" << std::endl;
          std::exit(1);
        }
      }
    }
    PrintStats(Iter0_ + niter);
    Iter0_ += niter;
  }

  void RunPretrainingSR(int nsweeps, int niter) {
    nsweeps_ = nsweeps;
    SetOutputFilePretraining();
    for (int i = 0; i < niter; i++) {
      SampleFull(); // pretraining+SR: serve la matrice per-campione

      PretrainingGradient();
      CalcQuantumFisher();

      CalcSigmaElocmean();
      CalcExactLossFunction();
      PrintStats(i);

      UpdateParametersSR();
    }
    PrintStats(Iter0_ + niter);
    Iter0_ += niter;
  }

  /*void RunTraining(int nsweeps,int niter){
    nsweeps_=nsweeps;
    SetOutputFileTraining("TrainingTest/Training.dat");
    for(int i=0;i<niter;i++){
      Sample();

      TrainingGradient();

      CalcSigmaElocmean();
      CalcExactLossFunction();
      PrintStats(i);

      UpdateParameters();
    }
    PrintStats(Iter0_+niter);
    Iter0_+=niter;
  }*/
  void RunTraining(int nsweeps, int niter, int loadtype = 0, int nprint = 2000,
                   std::string label = "H2",
                   std::string trainingfn = "TrainingH2.dat") {

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    nsweeps_ = nsweeps; // salva numero di sweeps
    // increase MC attempts per sweep: sweepfactor multiplies nspins
    sweepfactor_ = 4; // increase this if you want more moves per MC step
    SetOutputFileTraining(trainingfn); // predispone il file su cui stampare
                                       // iterazione corrente, Eloc, sigma, loss
    if (rank == 0) {
      sampler_.SetFileStates(
          "Configuration.dat"); // predispone il file su cui stampare la
                                // configurazione corrente
    }
    if (loadtype == 2 || loadtype == 3) // se ricarico i pesi, ricarico anche lr
    {
      opt_.SetEta(
          nqs_.GetEtaAfterRestart()); // carica learning rate e progresso nel
                                      // decadimento dalla run precedente
      opt_.SetNiter(nqs_.GetNiterAfterRestart());
      // equilibrate
      /*
      std::cout<<"equilibrate before starting training"<<std::endl;
      for(int i=0; i<nqs_.GetNiterAfterRestart(); i++){
        Sample();
      }*/
    }
    // solo il master apre/scrive: piu' stream aperti in troncamento sullo
    // stesso file da rank diversi si azzerano a vicenda
    std::ofstream outeta, outacc;
    if (rank == 0) {
      std::filesystem::create_directories("SamplingTest");
      outeta.open("SamplingTest/LR" + label + ".dat");
      outacc.open("SamplingTest/Acceptance" + label + ".dat");
      outacc << "i" << setw(12) << "acc" << std::endl;
    }

    for (int i = 0; i < niter; i++) { // ciclo sul numero di iterazioni

      Sample(); // chiama il sample da sampler, che campiona con metropolis, poi
                // salva le derlogs, le energie, e calcola l'energia media
      // per il caso nucleare qua non dovrebbe cambiare nulla, devo solo
      // aggiornare l'update delle lt/energie in nqs
      //  1. Calcola la somma delle energie usando MPI_Allreduce

      double sum_energies = 0.0;
      MPI_Allreduce(&elocmean_, &sum_energies, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);

      // 2. Calcola la media dividendo per il numero di nodi
      double avg_energy = sum_energies / size;

      double sq_diff_sum = 0.0;
      double local_sq_diff =
          (elocmean_ - avg_energy) * (elocmean_ - avg_energy);
      MPI_Allreduce(&local_sq_diff, &sq_diff_sum, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);

      double std_dev = std::sqrt(sq_diff_sum / size);
      double std_err = std_dev / std::sqrt(size);

      // 3. Broadcast the averaged energy to all nodes
      MPI_Bcast(&avg_energy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&std_err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      // 4. Update the local energy with the averaged energy
      elocmean_ = avg_energy;
      sigmaElocmean_ = std_err;

      TrainingGradient();

      // 1. Calcola la somma dei gradienti usando MPI_Allreduce
      std::vector<std::complex<double>> sum_gradients(grad_.size());
      MPI_Allreduce(grad_.data(), sum_gradients.data(), grad_.size(),
                    MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);

      // 2. Calcola la media dividendo per il numero di nodi
      std::vector<std::complex<double>> avg_gradients(grad_.size());
      for (int j = 0; j < grad_.size(); ++j) {
        avg_gradients[j] =
            sum_gradients[j] /
            static_cast<double>(size); // qua sommava sum_gradients[j].real() e
                                       // non capivo il senso, ho tolto il .real
      }

      // 3. Broadcast the averaged gradients to all nodes
      MPI_Bcast(avg_gradients.data(), avg_gradients.size(), MPI_DOUBLE_COMPLEX,
                0, MPI_COMM_WORLD);

      // 4. Update parameters using the averaged gradients
      for (int j = 0; j < grad_.size(); ++j) {
        grad_[j] = avg_gradients[j];
      }

      // CalcSigmaElocmean();
      // CalcExactLossFunction();
      PrintStats(i); // stampa su file Eloc media, sigmamediaEloc, loss, bias e
                     // pesi della rbm, e numero dell'iterazione corrente (i)

      UpdateParameters(); // aggiorna i pesi della rbm
      MPI_Barrier(MPI_COMM_WORLD);

      if (((i + 1) % 50) == 0) {
        CheckParametersConsistency();
        // sampler_.InitRandomState();
        // nqs_.InitLt(sampler_.GetState());
      }
      if ((i + 1) % nprint == 0) {
        SetupReset(i, label);
      }
      if (rank == 0) {
        outacc << i << setw(12) << sampler_.Acceptance() << std::endl;
        outeta << i << setw(12) << opt_.GetEta() << std::endl;
      }
    }
    outacc.close();
    outeta.close();
    Iter0_ += niter;
  }

  void CheckParametersConsistency() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto pars = nqs_.GetParameters();

    // derlogs corrette gli danno un po piu di vita, ora si blocca dopo 120 iter

    // Tutti i processi inviano i loro vettori 'pars' al processo 0
    if (rank != 0) {
      MPI_Send(pars.data(), pars.size(), MPI_DOUBLE_COMPLEX, 0, 0,
               MPI_COMM_WORLD);
    } else {
      // Il processo 0 riceve i vettori dagli altri processi e li confronta con
      // il suo
      std::vector<std::complex<double>> received_pars(pars.size());
      double epsilon = 0;
      for (int i = 1; i < size;
           ++i) { // riceve un vettore da ogni noto e confronta
        MPI_Recv(received_pars.data(), pars.size(), MPI_DOUBLE_COMPLEX, i, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int j = 0; j < pars.size(); ++j) {
          if (pars[j] != received_pars[j]) {
            std::cerr << "Errore: i parametri del nodo " << i
                      << " non sono consistenti con il master" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
          }
        }
      }
      std::cout << "i parametri dei nodi sono consistenti" << std::endl;
    }
  }

  /*void RunTrainingSR(int nsweeps,int niter, std::string outputfile){
    nsweeps_=nsweeps;
    SetOutputFileTraining(outputfile);
    for(int i=0;i<niter;i++){
      Sample();

      TrainingGradient();
      CalcQuantumFisher();

      CalcSigmaElocmean();
      CalcExactLossFunction();			//calcola distanza del mio stato
  dallo stato target PrintStats(i);

      UpdateParametersSR();
    }
    PrintStats(Iter0_+niter);
    Iter0_+=niter;
  }*/

  void RunTrainingSR(int nsweeps, int niter, int loadtype = 0,
                     int nprint = 3000, std::string label = "H2",
                     std::string trainingfn = "TrainingH2.dat") {

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    nsweeps_ = nsweeps;
    SetOutputFileTraining(trainingfn);

    if (rank == 0) {
      sampler_.SetFileStates("configuration_SR.dat");
    }

    // solo il master apre/scrive (vedi RunTraining)
    std::ofstream outeta, outacc;
    if (rank == 0) {
      std::filesystem::create_directories("SamplingTest");
      outeta.open("SamplingTest/LR" + label + ".dat");
      outacc.open("SamplingTest/Acceptance" + label + ".dat");
      outacc << "i" << setw(12) << "acc" << std::endl;
    }

    for (int i = 0; i < niter; i++) {

      SampleFull(); // SR: CalcQuantumFisher ha bisogno delle derLogs complete

      // 1. Calcola la somma delle energie usando MPI_Allreduce
      double sum_energies = 0.0;
      MPI_Allreduce(&elocmean_, &sum_energies, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);

      // 2. Calcola la media dividendo per il numero di nodi
      double avg_energy = sum_energies / size;

      double sq_diff_sum = 0.0;
      double local_sq_diff =
          (elocmean_ - avg_energy) * (elocmean_ - avg_energy);
      MPI_Allreduce(&local_sq_diff, &sq_diff_sum, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);

      double std_dev = std::sqrt(sq_diff_sum / size);
      double std_err = std_dev / std::sqrt(size);

      // 3. Broadcast the averaged energy to all nodes
      MPI_Bcast(&avg_energy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&std_err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      // 4. Update the local energy with the averaged energy
      elocmean_ = avg_energy;
      sigmaElocmean_ = std_err;
      // --- gather mean radius from sampler (MPI reduction) ---

      TrainingGradient(); // calcola i gradienti stocastici lungo ogni parametro

      CalcQuantumFisher(); // matrice di fisher

      // Gather all local Fisher matrices on the master node (rank 0)
      std::vector<std::complex<double>> local_fisher_flat(npar_ * npar_);
      for (int i = 0; i < npar_; ++i) {
        for (int j = 0; j < npar_; ++j) {
          local_fisher_flat[i * npar_ + j] = QFisher0_[i][j];
        }
      }

      std::vector<std::complex<double>> gathered_fisher(npar_ * npar_ * size);
      MPI_Gather(local_fisher_flat.data(), npar_ * npar_, MPI_DOUBLE_COMPLEX,
                 gathered_fisher.data(), npar_ * npar_, MPI_DOUBLE_COMPLEX, 0,
                 MPI_COMM_WORLD);

      // Calculate the average Fisher matrix on the master node
      std::vector<std::complex<double>> avg_fisher_flat(npar_ * npar_);
      if (rank == 0) {
        for (int i = 0; i < npar_ * npar_; ++i) {
          std::complex<double> sum = 0.0;
          for (int j = 0; j < size; ++j) {
            sum += gathered_fisher[j * npar_ * npar_ + i];
          }
          avg_fisher_flat[i] = sum / std::complex<double>(size);
        }
      }

      // Broadcast the average Fisher matrix to all nodes
      MPI_Bcast(avg_fisher_flat.data(), npar_ * npar_, MPI_DOUBLE_COMPLEX, 0,
                MPI_COMM_WORLD);

      // Update the local Fisher matrix with the average
      for (int i = 0; i < npar_; ++i) {
        for (int j = 0; j < npar_; ++j) {
          QFisher0_[i][j] = avg_fisher_flat[i * npar_ + j];
        }
      }

      // 1. Calcola la somma dei gradienti usando MPI_Allreduce
      std::vector<std::complex<double>> sum_gradients(grad_.size());
      MPI_Allreduce(grad_.data(), sum_gradients.data(), grad_.size(),
                    MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);

      // 2. Calcola la media dividendo per il numero di nodi
      std::vector<std::complex<double>> avg_gradients(grad_.size());
      for (int j = 0; j < grad_.size(); ++j) {
        avg_gradients[j] =
            sum_gradients[j] /
            static_cast<double>(size); // anche qua ho tolto il .real()
      }

      // 3. Broadcast the averaged gradients to all nodes
      MPI_Bcast(avg_gradients.data(), avg_gradients.size(), MPI_DOUBLE_COMPLEX,
                0, MPI_COMM_WORLD);

      // 4. Update parameters using the averaged gradients
      for (int j = 0; j < grad_.size(); ++j) {
        grad_[j] = avg_gradients[j];
      }
      PrintStats(i);
      UpdateParametersSR();
      MPI_Barrier(MPI_COMM_WORLD);
      if (((i + 1) % 50) == 0) {
        CheckParametersConsistency();
      }
      if ((i + 1) % nprint == 0) { // prima era if(i % nprint): salvava il
                                   // checkpoint a ogni iter TRANNE i multipli
        SetupReset(i, label);
      }
      if (rank == 0) {
        outacc << i << setw(12) << sampler_.Acceptance() << std::endl;
        outeta << i << setw(12) << opt_.GetEta() << std::endl;
      }
    }
    outacc.close();
    outeta.close();
    Iter0_ += niter;
  }

  void UpdateParameters() {
    auto pars = nqs_.GetParameters();
    opt_.Update(grad_, pars); // usa lo stesso grad per training e pretraining
    nqs_.SetParameters(pars);
  }

  void UpdateParametersSR() {
    auto pars = nqs_.GetParameters();
    opt_.Update(
        grad_, QFisher0_,
        pars); // no number per vector+eigen, 1 per only eigen, 2 per lapack
    nqs_.SetParameters(pars);
  }

  void SetOutputFilePretraining() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) // un solo stream sul file: altri rank lo troncherebbero
      return;
    std::filesystem::create_directories("TrainingTest");
    out_.open("TrainingTest/Pretraining.dat", std::ios_base::out);
    if (!out_.is_open()) {
      std::cerr << "# Error : Cannot open file for writing optimization data"
                << std::endl;
      std::abort();
    }
    out_ << "Niter" << "\t" << "Eloc" << "\t" << "sigma" << "\t" << "loss"
         << "\t";
    for (int i = 0; i < npar_; i++)
      out_ << "par" << i << "\t";
    out_ << endl;

    std::cout << "# Saving sampled configuration to file "
              << "TrainingTest/Pretraining.dat" << std::endl;
  }

  void SetOutputFileTraining(
      std::string outputfile) { // predispone il file su cui stampare iterazione
                                // corrente, Eloc, sigma, loss
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) // un solo stream sul file: altri rank lo troncherebbero
      return;
    out_.open(outputfile.c_str(), std::ios_base::out);
    if (!out_.is_open()) {
      std::cerr << "# Error : Cannot open file for writing optimization data"
                << std::endl;
      std::abort();
    }
    out_ << "Niter" << "\t" << "Eloc" << "\t" << "sigma" << "\t" << "loss"
         << "\t" << "mean_radius" << endl;

    std::cout << "# Saving sampled configuration to file " << outputfile.c_str()
              << std::endl;
  }

  void
  PrintStats(int i) { // stampa su file Eloc media, sigmamediaEloc, loss, bias e
                      // pesi della rbm, e numero dell'iterazione corrente (i)
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
      out_ << i + Iter0_ << "\t" << scientific << elocmean_ << "\t"
           << sigmaElocmean_ << "\t" << mean_radius_
           << "\t"; // << loss_ << "\t";
      auto pars = nqs_.GetParameters();
      /*  for (const auto &par : pars) {
          out_ << "(" << par.real();
          if (par.imag() >= 0.)
            out_ << "+";
          else
            out_ << "-";
          out_ << abs(par.imag()) << "j)" << "\t";
        }*/
      out_ << endl;

      // std::cout<<i+Iter0_<<"  Elocmean="<<scientific<<elocmean_<<"gradnorm=
      // "<<grad_.norm()<<"
      // maxparam="<<rbm_.GetParameters().array().abs().maxCoeff()<<"
      // "std::endl;
      std::cout << i << "th iteration: done writing optimization data"
                << std::endl;
    }
  }

  // prepara un file di reset dopo N iterazioni: stampa nv, nh, eta, niter e
  // tutti i parametri della rbm
  void SetupReset(int N, std::string label) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // solo il master scrive il checkpoint: se anche gli altri rank aprissero
    // lo stesso file in troncamento potrebbero azzerarlo dopo la scrittura
    if (rank != 0)
      return;
    std::filesystem::create_directories("TrainingTest");
    std::ofstream out("TrainingTest/weights" + label + ".dat");
    if (!out.is_open()) {
      std::cerr << "# Error : Cannot open checkpoint file TrainingTest/weights"
                << label << ".dat" << std::endl;
      return;
    }
    {

      out << nqs_.Nspins() << std::endl;  // nv
      out << nqs_.NHspins() << std::endl; // nh
      out << opt_.GetEta() << std::endl;  // eta
      out << opt_.GetNiter()
          << std::endl; // niter, per continuare il decadimento del learning
                        // rate in maniera coerente
      auto pars = nqs_.GetParameters(); // print parameters
      std::cout << "nparametri: " << nqs_.GetNpar() << std::endl;

      for (const auto &par : pars) {
        out << "(" << par.real() << ",";
        if (par.imag() >= 0.)
          out << "+";
        else
          out << "-";
        out << abs(par.imag()) << ")" << std::endl;
      }
    }
    out.close();
  }
  // Debug function to check that the logarithm of the derivative is
  // computed correctly
  /*  void CheckDerLog(double eps=1.0e-4){
      sampler_.Reset(true);

      auto ders=rbm_.DerLog(sampler_.Visible());

      auto pars=rbm_.GetParameters();

      for(int i=0;i<npar_;i++){
        pars(i)+=eps;
        rbm_.SetParameters(pars);
        double valp=rbm_.LogVal(sampler_.Visible());

        pars(i)-=2*eps;
        rbm_.SetParameters(pars);
        double valm=rbm_.LogVal(sampler_.Visible());

        pars(i)+=eps;

        double numder=(-valm+valp)/(eps*2);

        if(std::abs(numder-ders(i))>eps*eps){
          cerr<<"Possible error on parameter "<<i<<". Expected: "<<ders(i)<<"
    Found: "<<numder<<endl;
        }
      }
    } */

  void realPart(const vector<complex<double>> &x, vector<double> &y) {
    for (int i = 0; i < x.size(); i++) {
      y[i] = x[i].real();
    }
  }
};
