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
class VariationalCG {

  Hamiltonian &ham_;
  Nqs &nqs_;
  Sampler &sampler_;
  Optimizer opt_;

  int npar_;
  vector<vector<int>> configSamp_;
  vector<vector<int>> indexSamp_;
  vector<vector<complex<double>>> derLogs_;
  vector<double> elocs_;
  // E_loc COMPLESSA: serve al GRADIENTE. Per parametri olomorfi il gradiente
  // e' <O* (E_loc - <E>)> e scartare Im(E_loc) toglie una componente reale.
  // elocs_ resta reale perche' e' la stima fisica riportata e ridotta via MPI.
  vector<complex<double>> elocsC_;
  // diagnostica del CG
  double cgResidual_ = 0.0, cgLambdaEff_ = 0.0, cgStepNorm_ = 0.0;
  int cgIters_ = 0, cgClipped_ = 0;
  double elocmean_;
  double sigmaElocmean_;
  double loss_;
  int nsweeps_;
  vector<complex<double>> grad_;
  vector<vector<complex<double>>> QFisher0_;
  // MatrixXcd QFisher0_;
  // complex<double>* QFisher0_; per lapack-> costruttore, distruttore,
  // calcQFisher e UpdateParametersSR e include di SR
std::vector<std::complex<double>> Omean_; //per CG-FAC
  ofstream out_;

  int Iter0_;

public:
  VariationalCG(Sampler &sampler, Optimizer &opt)
      : ham_(sampler.GetHamiltonian()), nqs_(sampler.GetNqs()),
        sampler_(sampler), opt_(opt) {

    npar_ = nqs_.GetNpar();

    grad_.resize(npar_);
    opt_.SetNpar(npar_);
    Iter0_ = 0;
    // QFisher0_ = new complex<double>[npar_*npar_];
  }

  ~VariationalCG() {
    // delete[] QFisher0_;
    out_.close();
  }

  void Sample() {
    // Il CG ha bisogno delle derivate logaritmiche PER CAMPIONE (sono il
    // fattore di rango basso della matrice di Fisher). Il sampler, per
    // risparmiare memoria, le accumula soltanto: qui vanno richieste esplicitamente.
    sampler_.SetStoreFull(true);
    sampler_.Sample(
        nsweeps_); // inizializza stato random, campiona configurazioni con
                   // metropolis e calcola energia locale+Derlogs, tutto
                   // Niter*Nsweeps volte
    configSamp_ = sampler_.GetSamplingConfig(); // statehistory
    indexSamp_ = sampler_.GetSamplingIndex();   // stateindexhistory
    elocs_.resize(nsweeps_);
    realPart(sampler_.GetSamplingEnergies(),
             elocs_); // prendo solo la parte reale di Eloc, in teoria la parte
                      // immaginaria dovrebbe mediare a zero
    elocmean_ = accumulate(elocs_.begin(), elocs_.end(), 0.0) /
                elocs_.size();        // media di Eloc
    elocsC_ = sampler_.GetSamplingEnergies(); // energie COMPLESSE per il gradiente
    derLogs_ = sampler_.GetDerLogs(); // salvo derlogs
  }


  void RunTraining(int nsweeps, int niter, int loadtype = 0, int nprint = 2000,
                   std::string label = "H2",
                   std::string trainingfn = "Training.dat") {

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    nsweeps_ = nsweeps;                // salva numero di sweeps
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
    // solo il master apre e scrive: piu' rank in troncamento sullo stesso
    // file si azzererebbero a vicenda
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
    PrintStats(Iter0_ + niter);
    Iter0_ += niter;
  }

  void CheckParametersConsistency() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto pars = nqs_.GetParameters();


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

  

  void UpdateParameters() {
    auto pars = nqs_.GetParameters();
    opt_.Update(grad_, pars); // usa lo stesso grad per training e pretraining
    nqs_.SetParameters(pars);
  }

  void SetOutputFileTraining(
      std::string outputfile) { // predispone il file su cui stampare iterazione
                                // corrente, Eloc, sigma, loss
    out_.open(outputfile.c_str(), std::ios_base::out);
    if (!out_.is_open()) {
      std::cerr << "# Error : Cannot open file for writing optimization data"
                << std::endl;
      std::abort();
    }
    out_ << "Niter" << "\t" << "Eloc" << "\t" << "sigma" << "\t" << "loss"
         << "\t";
    // for (int i = 0; i < npar_; i++)
    // out_ << "par" << i << "\t";
    // out_ << endl;

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
           << sigmaElocmean_ << "\t"; // << loss_ << "\t";
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
    std::ofstream out("TrainingTest/weights" + label + std::to_string(N) +
                      ".dat");
    if (rank == 0) {

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
 
  
  void realPart(const vector<complex<double>> &x, vector<double> &y) {
    for (int i = 0; i < x.size(); i++) {
      y[i] = x[i].real();
    }
  }



























//////////////////////////CG_FAC METHODS///////////////////////////////////



void RunTrainingCG(int nsweeps, int niter, int loadtype = 0, int nprint = 3000,
                   std::string trainingfn = "TrainingH2.dat",
                   std::string label = "H2", double cg_lambda = 0.01,
                   double cg_rtol = 1e-4, int cg_maxiter = 50,
                   double cg_maxstep = 0.1) {

   int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

     nsweeps_ = nsweeps;
    SetOutputFileTraining(trainingfn);
// se ricarico i pesi, ricarico anche lr
   if (loadtype == 2 || loadtype == 3) 
    {
      opt_.SetEta(nqs_.GetEtaAfterRestart()); // carica learning rate e progresso nel
                                      // decadimento dalla run precedente
      opt_.SetNiter(nqs_.GetNiterAfterRestart());
    }
    // solo il master apre e scrive (vedi RunTraining)
    std::ofstream outeta, outacc, outcg;
    if (rank == 0) {
      std::filesystem::create_directories("SamplingTest");
      outeta.open("SamplingTest/LR" + label + ".dat");
      outacc.open("SamplingTest/Acceptance" + label + ".dat");
      outacc << "i" << setw(12) << "acc" << std::endl;
      outcg.open("SamplingTest/CG" + label + ".dat");
      outcg << "i" << setw(10) << "cg_iter" << setw(14) << "res_rel"
            << setw(14) << "lambda_eff" << setw(14) << "|delta|"
            << setw(9) << "clip" << std::endl;
    }

        for (int i = 0; i < niter; i++) { // ciclo sul numero di iterazioni


  Sample();

  const int M_local = derLogs_.size();
  const int P = nqs_.GetNpar();
  int M_total = 0;

  MPI_Allreduce(&M_local, &M_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (M_total == 0) {
    continue;
  }

  Omean_.assign(P, std::complex<double>(0.0, 0.0));
//omean contiene le medie dei capioni montecarlo delle derlogs su ogni parametro della rete
  for (int m = 0; m < M_local; m++) {
    for (int j = 0; j < P; j++) {
      Omean_[j] += derLogs_[m][j];
    }
  }
//calcolo Omean medio sui rank
 std::vector<std::complex<double>>Omean_global;
 Omean_global.assign(Omean_.size(), std::complex<double>(0.0, 0.0));

      MPI_Allreduce(Omean_.data(), Omean_global.data(), P, MPI_DOUBLE_COMPLEX, MPI_SUM,
                    MPI_COMM_WORLD);
  
  for (int j = 0; j < P; j++) {
    Omean_[j] = Omean_global[j] / static_cast<double>(M_total);
  }

//calcolo energie
   double local_energy_sum = 0.0;
      for (int m = 0; m < M_local; m++) {
        local_energy_sum += elocs_[m];
      }
   double sum_energies = 0.0;
      MPI_Allreduce(&local_energy_sum, &sum_energies, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);

      // 2. Calcola la media dividendo per il numero di nodi
double avg_energy = sum_energies / static_cast<double>(M_total);


      double sq_diff_sum = 0.0;
      double local_sq_diff = 0.0;
      for (int m = 0; m < M_local; m++) {
          local_sq_diff +=
              (elocs_[m] - avg_energy) * (elocs_[m] - avg_energy);
      }
      MPI_Allreduce(&local_sq_diff, &sq_diff_sum, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);

      double std_dev = std::sqrt(sq_diff_sum / static_cast<double>(M_total));
      double std_err = std_dev / std::sqrt(static_cast<double>(M_total));

      // 4. Update the local energy with the averaged energy
      elocmean_ = avg_energy;
      sigmaElocmean_ = std_err;

      CGGradient();
// 1. Calcola la somma dei gradienti usando MPI_Allreduce
      std::vector<std::complex<double>> sum_gradients(grad_.size());
      MPI_Allreduce(grad_.data(), sum_gradients.data(), grad_.size(),
                    MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);

      // 2. Calcola la media dividendo per il numero totale di campioni
      for (int j = 0; j < grad_.size(); ++j) {
        grad_[j] = sum_gradients[j] / static_cast<double>(M_total);
      }

      PrintStats(i);
      UpdateParametersCG(cg_lambda, cg_rtol, cg_maxiter, cg_maxstep);
      MPI_Barrier(MPI_COMM_WORLD);
            if (((i + 1) % 50) == 0) 
        CheckParametersConsistency();
            if ((i + 1) % nprint == 0) 
        SetupReset(i, label);
      if (rank == 0) {
        outacc << i << setw(12) << sampler_.Acceptance() << std::endl;
        outcg << i << setw(10) << cgIters_ << setw(14) << cgResidual_
              << setw(14) << cgLambdaEff_ << setw(14) << cgStepNorm_
              << setw(9) << cgClipped_ << std::endl;
        outeta << i << setw(12) << opt_.GetEta() << std::endl;
      }

   
}
    
    outacc.close();
    outeta.close();
    PrintStats(Iter0_ + niter);
    Iter0_ += niter;

}
//metodo per l'update matrix-free: invece di calcolare tutta la fischer, ne calcolo l'azione component wise sui vettori
//si può ottenere riscalando le Oi nella definizione della S
  inline std::vector<std::complex<double>> Apply_Fischer(const std::vector<std::complex<double>> &v){

      std::vector<std::complex<double>> vi;
      vi.assign(nqs_.GetNpar(), std::complex<double>(0.0, 0.0));
      std::vector<std::complex<double>> vi_global;
      vi_global.assign(nqs_.GetNpar(), std::complex<double>(0.0, 0.0));
      int M_local = derLogs_.size();
      int M_total = 0;
      MPI_Allreduce(&M_local, &M_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
//loop sulle configurazioni campionate
      for(int m=0; m<M_local; m++){
           std::complex<double> acc=(0.0,0.0);
           //sommo sui parametri
        for(int j=0; j<nqs_.GetNpar(); j++){
        acc+=(derLogs_[m][j]-Omean_[j])*v[j];
        }
        //calcolo le componenti del vettore risultante (Sv)

        for(int i=0; i< nqs_.GetNpar(); i++){
          vi[i]+=conj(derLogs_[m][i]-Omean_[i])*acc;
        }
      }

      MPI_Allreduce(vi.data(), vi_global.data(), nqs_.GetNpar(),
                    MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);

      if (M_total > 0) {
        for (int i=0; i<nqs_.GetNpar(); i++) {
          vi_global[i] /= static_cast<double>(M_total);
        }
      }
    
    return vi_global;
    }

//introduce damping sulla fischer per stabilizzare l'approssimazione
  // Scala tipica degli autovalori di S: media della diagonale
  // S_ii = <|O_i - <O_i>|^2>. Rende il damping ADIMENSIONALE: un lambda
  // assoluto e' inutilizzabile perche' la scala di S cambia di ordini di
  // grandezza durante il training.
  inline double MeanFisherDiagonal() {
    const int P = nqs_.GetNpar();
    int M_local = derLogs_.size(), M_total = 0;
    MPI_Allreduce(&M_local, &M_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    double loc = 0.0;
    for (int m = 0; m < M_local; m++)
      for (int j = 0; j < P; j++) loc += std::norm(derLogs_[m][j] - Omean_[j]);
    double glob = 0.0;
    MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (M_total > 0 && P > 0) return glob / (double(M_total) * double(P));
    return 0.0;
  }

  inline std::vector<std::complex<double>> Apply_A(const std::vector<std::complex<double>> &v, double lambda) {
//\lambda controlla quanto ti fidi della curvatura: piccolo = molto fiducia nella Fisher; grande = ti proteggi e ti avvicini a SGD.
  std::vector<std::complex<double>> Av = Apply_Fischer(v);
//introduco damping
  for (int i = 0; i < nqs_.GetNpar(); i++) {
    Av[i] += lambda * v[i];
  }

  return Av;
}



//calcola gradiente 
void CGGradient() {

  //inizializzo vettore-gradiente
  for (int j = 0; j < npar_; j++) 
    grad_[j] = 0.0;
//loop sulle configurazioni campionate
  for (int m = 0; m < derLogs_.size(); m++) {
    std::complex<double> deltaE = elocsC_[m] - elocmean_;
    //calcolo <O_i^m-<O_i>>
    //loop sui parametri
    for (int j = 0; j < npar_; j++) {

      std::complex<double> Otilde = derLogs_[m][j] - Omean_[j];

      grad_[j] += deltaE * std::conj(Otilde);
    }
  }
}

void TrainingGradient() {

  Omean_.assign(npar_, std::complex<double>(0.0, 0.0));
  for (int m = 0; m < derLogs_.size(); m++) {
    for (int j = 0; j < npar_; j++) {
      Omean_[j] += derLogs_[m][j];
    }
  }
  if (!derLogs_.empty()) {
    for (int j = 0; j < npar_; j++) {
      Omean_[j] /= static_cast<double>(derLogs_.size());
    }
  }

  //inizializzo vettore-gradiente
  for (int j = 0; j < npar_; j++)
    grad_[j] = 0.0;
//loop sulle configurazioni campionate
  for (int m = 0; m < derLogs_.size(); m++) {
    std::complex<double> deltaE = elocsC_[m] - elocmean_;
    //calcolo <O_i^m-<O_i>>
    //loop sui parametri
    for (int j = 0; j < npar_; j++) {

      std::complex<double> Otilde = derLogs_[m][j] - Omean_[j];

      grad_[j] += deltaE * std::conj(Otilde) / double(derLogs_.size());
    }
  }
}


std::complex<double> Dot( //calcola prodotto scalare ab=a^dagger b
    const std::vector<std::complex<double>> &a,
    const std::vector<std::complex<double>> &b){

      if(a.size()!=b.size()){
        std::cerr<<"ERROR: SCALAR PRODUCT RECEIVED TO IMPUT VECTORS WITH DIFFERENT SIZES"<<std::endl;
        exit(1);
      }
      std::complex<double> acc=(0.0,0.0);
      for(int i=0; i<a.size(); i++){

        acc+=std::conj(a[i])*b[i];

      }
      return acc;
    }

//risolvo iterativamente il sistema per trovare il deltatheta, invece che invertire la matice di fischer
// lambda e' RELATIVO alla scala di S (vedi MeanFisherDiagonal); rtol e' una
// tolleranza RELATIVA (||r|| <= rtol*||b||); maxstep e' una trust region sulla
// norma di delta, senza la quale S^-1 amplifica le direzioni a piccolo
// autovalore e il primo update distrugge la funzione d'onda.
std::vector<std::complex<double>> ConjugateGradientSolver(double lambda = 0.01,
                                                          double rtol = 1e-4,
                                                          int maxiter = 50,
                                                          double maxstep = 0.1) {

  int P = npar_;
  const double sdiag = MeanFisherDiagonal();
  const double lambda_eff = lambda * (sdiag > 0.0 ? sdiag : 1.0);
  cgLambdaEff_ = lambda_eff;
  cgIters_ = 0; cgResidual_ = 0.0; cgStepNorm_ = 0.0; cgClipped_ = 0;
  const double tiny = 1.0e-14;

  std::vector<std::complex<double>> b(P);
  std::vector<std::complex<double>> dt(P);
  std::vector<std::complex<double>> r(P);
  std::vector<std::complex<double>> p(P);
  std::vector<std::complex<double>> Ap(P);

  // CONVENZIONE DEI SEGNI: risolviamo (S + lambda I) delta = +grad, cosi' delta
    // e' una quantita' "gradient-like" che l'ottimizzatore SOTTRAE, esattamente
    // come CGFAC fa con grad_ (pars += -eta*direzione). Con b = -grad_ il segno
    // verrebbe negato due volte e l'energia SALIREBBE invece di scendere.
  for (int i = 0; i < P; i++) {
    b[i] = grad_[i];
  }

  // x = 0
  dt.assign(P, std::complex<double>(0.0, 0.0));

  // r = b
  r = b;

  // p = r
  p = r;

  std::complex<double> rr = Dot(r, r);
  const double bnorm = std::sqrt(std::max(std::real(rr), 0.0));
  if (bnorm < tiny) {
    return dt;
  }
  const double target = rtol * bnorm;

  for (int k = 0; k < maxiter; k++) {

    Ap = Apply_A(p, lambda_eff);

    std::complex<double> pAp = Dot(p, Ap);
    if (std::abs(pAp) < tiny) {
      break;
    }

    std::complex<double> alpha = rr / pAp;

    for (int i = 0; i < P; i++)
      dt[i] += alpha * p[i];

    for (int i = 0; i < P; i++)
      r[i] -= alpha * Ap[i];

    std::complex<double> rr_new = Dot(r, r);

    cgIters_ = k + 1;
    if (std::sqrt(std::max(std::real(rr_new), 0.0)) <= target) { rr = rr_new; break; }

    if (std::abs(rr) < tiny) {
      break;
    }

    std::complex<double> beta = rr_new / rr;

    for (int i = 0; i < P; i++)
      p[i] = r[i] + beta * p[i];

    rr = rr_new;
  }

  cgResidual_ = std::sqrt(std::max(std::real(rr), 0.0)) / bnorm;
  // trust region: limita la norma dello spostamento nei parametri
  double dnorm = std::sqrt(std::max(std::real(Dot(dt, dt)), 0.0));
  cgStepNorm_ = dnorm;
  if (maxstep > 0.0 && dnorm > maxstep) {
    const double sc = maxstep / dnorm;
    for (int i = 0; i < P; i++) dt[i] *= sc;
    cgClipped_ = 1;
  }
  return dt;
}



void UpdateParametersCG(double lambda, double rtol, int maxiter, double maxstep){
  std::vector<std::complex<double>> dt =
      ConjugateGradientSolver(lambda, rtol, maxiter, maxstep);
    auto pars = nqs_.GetParameters();
opt_.Update(dt, pars);
  nqs_.SetParameters(pars);
}
};



























