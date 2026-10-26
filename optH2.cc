#include "mpi.h" // Include per MPI
#include "src/nqs_preopt.hh"
#include "src/CGFAC.cc"
#include "src/VariationalCG.cc"
#include "src/ArgonneV6Lattice.cc"
#include <getopt.h>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv); // Inizializza MPI

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Ottiene il rank del processo
  MPI_Comm_size(MPI_COMM_WORLD, &size); // Ottiene il numero totale di processi

  auto opts = ReadOptionsFromFile("INPUTH2_CG.dat");
  // bool isospin=std::get<bool>(opts["isospin"]);
  bool isospin = std::get<bool>(opts["isospin"]);

  int isoUp = std::get<int>(opts["iso_up"]);
  int isoDown = std::get<int>(opts["iso_down"]);
  if(rank==0){
std::cout<<"running on "<<size<<"nodes"<<std::endl;
  }
  // Definining the neural-network wave-function
  int loadtype = std::get<int>(opts["loadtype"]);
  // 0:random params, 1: random nh, 2: all loaded parameters, 3: loaded+ new
  // random nhs
  Nqs wavef(std::get<int>(opts["nv"]), std::get<int>(opts["nh"]),
            std::get<double>(opts["sigma"]), loadtype, isospin,
            std::get<std::string>(opts["restart"]));

  int nsweeps = std::get<int>(opts["nsweeps"]);
  int nspins = wavef.Nspins();
  double magnetization = std::get<int>(opts["M"]);
  int Nup = std::get<int>(opts["spin_up"]);
  int Ndown = std::get<int>(opts["spin_down"]);
  // Problem hamiltonian inferred from file name
  std::string model =
      std::get<std::string>(opts["model"]); // ArgonneV6 for deuteron
  // bool printstates=std::get<bool>(opts["filestates"]);

  bool printstates = std::get<bool>(opts["printstates"]);
  // Imposta seed diverso per ogni processo
  int seed = std::get<int>(opts["seed"]) + rank * 10;

  if (rank == 0) {
    std::cout << "Running on " << size << " " << "nodes" << std::endl;
    std::cout << "Model: " << model << std::endl;
    std::cout << "Number of sweeps: " << nsweeps / size << std::endl;
  }

  try {
   /* if (model == "Trap1d") {
      Trap1d hamiltonian(nspins, magnetization);
      typedef Sampler<Nqs, Trap1d> Sampler;
      Sampler sampler(wavef, hamiltonian, seed, Nup, Ndown);
      if (printstates && rank == 0)
        sampler.SetFileStates(std::get<std::string>(opts["filestates"]));
      double eta = 0.0009, eta1 = 1000;
      Sgd opt(eta, eta1);
      Variational<Trap1d, Nqs, Sampler, Sgd> var(sampler, opt);
      var.RunTraining(nsweeps / size, 1000);

    } else if (model == "Trap3d") {
      Trap3d hamiltonian(nspins, magnetization);
      typedef Sampler<Nqs, Trap3d> Sampler;
      Sampler sampler(wavef, hamiltonian, seed, Nup, Ndown);
      if (printstates && rank == 0)
        sampler.SetFileStates(std::get<std::string>(opts["filestates"]));
      double eta = 0.0009, eta1 = 1000;
      Sgd opt(eta, eta1);
      Variational<Trap3d, Nqs, Sampler, Sgd> var(sampler, opt);
      var.RunTraining(nsweeps / size, 1000);

    } else if (model == "Coulomb1d") {
      Coulomb1d hamiltonian(nspins, magnetization);
      typedef Sampler<Nqs, Coulomb1d> Sampler;
      Sampler sampler(wavef, hamiltonian, seed, Nup, Ndown);
      if (printstates && rank == 0)
        sampler.SetFileStates(std::get<std::string>(opts["filestates"]));
      double eta = 0.00012, eta1 = 20000;
      Sgd opt(eta, eta1);
      Variational<Coulomb1d, Nqs, Sampler, Sgd> var(sampler, opt);
      var.RunTraining(nsweeps / size, 1000);

    } else if (model == "Coulomb3d") {
      Coulomb3d hamiltonian(nspins, magnetization,
                            std::get<double>(opts["traplength"]),
                            std::get<double>(opts["Nprotons"]));
      typedef Sampler<Nqs, Coulomb3d> Sampler;
      Sampler sampler(wavef, hamiltonian, seed, Nup, Ndown);
      if (printstates && rank == 0)
        sampler.SetFileStates(std::get<std::string>(opts["filestates"]));
      double eta = std::get<double>(opts["eta"]);
      double eta1 = std::get<double>(opts["eta1"]);
      Sgd opt(eta, eta1);
      Variational<Coulomb3d, Nqs, Sampler, Sgd> var(sampler, opt);
      int Niter = std::get<int>(opts["niter"]);
      int get_pars = std::get<int>(opts["get_pars"]);
      for (int i = 0; i < 50; i++)
        sampler.Sample(5);
      var.RunTraining(5 / size, 1, loadtype, get_pars);
    
      // sampler.OutputEnergyValues();

    }

    else*/ if (model == "ArgonneV6") {

      ArgonneV6 hamiltonian(nspins, magnetization,
                            std::get<double>(opts["traplength"]),
                            std::get<double>(opts["Nprotons"]));
      typedef SamplerH2<Nqs, ArgonneV6> SamplerH2; // deuteron uses a modified sampler which leaves the COM
                     // fixed
      SamplerH2 sampler(wavef, hamiltonian, seed, Nup, Ndown, isoUp, isoDown);

      double eta = std::get<double>(opts["eta"]);
      double eta1 = std::get<double>(opts["eta1"]);
      int Niter = std::get<int>(opts["niter"]);

      int get_pars = std::get<int>(opts["get_pars"]);
      // sampler.InitRandomStateH2();
  // for(int i=0; i<100000; i++)
      // sampler.Sample(20);
      //sampler.InitRandomState();
      //wavef.InitLt(sampler.GetState());
      // sampler.Sample(10);
      std::string label=std::get<std::string>(opts["label"]);
    std::string trainingfn=std::get<std::string>(opts["trainingfn"]);

      // scelta dell'ottimizzatore: "cg" usa VariationalCG + CGFAC (gradiente
      // naturale risolto col gradiente coniugato), altrimenti SGD.
      // selezione da INPUTH2.dat (chiave "optimizer"), con override da
      // riga di comando:  ./optH2 cg
      std::string optimizer = "sgd";
      if (opts.count("optimizer"))
        optimizer = std::get<std::string>(opts["optimizer"]);
      if (argc > 1)
        optimizer = argv[1];
      double cg_lambda = opts.count("cg_lambda")
                             ? std::get<double>(opts["cg_lambda"]) : 0.01;
      double cg_rtol = opts.count("cg_rtol")
                           ? std::get<double>(opts["cg_rtol"]) : 1e-4;
      int cg_maxiter = opts.count("cg_maxiter")
                           ? std::get<int>(opts["cg_maxiter"]) : 50;
      double cg_maxstep = opts.count("cg_maxstep")
                              ? std::get<double>(opts["cg_maxstep"]) : 0.1;

      if (optimizer == "cg") {
        if (rank == 0)
          std::cout << "# Optimizer: CG-SR  lambda_rel=" << cg_lambda
                    << " rtol=" << cg_rtol << " maxiter=" << cg_maxiter
                    << " maxstep=" << cg_maxstep << std::endl;
        CGFAC opt(eta, eta1);
        VariationalCG<ArgonneV6, Nqs, SamplerH2, CGFAC> var(sampler, opt);
        var.RunTrainingCG(nsweeps / size, Niter, loadtype, get_pars, trainingfn,
                          label, cg_lambda, cg_rtol, cg_maxiter, cg_maxstep);
      } else {
        if (rank == 0)
          std::cout << "# Optimizer: SGD" << std::endl;
        Sgd opt(eta, eta1);
        Variational<ArgonneV6, Nqs, SamplerH2, Sgd> var(sampler, opt);
        var.RunTraining(nsweeps / size, Niter, loadtype, get_pars, label,
                        trainingfn);
      }
      // var.RunTraining(5, 5, loadtype, get_pars);s
        //double acc=sampler.Acceptance();
    //  std::cout<<"acceptance="<<acc<<std::endl;

    }

    else if (model == "ArgonneV6Lattice") {
      // Potenziale AV6' smussato (regolatore gaussiano, larghezza
      // smear_b_factor*dx) e ricalibrato per riprodurre -2.2246 MeV su
      // QUESTO reticolo (vedi src/ArgonneV6Lattice.cc). La calibrazione
      // avviene una sola volta, qui nel costruttore.
      double smear_b_factor = opts.count("smear_b_factor")
                                   ? std::get<double>(opts["smear_b_factor"])
                                   : 1.0;
      ArgonneV6Lattice hamiltonian(nspins, magnetization,
                                   std::get<double>(opts["traplength"]),
                                   std::get<double>(opts["Nprotons"]),
                                   197.327053, (938.27231 + 939.56563) / 2,
                                   smear_b_factor);
      typedef SamplerH2<Nqs, ArgonneV6Lattice> SamplerH2;
      SamplerH2 sampler(wavef, hamiltonian, seed, Nup, Ndown, isoUp, isoDown);

      double eta = std::get<double>(opts["eta"]);
      double eta1 = std::get<double>(opts["eta1"]);
      int Niter = std::get<int>(opts["niter"]);
      int get_pars = std::get<int>(opts["get_pars"]);
      std::string label = std::get<std::string>(opts["label"]);
      std::string trainingfn = std::get<std::string>(opts["trainingfn"]);

      std::string optimizer = "sgd";
      if (opts.count("optimizer"))
        optimizer = std::get<std::string>(opts["optimizer"]);
      if (argc > 1)
        optimizer = argv[1];
      double cg_lambda = opts.count("cg_lambda")
                             ? std::get<double>(opts["cg_lambda"]) : 0.01;
      double cg_rtol = opts.count("cg_rtol")
                           ? std::get<double>(opts["cg_rtol"]) : 1e-4;
      int cg_maxiter = opts.count("cg_maxiter")
                           ? std::get<int>(opts["cg_maxiter"]) : 50;
      double cg_maxstep = opts.count("cg_maxstep")
                              ? std::get<double>(opts["cg_maxstep"]) : 0.1;

      if (optimizer == "cg") {
        if (rank == 0)
          std::cout << "# Optimizer: CG-SR  lambda_rel=" << cg_lambda
                    << " rtol=" << cg_rtol << " maxiter=" << cg_maxiter
                    << " maxstep=" << cg_maxstep << std::endl;
        CGFAC opt(eta, eta1);
        VariationalCG<ArgonneV6Lattice, Nqs, SamplerH2, CGFAC> var(sampler, opt);
        var.RunTrainingCG(nsweeps / size, Niter, loadtype, get_pars, trainingfn,
                          label, cg_lambda, cg_rtol, cg_maxiter, cg_maxstep);
      } else {
        if (rank == 0)
          std::cout << "# Optimizer: SGD" << std::endl;
        Sgd opt(eta, eta1);
        Variational<ArgonneV6Lattice, Nqs, SamplerH2, Sgd> var(sampler, opt);
        var.RunTraining(nsweeps / size, Niter, loadtype, get_pars, label,
                        trainingfn);
      }
    }

    else {
      if (rank == 0) {
        std::cerr << "Error: Unsupported model type!" << std::endl;
      }
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  } catch (const std::exception &e) {
    std::cerr << "Error on rank " << rank << ": " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Sincronizzazione al termine del calcolo
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "Training completed successfully!" << std::endl;
  }

  MPI_Finalize();
  return 0;
}
