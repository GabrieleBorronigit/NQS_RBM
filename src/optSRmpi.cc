#include "src/nqs_preopt.hh"
#include "mpi.h" // Include per MPI
#include <iostream>
#include <map>
#include <string>
#include <sstream>
#include <getopt.h>

int main(int argc, char *argv[]){

  MPI_Init(&argc, &argv); // Inizializza MPI

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Ottiene il rank del processo
  MPI_Comm_size(MPI_COMM_WORLD, &size); // Ottiene il numero totale di processi

  auto opts=ReadOptions(argc,argv);

  //Definining the neural-network wave-function
  Nqs wavef(opts["filename"]);

  int nsweeps=std::stod(opts["nsweeps"]);
  int nspins=wavef.Nspins();
  double magnetization = std::stod(opts["M"]);
  int Nup = std::stoi(opts["spin_up"]);
  int Ndown = std::stoi(opts["spin_down"]);

  //Problem hamiltonian inferred from file name
  std::string model=opts["model"];

  bool printstates=opts.count("filestates");

  // Imposta seed diverso per ogni processo
  int seed = std::stoi(opts["seed"]) + rank*10; 

  if (rank == 0) {
        std::cout << "Running on " << size << " " << "nodes" << std::endl;
        std::cout << "Model: " << model << std::endl;
        std::cout << "Number of sweeps: " << nsweeps/size << std::endl;
  }

    try {
        if (model == "Trap1d") {
            Trap1d hamiltonian(nspins, magnetization);
            typedef Sampler<Nqs, Trap1d> Sampler;
            Sampler sampler(wavef, hamiltonian, seed, Nup, Ndown);
            if (printstates && rank == 0) sampler.SetFileStates(opts["filestates"]);
            double eta = 0.0009, eta1 = 1000;
            SR opt(eta, eta1);
            Variational<Trap1d, Nqs, Sampler, SR> var(sampler, opt);
            var.RunTrainingSR(nsweeps/size, 50);

        } else if (model == "Trap3d") {
            Trap3d hamiltonian(nspins, magnetization);
            typedef Sampler<Nqs, Trap3d> Sampler;
            Sampler sampler(wavef, hamiltonian, seed, Nup, Ndown);
            if (printstates && rank == 0) sampler.SetFileStates(opts["filestates"]);
            double eta = 0.0009, eta1 = 1000;
            SR opt(eta, eta1);
            Variational<Trap3d, Nqs, Sampler, SR> var(sampler, opt);
            var.RunTrainingSR(nsweeps/size, 50);

        } else if (model == "Coulomb1d") {
            Coulomb1d hamiltonian(nspins, magnetization);
            typedef Sampler<Nqs, Coulomb1d> Sampler;
            Sampler sampler(wavef, hamiltonian, seed, Nup, Ndown);
            if (printstates && rank == 0) sampler.SetFileStates(opts["filestates"]);
            double eta = 0.0009, eta1 = 1000;
            SR opt(eta, eta1);
            Variational<Coulomb1d, Nqs, Sampler, SR> var(sampler, opt);
            var.RunTrainingSR(nsweeps/size, 50);

        } else if (model == "Coulomb3d") {
            Coulomb3d hamiltonian(nspins, magnetization);
            typedef Sampler<Nqs, Coulomb3d> Sampler;
            Sampler sampler(wavef, hamiltonian, seed, Nup, Ndown);
            if (printstates && rank == 0) sampler.SetFileStates(opts["filestates"]);
            double eta = 0.2, eta1 = 1.5*pow(10,8);
            SR opt(eta, eta1);
            Variational<Coulomb3d, Nqs, Sampler, SR> var(sampler, opt);
            var.RunTrainingSR(nsweeps/size, 50);

        } else {
            if (rank == 0) {
                std::cerr << "Error: Unsupported model type!" << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } catch (const std::exception& e) {
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
