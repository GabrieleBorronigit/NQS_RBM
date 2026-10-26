#include "nqs_preopt.hh"
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include "mpi.h"

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto opts = ReadOptionsFromFile("INPUT.dat");

    bool isospin = std::get<bool>(opts["isospin"]);
    int isoUp = std::get<int>(opts["iso_up"]);
    int isoDown = std::get<int>(opts["iso_down"]);

    int loadtype = std::get<int>(opts["loadtype"]);
    Nqs wavef(
        std::get<int>(opts["nv"]),
        std::get<int>(opts["nh"]),
        std::get<double>(opts["sigma"]),
        loadtype,
        isospin,
        std::get<std::string>(opts["restart"])
    );

    int nsweeps = std::get<int>(opts["nsweeps"]);
    int nspins = wavef.Nspins();
    double magnetization = std::get<int>(opts["M"]);
    int Nup = std::get<int>(opts["spin_up"]);
    int Ndown = std::get<int>(opts["spin_down"]);

    std::string model = std::get<std::string>(opts["model"]);
    bool printstates = std::get<bool>(opts["printstates"]);
    int seed = std::get<int>(opts["seed"]) + rank * 10;

    if (rank == 0) {
        std::cout << "Running on " << size << " nodes" << std::endl;
        std::cout << "Model: " << model << std::endl;
        std::cout << "Sweeps per node: " << nsweeps / size << std::endl;
    }

    std::ofstream out(std::get<std::string>(opts["trainingfn"]));

    if (model == "Coulomb3d") {

        Coulomb3d hamiltonian(
            nspins, magnetization,
            std::get<double>(opts["traplength"]),
            std::get<double>(opts["Nprotons"])
        );

        typedef Sampler<Nqs, Coulomb3d> Sampler;
        Sampler sampler(wavef, hamiltonian, seed, Nup, Ndown);
        if (printstates && rank == 0)
            sampler.SetFileStates(std::get<std::string>(opts["filestates"]));

        int Niter = std::get<int>(opts["niter"]);

        for (int i = 0; i < Niter; i++) {

            sampler.Sample(nsweeps / size);

            std::vector<std::complex<double>> elocs = sampler.GetSamplingEnergies();
            std::complex<double> elocmean_c =
                std::accumulate(elocs.begin(), elocs.end(),
                                std::complex<double>(0.0, 0.0)) / static_cast<double>(elocs.size());

            // prendo solo la parte reale dell’energia
            double elocmean = std::real(elocmean_c);

            double global_sum = 0.0;
            MPI_Allreduce(&elocmean, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            double avg_energy = global_sum / size;

            double local_sq = (elocmean - avg_energy) * (elocmean - avg_energy);
            double global_sq = 0.0;
            MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            double std_dev = std::sqrt(global_sq / size);
            double std_err = std_dev / std::sqrt(size);

            if (rank == 0)
                out << std::setw(16) << avg_energy
                    << std::setw(16) << std_err << '\n';
        }
    }

    MPI_Finalize();
    return 0;
}
