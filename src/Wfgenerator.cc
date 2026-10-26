#include "nqs_preopt.hh"
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


    auto opts=ReadOptionsFromFile("../INPUT.dat"); //usa restartfn come filename per caricare i pesi e loadtype=2
    Nqs wf(std::get<int>(opts["nv"]), std::get<int>(opts["nh"]), std::get<double>(opts["sigma"]), std::get<int>(opts["loadtype"]), std::get<std::string>(opts["restart"]));
    Coulomb3d hamiltonian(wf.Nspins(), std::get<int>(opts["M"]));
    Sampler<Nqs, Coulomb3d> sampler(wf, hamiltonian, std::get<int>(opts["seed"])+rank*10, std::get<int>(opts["spin_up"]),  std::get<int>(opts["spin_down"]));
      //calcola energia con la psi caicata
    int bins = 50;
    int niter = 20;
    int nsweeps=(int)(150000/size);
		if(rank==0){
std::cout<<"ci sono"<<std::endl;
}
    std::ofstream out("energyAfterTraining200nh.dat");
    if(!out.is_open()){
        std::cerr << "Errore apertura file output" << std::endl;
        return 1;
    }
             if(rank==0){
std::cout<<"ci sono sto per samplare"<<std::endl;
}

for (int it = 0; it < niter; it++) {
    if(rank==0){
        std::cout<<"iter "<<it<<" calculating energy..."<<std::endl;
    }
    // 1) Campionamento locale
    sampler.Sample(nsweeps);
    const auto energies = sampler.GetSamplingEnergies();

    // 2) Somme locali (solo parte reale)
    double local_sum = 0.0, local_sumsq = 0.0;
    int local_n = static_cast<int>(energies.size());
    for (const auto &z : energies) {
        const double e = z.real();
        local_sum   += e;
        local_sumsq += e * e;
    }

    // 3) Riduzioni globali
    double global_sum = 0.0, global_sumsq = 0.0;
    int global_n = 0;
    MPI_Allreduce(&local_sum,   &global_sum,   1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_sumsq, &global_sumsq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_n,     &global_n,     1, MPI_INT,    MPI_SUM, MPI_COMM_WORLD);

    // 4) Media globale ed errore (naïf, senza blocking)
    double Eglob = 0.0, sigma_mean = 0.0;
    if (global_n > 0) {
        Eglob = global_sum / static_cast<double>(global_n);
        const double var_sample = std::max(0.0,global_sumsq / static_cast<double>(global_n) - Eglob * Eglob);
        sigma_mean = std::sqrt(var_sample / static_cast<double>(global_n)); // SEM
    }

    if (rank == 0) {
        out << it << "  "
            << std::setprecision(12) << Eglob << "  "
            << sigma_mean << '\n';
    }
}

    
        // Sincronizzazione al termine del calcolo
    MPI_Barrier(MPI_COMM_WORLD);
 out.close();
    if (rank == 0) {
        std::cout << "Training completed successfully!" << std::endl;
    
    }

    MPI_Finalize();
    return 0;
}
