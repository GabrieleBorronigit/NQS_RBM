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

 auto opts=ReadOptionsFromFile("INPUTH2.dat");

  //Definining the neural-network wave-function
 int loadtype= std::get<int>(opts["loadtype"]);
 //0:random params, 1: random nh, 2: all loaded parameters, 3: loaded+ new random nhs

bool isospin=true;
  Nqs wavef(std::get<int>(opts["nv"]), std::get<int>(opts["nh"]), std::get<double>(opts["sigma"]), loadtype, isospin, std::get<std::string>(opts["restart"]));

  int nsweeps=std::get<int>(opts["nsweeps"]);
  int nspins=wavef.Nspins();
  double magnetization = std::get<int>(opts["M"]);
  int Nup = std::get<int>(opts["spin_up"]);
  int Ndown = std::get<int>(opts["spin_down"]);
  //Problem hamiltonian inferred from file name
 // bool printstates=std::get<bool>(opts["filestates"]);

bool  printstates=std::get<bool>(opts["printstates"]);
  // Imposta seed diverso per ogni processo
  int seed = std::get<int>(opts["seed"]) + rank*10; 

  if (rank == 0) {
        std::cout << "Running on " << size << " " << "nodes" << std::endl;
        std::cout << "Number of sweeps: " << nsweeps/size << std::endl;
  }
if(isospin==false){
            Coulomb3d hamiltonian(nspins, magnetization, std::get<double>(opts["traplength"]), std::get<double>(opts["Nprotons"]));
            typedef Sampler<Nqs, Coulomb3d> Sampler;
            Sampler sampler(wavef, hamiltonian, seed, Nup, Ndown);
            if (printstates && rank == 0) sampler.SetFileStates(std::get<std::string>(opts["filestates"]));
            int Niter=std::get<int>(opts["niter"]);
            int get_pars=std::get<int>(opts["get_pars"]);
            std::string label=std::get<std::string>(opts["label"]);
            sampler.Plot(nsweeps);

if(Nup+Ndown==2){
int L  = 11;
int c  = (L - 1)/2;
int Nsites = L*L*L;
int site_core = c + L*(c + L*c);

// indici: 0 = up, 1 = down
int e_up   = 0;
int e_down = 1;

std::vector<std::vector<double>> psi_z1z2(L, std::vector<double>(L, 0.0));

for (int z_up = 0; z_up < L; ++z_up) {
    for (int z_down = 0; z_down < L; ++z_down) {

        int pos_up   = c + L*(c + L*z_up);
        int pos_down = c + L*(c + L*z_down);

        std::vector<int> state(wavef.GetNvtot(), -1);

        // up
        state[pos_up] = 1;

        // down
        state[Nsites + pos_down] = 1;

        auto logpsi = wavef.LogVal(state);
        double psi  = std::exp(logpsi.real());   // |ψ|

        psi_z1z2[z_up][z_down] = psi;
    }
}

  if (rank == 0) {
    std::ofstream outxyhe("psi_z1z2He.dat");
    outxyhe << std::setprecision(15);
    for (int z1 = 0; z1 < L; ++z1) {
      for (int z2 = 0; z2 < L; ++z2) {
        outxyhe << z1 << " " << z2 << " " << psi_z1z2[z1][z2] << "\n";
      }
      outxyhe << "\n";
    }
    outxyhe.close();
  }



        // elio singoletto: Nup = 1, Ndown = 1  => Ne = 2
        int Ne = Nup + Ndown;                 // controlla che Ne == 2

        // indici logici degli elettroni
         e_up   = 0;   // elettrone up (core)
         e_down = 1;   // elettrone down (mobile sul piano xy)

        std::vector<std::vector<double>> psi_xy(L, std::vector<double>(L, 0.0));

        int z_plane = c;  // piano equatoriale z = c

        for (int x = 0; x < L; ++x) {
            for (int y = 0; y < L; ++y) {

                // posizione up fissato al nucleo
                int pos_up   = site_core;

                // posizione down mobile sul piano xy a z = c
                int pos_down = x + L * (y + L * z_plane);

                // costruisci lo stato RBM: [0..Nsites-1] = up, [Nsites..2Nsites-1] = down
                std::vector<int> state(wavef.GetNvtot(), -1);

                // up al nucleo
                state[pos_up] = 1;

                // down mobile
                state[Nsites + pos_down] = 1;

                // valuta |psi|
                std::complex<double> logpsi = wavef.LogVal(state);
                double psi = std::exp(logpsi.real());   // |Ψ|

                psi_xy[x][y] = psi;
            }
        }

        // scrivi su file x y |psi|
        std::ofstream out_xy("psi_xy_He_singlet.dat");
        out_xy << std::setprecision(15);
        for (int y = 0; y < L; ++y) {
            for (int x = 0; x < L; ++x) {
                out_xy << x << " " << y << " " << psi_xy[x][y] << "\n";
            }
            out_xy << "\n";
        }
        out_xy.close();
                // --- caso 2 up sul piano xy (Pauli manuale) ---

   
   

        // 2 up: Ne = 2, nessun down
        int e_up_core   = 0;   // up fissato al nucleo
        int e_up_mobile = 1;   // up mobile sul piano xy

        std::vector<std::vector<double>> psi_xy_2up(L, std::vector<double>(L, 0.0));

        for (int x = 0; x < L; ++x) {
            for (int y = 0; y < L; ++y) {

                int pos_core   = site_core;
                int pos_mobile = x + L * (y + L * z_plane);

                // Pauli "a mano": up mobile non può stare sul nucleo (dove c'è l'altro up)
                if (pos_mobile == pos_core) {
                    psi_xy_2up[x][y] = 0.0;
                    continue;
                }

                std::vector<int> state(wavef.GetNvtot(), -1);

                // entrambi nel settore up
                state[pos_core]   = 1;
                state[pos_mobile] = 1;

                auto logpsi = wavef.LogVal(state);
                double psi  = std::exp(logpsi.real());

                psi_xy_2up[x][y] = psi;
            }
        }

        std::ofstream out_xy_2up("psi_xy_He_2up.dat");
        out_xy_2up << std::setprecision(15);
        for (int y = 0; y < L; ++y) {
            for (int x = 0; x < L; ++x) {
                out_xy_2up << x << " " << y << " " << psi_xy_2up[x][y] << "\n";
            }
            out_xy_2up << "\n";
        }
        out_xy_2up.close();




int e_up1 = 0;
int e_up2 = 1;

// mappa |psi|(z1,z2) per i due up sul filo z (x=y=c)
std::vector<std::vector<double>> psi_z1z22(L, std::vector<double>(L, 0.0));

for (int z1 = 0; z1 < L; ++z1) {
    for (int z2 = 0; z2 < L; ++z2) {

        // posizione spaziale (senza spin) dei due up sulla colonna (c,c,z)
        int pos_up1 = c + L * (c + L * z1);
        int pos_up2 = c + L * (c + L * z2);

        // Pauli "a mano": due up nello stesso sito ⇒ ampiezza zero
        if (pos_up1 == pos_up2) {
            psi_z1z2[z1][z2] = 0.0;
            continue;
        }

        // costruisci lo stato visibile RBM: [0..Nsites-1] = up
        std::vector<int> state(wavef.GetNvtot(), -1);

        state[pos_up1] = 1;  // primo up
        state[pos_up2] = 1;  // secondo up

        // valuta log ψ e |ψ|
        std::complex<double> logpsi = wavef.LogVal(state);
        double psi = std::exp(logpsi.real());   // |ψ|, se LogVal = log|ψ|

        psi_z1z22[z1][z2] = psi;
    }
}

// scrittura su file: z1 z2 |psi|
std::ofstream outzz("psi_z1z2_He_2up.dat");
outzz << std::setprecision(15);
for (int z1 = 0; z1 < L; ++z1) {
    for (int z2 = 0; z2 < L; ++z2) {
        outzz << z1 << " " << z2 << " " << psi_z1z22[z1][z2] << "\n";
    }
    outzz << "\n";
}
outzz.close();



}

if(Nup+Ndown==3){
        // --- parametri reticolo e indici particelle ---
        int L  = 11;
        int c  = (L - 1) / 2;                 // centro reticolo (5)
        int Ne = Nup + Ndown;                // per il litio: 3
        int site_core = c + L * (c + L * c); // indice del nucleo (c,c,c)
        int Nsites = L * L * L;              // numero di siti per spin

        // indici logici nel vettore "positions"
        int e_up1  = 0; // primo elettrone up
        int e_up2  = 1; // secondo elettrone up
        int e_down = 2; // elettrone down

        // mappa ψ(z1,z2) per due up che si muovono su z, down fisso al nucleo
        std::vector<std::vector<double>> psi_z1z2(L, std::vector<double>(L, 0.0));

        for (int z1 = 0; z1 < L; ++z1) {
            for (int z2 = 0; z2 < L; ++z2) {

                // posizioni spaziali (senza spin) dei due up lungo la stessa colonna (x=y=c)
                int pos_up1  = c + L * (c + L * z1);
                int pos_down  = c + L * (c + L * z2);
                int pos_up2 = site_core;              // down fissato al nucleo

                // Pauli “a mano”: due up nello stesso sito ⇒ ampiezza zero
                if (pos_up1 == pos_up2) {
                    psi_z1z2[z1][z2] = 0.0;
                    continue;
                }

                // vettore delle posizioni (solo per chiarezza, NON contiene lo shift di spin)
                std::vector<int> positions(Ne);
                positions[e_up1]  = pos_up1;
                positions[e_up2]  = pos_up2;
                positions[e_down] = pos_down;

                // stato visibile RBM: [0..Nsites-1] = up, [Nsites..2*Nsites-1] = down
                std::vector<int> state(wavef.GetNvtot(), -1);

                // occupo i siti up
                state[ positions[e_up1] ] = 1;         // up1
                state[ positions[e_up2] ] = 1;         // up2

                // occupo il sito down (shiftato di Nsites)
                state[ Nsites + positions[e_down] ] = 1;

                // calcolo log ψ e il modulo |ψ|
                std::complex<double> logpsi = wavef.LogVal(state);
                double psi = std::exp(logpsi.real());  // |ψ| (se LogVal = log|ψ|)

                psi_z1z2[z1][z2] = psi;
            }
        }



  if (rank == 0) {
    std::ofstream outzz("psi_z1z2.dat");
    outzz << std::setprecision(15);
    for (int z1 = 0; z1 < L; ++z1) {
      for (int z2 = 0; z2 < L; ++z2) {
        outzz << z1 << " " << z2 << " " << psi_z1z2[z1][z2] << "\n";
      }
      outzz << "\n";
    }
    outzz.close();
  }

  //due elettroni sul nucleo e uno in movimento su piano xy
  std::vector<std::vector<double>> psi_xy(L, std::vector<double>(L, 0.0));

  for (int x1 = 0; x1 < L; ++x1) {
    for (int y1 = 0; y1 < L; ++y1) {

      // z0=znucleo
      int site_up1 = x1 + L * (y1 + L * c);


      // vettore delle posizioni (siti senza spin)
      std::vector<int> positions(Ne);
      positions[e_up1]  = site_up1;
      positions[e_up2]  = site_core;
      positions[e_down] = site_core;    // down fissato al nucleo
           // ===== Pauli "a mano": se i due up sono sullo stesso sito =====
      if (site_up1 == site_core) {
        psi_z1z2[x1][y1] = 0.0;
        continue;  // salta la valutazione della RBM
      }
      // =============================================================
      // stato visibile RBM: primi N siti = spin up, ultimi N = spin down
      std::vector<int> state(wavef.GetNvtot(), -1);
      int Nsites = L * L * L;           // numero siti per spin

      // up1 e up2 nel settore "up"
      state[positions[e_up1]] = 1;
      state[positions[e_up2]] =1;

      // down nel settore "down" (shiftato di Nsites)
      state[Nsites + positions[e_down]] =1;

      // log ψ dalla RBM e modulo
      std::complex<double> logpsi = wavef.LogVal(state);
      double psi = std::exp(logpsi.real());   // |ψ| (non |ψ|^2)

      psi_z1z2[x1][y1] = psi;
    }
  }

  if (rank == 0) {
    std::ofstream outxy("psi_xy.dat");
    outxy << std::setprecision(15);
    for (int z1 = 0; z1 < L; ++z1) {
      for (int z2 = 0; z2 < L; ++z2) {
        outxy << z1 << " " << z2 << " " << psi_z1z2[z1][z2] << "\n";
      }
      outxy << "\n";
    }
    outxy.close();
  }
}
  
}

else{

    ArgonneV6 hamiltonian(nspins, magnetization,
                            std::get<double>(opts["traplength"]),
                            std::get<double>(opts["Nprotons"]));
      typedef SamplerH2<Nqs, ArgonneV6> SamplerH2; // deuteron uses a modified sampler which leaves the COM


      SamplerH2 samplerh2(wavef, hamiltonian, seed, Nup, Ndown,1,1);
        samplerh2.Plot(nsweeps);

}
MPI_Finalize();
  return 0;
}







