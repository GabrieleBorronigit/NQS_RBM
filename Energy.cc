// ============================================================================
// EnergyH2 - valutazione di SOLA INFERENZA per il deutone (ArgonneV6 + RBM)
//
// Carica i pesi di una rete gia' addestrata (loadtype=2) e ne misura l'energia
// SENZA aggiornare alcun parametro: campionamento Metropolis lungo, statistica
// con analisi a blocchi e stima del tempo di autocorrelazione, aggregazione
// MPI su catene indipendenti.
//
// Misura anche <r_pn> (distanza protone-neutrone) e l'istogramma radiale, utili
// per confrontare la struttura spaziale con i plot g(r)/rho(r).
//
// Build: mpic++ -std=c++17 -O2 EnergyH2.cc -o EnergyH2
// Run:   mpirun -np N ./EnergyH2 [file_input]        (default: INPUTH2.dat)
//
// L'input e' lo stesso INPUTH2.dat del training, ma:
//   loadtype  2                 <- obbligatorio: carica TUTTI i pesi
//   restartfn <file_pesi>       <- es. TrainingTest/weightsH2.dat
//   nsweeps   <n>               <- sweep di MISURA totali (divisi tra i rank)
// I parametri di training (eta, niter, ...) vengono ignorati.
//
// NOTA SULL'INTERPRETAZIONE: il numero prodotto va confrontato con il ground
// state ESATTO dello stesso reticolo (vedi diagH2.cc), non con il binding
// sperimentale del deutone: a passo reticolare grossolano il modello
// discretizzato sovralega, quindi E_esatta(reticolo) != -2.22 MeV.
// ============================================================================

#include "mpi.h"
#include "src/nqs_preopt.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct BlockStats {
  double mean = 0.0;
  double err = 0.0;      // errore standard dalle medie di blocco
  double tau = 0.0;      // tempo di autocorrelazione (in unita' di sweep)
  long nsamples = 0;
};

// Medie di blocco locali: servono a decorrelare prima di aggregare tra rank.
std::vector<double> BlockMeans(const std::vector<double> &x, int nblocks) {
  std::vector<double> bm;
  if (x.empty() || nblocks <= 0)
    return bm;
  const int bs = static_cast<int>(x.size()) / nblocks;
  if (bs < 1)
    return bm;
  for (int b = 0; b < nblocks; ++b) {
    double s = 0.0;
    for (int i = b * bs; i < (b + 1) * bs; ++i)
      s += x[i];
    bm.push_back(s / static_cast<double>(bs));
  }
  return bm;
}

// Statistica globale: raccoglie le medie di blocco di TUTTI i rank e le tratta
// come campioni indipendenti (catene MPI indipendenti + blocchi decorrelati).
BlockStats GlobalStats(const std::vector<double> &local, int nblocks, int rank,
                       int size) {
  BlockStats st;

  // varianza non bloccata (serve per il tempo di autocorrelazione)
  double loc_sum = 0.0, loc_sumsq = 0.0;
  long loc_n = static_cast<long>(local.size());
  for (double v : local) {
    loc_sum += v;
    loc_sumsq += v * v;
  }
  double g_sum = 0.0, g_sumsq = 0.0;
  long g_n = 0;
  MPI_Allreduce(&loc_sum, &g_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&loc_sumsq, &g_sumsq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&loc_n, &g_n, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (g_n <= 1)
    return st;
  const double mean = g_sum / static_cast<double>(g_n);
  const double var_raw =
      std::max(0.0, g_sumsq / static_cast<double>(g_n) - mean * mean);

  // medie di blocco da tutti i rank
  std::vector<double> bm = BlockMeans(local, nblocks);
  const int nb = static_cast<int>(bm.size());
  std::vector<double> all;
  if (rank == 0)
    all.resize(static_cast<size_t>(nb) * size);
  MPI_Gather(bm.data(), nb, MPI_DOUBLE, rank == 0 ? all.data() : nullptr, nb,
             MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double err = 0.0, tau = 0.0;
  if (rank == 0 && all.size() > 1) {
    const double m = std::accumulate(all.begin(), all.end(), 0.0) / all.size();
    double v = 0.0;
    for (double b : all)
      v += (b - m) * (b - m);
    v /= static_cast<double>(all.size() - 1); // varianza delle medie di blocco
    err = std::sqrt(v / static_cast<double>(all.size()));
    const double bs = static_cast<double>(local.size()) / std::max(1, nblocks);
    if (var_raw > 0.0)
      tau = 0.5 * bs * v / var_raw; // stima standard di tau
  }
  MPI_Bcast(&err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&tau, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  st.mean = mean;
  st.err = err;
  st.tau = tau;
  st.nsamples = g_n;
  return st;
}

} // namespace

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const std::string inputfile = (argc > 1) ? argv[1] : "INPUTH2.dat";
  auto opts = ReadOptionsFromFile(inputfile);

  const bool isospin = std::get<bool>(opts["isospin"]);
  const int isoUp = std::get<int>(opts["iso_up"]);
  const int isoDown = std::get<int>(opts["iso_down"]);
  const int Nup = std::get<int>(opts["spin_up"]);
  const int Ndown = std::get<int>(opts["spin_down"]);
  const double magnetization = std::get<int>(opts["M"]);
  const double traplength = std::get<double>(opts["traplength"]);
  const int nsweeps_tot = std::get<int>(opts["nsweeps"]);
  const int seed = std::get<int>(opts["seed"]) + rank * 1000;

  const int loadtype = std::get<int>(opts["loadtype"]);
  if (loadtype != 2) {
    if (rank == 0)
      std::cerr << "EnergyH2: serve loadtype 2 (pesi completi da restartfn), "
                   "trovato "
                << loadtype << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  const std::string wfile = std::get<std::string>(opts["restart"]);

  // --- funzione d'onda addestrata (nessun update dei parametri) ---
  Nqs wavef(std::get<int>(opts["nv"]), std::get<int>(opts["nh"]),
            std::get<double>(opts["sigma"]), loadtype, isospin, wfile);
  const int nv = wavef.Nspins();
  const int l = static_cast<int>(std::lround(std::cbrt(nv)));
  const double dx = traplength / l;

  ArgonneV6 hamiltonian(nv, magnetization, traplength,
                        std::get<double>(opts["Nprotons"]));
  SamplerH2<Nqs, ArgonneV6> sampler(wavef, hamiltonian, seed, Nup, Ndown, isoUp,
                                    isoDown);

  // sweep di misura per rank; scaldiamo con una frazione extra
  const int nsweeps = std::max(100, nsweeps_tot / std::max(1, size));
  const int movesPerSweep = l * l; // stessa scala del sampler (2 corpi)
  const int ntherm = std::max(500, nsweeps / 5);

  if (rank == 0) {
    std::cout << "# ============ EnergyH2: sola inferenza ============\n";
    std::cout << "# pesi          : " << wfile << "\n";
    std::cout << "# reticolo      : " << l << "^3 = " << nv
              << " siti, traplength = " << traplength << " fm (dx = " << dx
              << " fm)\n";
    std::cout << "# nh            : " << wavef.NHspins()
              << "   npar = " << wavef.GetNpar() << "\n";
    std::cout << "# rank          : " << size << "  x  " << nsweeps
              << " sweep di misura (+" << ntherm << " di termalizzazione)\n";
    std::cout << "# ================================================\n";
    std::flush(std::cout);
  }

  // --- catena Metropolis guidata a mano: nessuna derivata, nessun update ---
  sampler.InitRandomState();
  wavef.InitLt(sampler.GetState());

  for (int t = 0; t < ntherm; ++t)
    for (int k = 0; k < movesPerSweep; ++k)
      sampler.MoveH2(2);

  std::vector<double> eloc;
  std::vector<double> rloc;
  eloc.reserve(nsweeps);
  rloc.reserve(nsweeps);

  for (int n = 0; n < nsweeps; ++n) {
    for (int k = 0; k < movesPerSweep; ++k)
      sampler.MoveH2(2);

    sampler.MeasureEnergy();
    const auto &ens = sampler.GetSamplingEnergies();
    if (!ens.empty())
      eloc.push_back(ens.back().real());

    // distanza protone-neutrone della configurazione corrente
    const auto st = sampler.GetState();
    int p = -1, nn = -1;
    for (size_t i = 0; i < st.size(); ++i)
      if (st[i] == 1) {
        if (static_cast<int>(i) < 2 * nv)
          p = static_cast<int>(i);
        else
          nn = static_cast<int>(i);
      }
    if (p >= 0 && nn >= 0)
      rloc.push_back(hamiltonian.Distance(p % nv, nn % nv) * dx);

    if (rank == 0 && nsweeps >= 10 && (n + 1) % (nsweeps / 10) == 0) {
      std::cout << "#   ... " << (100 * (n + 1)) / nsweeps << "%" << std::endl;
      std::flush(std::cout);
    }
  }

  // --- statistica ---
  const int nblocks = 50;
  BlockStats E = GlobalStats(eloc, nblocks, rank, size);
  BlockStats R = GlobalStats(rloc, nblocks, rank, size);

  double acc_loc = sampler.Acceptance(), acc = 0.0;
  MPI_Allreduce(&acc_loc, &acc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  acc /= static_cast<double>(size);

  if (rank == 0) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n# ---------------- RISULTATO ----------------\n";
    std::cout << "  <E>        = " << E.mean << "  +/-  " << E.err << "  MeV\n";
    std::cout << "  <r_pn>     = " << R.mean << "  +/-  " << R.err << "  fm\n";
    std::cout << "  campioni   = " << E.nsamples << "  (tau ~ " << E.tau
              << " sweep)\n";
    std::cout << "  acceptance = " << acc << "\n";
    std::cout << "# -------------------------------------------\n";
    std::cout << "# CONFRONTO: usare il ground state ESATTO dello stesso\n"
                 "# reticolo (diagH2.cc), non il binding sperimentale.\n";

    std::ofstream out("EnergyH2_result.dat", std::ios::app);
    if (out.is_open()) {
      out << std::setprecision(10) << E.mean << " " << E.err << "   "
          << R.mean << " " << R.err << "   " << E.nsamples << " " << E.tau
          << " " << acc << "   " << wfile << "\n";
      std::cout << "# risultato appeso a EnergyH2_result.dat\n";
    }

    std::ofstream hist("EnergyH2_samples.dat");
    if (hist.is_open()) {
      hist << "# E_loc [MeV] (rank 0)\n";
      for (double v : eloc)
        hist << v << "\n";
    }
  }

  MPI_Finalize();
  return 0;
}