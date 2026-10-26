
#include "mpi.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

// Simple Monte Carlo sampling of a spin
// Wave-Function

/*
il codice lo riorganizzo così:
state: vettore di dimensione 4*nv, corrispondenti ai 4 canali attaccati:

[ protone spin su | protone spin giu | neutrone spin su | neutrone spin giu ]
cosi da avere sempre e solo due siti occupati.
in particolare un sito sara sempre nei primi due canali e uno negli ultimi due
per il deuterio

*/

// struct per dichiarare solo 2 collezioni di vettori: una per gli spin e una
// per gli isospin
struct ChannelIndexH2 {
  std::vector<int>
      Up; // stateindexup: lista dei siti occupati nel canale spin up
  std::vector<int>
      Down; // stateindexdown: lista dei siti occupati nel canale spin down
  std::vector<int> All;         // stateindex: [siti spin up | siti spin down]
  std::vector<int> AllUnsorted; // stesso ma non ordinato
};

struct ChannelStateH2 {

  std::vector<int> Up;   // stateup: reticolo nel canale spin up
  std::vector<int> Down; // statedown: reticolo nel canale spin down
};

template <class Wf, class Hamiltonian> class SamplerH2 {

  // wave-function
  Wf &wf_;

  // Hamiltonian
  Hamiltonian &hamiltonian_;

  // dimension of lattice
  // const int l_;

  // number of spins
  const int nspins_;
  int ntotspins_; // nspins*numero di canali (2 senza isospin, 4 con)
  // current state in the sampling

  std::vector<int> state_; // contiene il reticolo intero appiattito

  /* std::vector<int> stateUp_;//contiene il reticolo nel canale up appiattito
   std::vector<int> stateDown_; //reticolo nel down appiattito
   std::vector<int> stateIndex_; //lista degli indici lineari dei siti occupati
   dalle particelle (dimensione=mag) std::vector<int> stateIndexUp_;//contiene
   le posizioni delle particelle spin up (dim=Nup) std::vector<int>
   stateIndexDown_; //contiene le posizioni delle particelle spin down
   (dim=Ndown) std::vector<int> stateIndexUnsorted_; //stateindex, non ordinato
   std::vector<int> stateIsoUp_;
   std::vector<int> stateIsoDown_M;*/

  ChannelIndexH2 SpinIndex;
  ChannelStateH2 SpinState;

  int Nup_;
  int Ndown_;
  bool isospin_;
  int Nisoup_;   // isospin up
  int Nisodown_; // isospin down
  int Mag_;

  // random number generators and distributions
  std::mt19937 gen_;
  std::uniform_real_distribution<> distu_;
  std::uniform_int_distribution<> distn_;
  std::uniform_int_distribution<> distm_;
  std::uniform_int_distribution<> distdir_;

  // sampling statistics
  double accept_;
  double nmoves_;
  double MetroRatio_;
  // mosse Metropolis per sweep. Se <=0 usa il default nspins_*sweepfactor
  // (= l^3*sf). Per un sistema a 2 corpi diluito e' overkill: la decorrelazione
  // scala come la diffusione nella box (~l^2), non col volume (l^3).
  int movesOverride_ = -1;

  // container for indices of randomly chosen spins to be flipped
  std::vector<int> flips_;
  std::vector<int> mirrorflips_;      // da usare nel deuterio col com fissato
  std::vector<int> mirrorflipsRight_; // idem
  std::vector<int> flipsRight_;
  bool spinUp_ = true;
  bool isospinUp_ = true;

  // option to write the sampled configuration on a file
  bool writestates_;
  std::ofstream filestates_;

  // quantities needed by the hamiltonian
  // non-zero matrix elements
  std::vector<std::complex<double>> mel_;

  // flip connectors for the hamiltonian(see below for details)
  std::vector<std::vector<int>> flipsh_;

  // storage for measured values of the energy
  std::vector<std::complex<double>> energy_;

  // storage for sampled positions in MC
  std::vector<std::vector<int>> stateIndexHistory_;
  std::vector<std::vector<int>> stateHistory_;

  // storage for sampled logarithmic derivatives (usato solo se storeFull_)
  std::vector<std::vector<std::complex<double>>> derLogs_;
  // accumulatori on-the-fly per il gradiente SGD: evitano di tenere in memoria
  // l'intera matrice nsweeps x npar (che a nh=50, l=9 sono gia' ~GB per rank).
  //   sumDer_[j]  = sum_k  O_k[j]
  //   sumEDer_[j] = sum_k  Re(E_k) O_k[j]
  // con O_k = DerLog(stato k). Il gradiente si ricostruisce esattamente:
  //   grad[j] = 2/N ( conj(sumEDer_[j]) - <E> conj(sumDer_[j]) )
  std::vector<std::complex<double>> sumDer_;
  std::vector<std::complex<double>> sumEDer_;
  long nAccum_ = 0;
  bool storeFull_ = false; // true solo per il pretraining (serve per-campione)
  int nv_;
  int l_; // lunghezza del lato della box
  // accumulator for mean nucleon-nucleon radius (local to this MPI rank)

public:
  SamplerH2(Wf &wf, Hamiltonian &hamiltonian, int seed, int Nup, int Ndown,
            int isoup, int isodown)
      : wf_(wf), hamiltonian_(hamiltonian), distu_(0, 1), nspins_(wf.Nspins()),
        Mag_(hamiltonian.getMagnetizazion()), Nup_(Nup), Ndown_(Ndown),
        Nisoup_(isoup), Nisodown_(isodown), isospin_(wf_.Isospin()),
        distn_(0, nspins_ - 1), distm_(0, hamiltonian.getMagnetizazion() - 1),
        distdir_(0, hamiltonian.getDimensionSpace() * 2 - 1) {

    ntotspins_ = 4 * nspins_;
    writestates_ = true;
    nv_ = wf_.Nspins();
    l_ = std::cbrt(nv_);
    Seed(seed);
    ResetAv();
  }

  ~SamplerH2() {
    if (writestates_) {
      filestates_.close();
    }
  }
  // Uniform random number in [0,1)
  inline double Uniform() { return distu_(gen_); }

  inline void Seed(int seed) {
    if (seed < 0) {
      gen_.seed(std::time(nullptr));
    } else {
      gen_.seed(seed);
    }
  }

  void InitRandomState() {
    // siti: [ protone spin su | protone spin giu | neutrone spin su | neutrone
    // spin giu ]
    state_.assign(ntotspins_, -1); // riempie state con tutti -1
    // SpinIndex.Up.resize(Nup_);
    // SpinIndex.Down.resize(Ndown_);
    SpinIndex.All.resize(Mag_);
    // nel deuterio il centro di massa è fissato al centro della box.
    // devo generare i due nucleoni in modo tale che la retta che li congiunge
    // passi proprio per il COM, il move è poi fatto in modo da mantenere questa
    // simmetria per tutta la simulazione per un cubo di lato l, la condizione
    // che deve verificarsi è r1+r2=l-1 dove r sono le posizioni

    int c = (l_ - 1) / 2;        // centro box
    int axis = distn_(gen_) % 3; // 0:x, 1:y, 2:z
    int sign = (distn_(gen_) % 2) ? +1 : -1;

    // distanza iniziale casuale in [1, c]
    int dr = 1 + distn_(gen_) % std::max(1, c);

    int xp = c, yp = c, zp = c;
    if (axis == 0)
      xp = c + sign * dr;
    else if (axis == 1)
      yp = c + sign * dr;
    else
      zp = c + sign * dr;

    // riflessione per mantenere COM
    int xn = (l_ - 1) - xp;
    int yn = (l_ - 1) - yp;
    int zn = (l_ - 1) - zp;

    // coordinate a indice di reticolo
    int p = zp * l_ * l_ + yp * l_ + xp; // protone
    int n = zn * l_ * l_ + yn * l_ + xn; // neutrone (offset)

    // aggiungo offset di 1/2/3nv per decidere dove metterli come spin/isospin
    if (Nup_ == 2) { // due spin up
      // p non cambia, il primo canale è gia protone spin up
      n += 2 * nv_; // neutrone spin up
                    // SpinIndex.Up[0] = p;
                    // SpinIndex.Up[1] = n;
      SpinIndex.All[0] = p;
      SpinIndex.All[1] = n;
    } else if (Nup_ == 1) { // uno spin up ed uno spin down
      // coin flip per scegliere se protone spin up o neutrone spin up
      if (Uniform() >=
          0.5) { // protone spin up (ancora non devo fare niente su p)

        n += 3 * nv_; // neutrone spin down
        SpinIndex.All[0] = p;
        SpinIndex.All[1] = n;
        //  SpinIndex.Up[0] = p;
        // SpinIndex.Down[0] = n;

      } else { // neutrone su e protone giu
        p += nv_;
        n += 2 * nv_;
        SpinIndex.All[0] = p;
        SpinIndex.All[1] = n;
        // SpinIndex.Up[0] = n;
        // SpinIndex.Down[0] = p;
      }
      // SpinIndex.All = SpinIndex.Up;
      // SpinIndex.All.insert(SpinIndex.All.end(), SpinIndex.Down.begin(),
      //  SpinIndex.Down.end());
      // SpinIndex.AllUnsorted = SpinIndex.All;
      // std::sort(SpinIndex.All.begin(), SpinIndex.All.end());
    } else if (Nup_ == 0) { // due spin down
      p += nv_;
      n += 3 * nv_;
      SpinIndex.All[0] = p;
      SpinIndex.All[1] = n;
      // SpinIndex.Down[0] = p;
      // SpinIndex.Down[1] = n;
      // SpinIndex.All = SpinIndex.Down;
    }
    state_[p] = 1;
    state_[n] = 1;
    assert(SpinIndex.All[0] < 2 * nv_ && SpinIndex.All[0] >= 0);
    assert(SpinIndex.All[1] >= 2 * nv_ && SpinIndex.All[1] < state_.size());
    // assert(SpinIndex.Up[0] < nv_ || SpinIndex.Up[0] >= 2 * nv_);
    // std::cout << "INIT TEST------------SpinIndex.All.size()=" <<
    // SpinIndex.All.size() << std::endl;
    for (int i = 0; i < SpinIndex.All.size(); ++i) {
      //    std::cout << "All["<<i<<"]="<<SpinIndex.All[i]<<std::endl;
      assert(state_[SpinIndex.All[i]] == 1);
    }

    // std::cout<<"------------DEBUG
    // INITRANDOMSTATEH2()------------"<<std::endl;
    // std::cout//<<"state_.size()="<<state_.size()<<std::endl<<"SpinIndex.All.size()="<<SpinIndex.All.size()<<std::endl

    //<<"spinindexall:"<<SpinIndex.All[0]<<" , "<<SpinIndex.All[1]<<"\n
    //distance="<<hamiltonian_.Distance(SpinIndex.All[0]%nv_,SpinIndex.All[1]%nv_)*dx_<<std::endl;
  }

  inline int OppositeDirection(int dir) {
    return (dir % 2 == 0) ? dir + 1 : dir - 1;
    // le direzioni sono definite a coppie opposte: 0=x 1=-x 2=y 3=-y ecc
  }

  // propone la mossa, ritornando vero o falso se si puo fare o meno
  // funziona solo per il deuterio, per N nucleoni bisognerebbe
  // implementare il com fisso traslando tutte le particelle ogni volta
  // magari la prossima volta :)
  inline bool RandspinH2(std::vector<int> &flips, std::vector<int> &mirrorflips,
                         int nflips) {
    // MetroRatio_ = 1.0;
    flips.resize(nflips);
    mirrorflips.resize(nflips);
    // scelgo una delle due particelle da flippare per prima
    int part = (Uniform() >= 0.5 ? 0 : 1);
    flips[0] = SpinIndex.All[part];
    mirrorflips[0] = SpinIndex.All[1 - part];

    // check sui vicini
    std::vector<int> nn = hamiltonian_.FindNeighboursGlobal(flips[0], nv_);
    std::vector<int> mirrornn =
        hamiltonian_.FindNeighboursGlobal(mirrorflips[0], nv_);

    // scegli direzione a caso
    int direz = distdir_(gen_) % nn.size();
    int mirrordir = OppositeDirection(direz);

    flips[1] = nn[direz];
    mirrorflips[1] = mirrornn[mirrordir];

    if (flips[1] < 0 || mirrorflips[1] < 0 || flips[1] >= state_.size() ||
        mirrorflips[1] >= state_.size())
      return false;

    return true;

    // if (state_[flips[1]] == +1 || state_[mirrorflips[1]] == +1)
    // return false;
  }

  void MoveH2(int nflips) {
    // std::cout << "DEBUG MoveH2 start" << std::endl;
    // std::cout << "state_.size() = " << state_.size() << std::endl;
    // std::cout << "nv_ = " << nv_ << std::endl;

    // std::cout << "SpinIndex.All.size()=" << SpinIndex.All.size() <<
    // std::endl;
    for (int i = 0; i < SpinIndex.All.size(); ++i) {
      // std::cout << "All["<<i<<"]="<<SpinIndex.All[i]<<std::endl;
      assert(state_[SpinIndex.All[i]] == 1);
    }

    flipsRight_.resize(
        nflips); // vettore che salva i flips proposti da randspin (flips[0] ha
                 // il sito di partenza flip[1] quello di arrivo)
    // diagnostic flag: did the current proposal reduce inter-particle distance?
    mirrorflipsRight_.resize(nflips); // idem
    double r = Uniform();
    std::vector<int> move;
    double acceptance = 0;
    // a volte non muovo ma inverto gli spin, a volte gli isospin

    if (r <= 0.2) {
      move = FlipRandomSpin();
      // std::cout<<"proposto flip degli spin"<<std::endl;
      //  std::cout<<"calcolando accettanza..."<<std::endl;
      acceptance = std::norm(wf_.PoP(state_, move));
      // std::cout<<"accettanza calcolata..."<<std::endl;
    }

    else if (r > 0.20 && r <= 0.4) {
      move = FlipPNH2();
      if (move.empty()) {
        // proposta identica allo stato corrente: non contare la mossa
        return;
      }
      // std::cout<<"proposto flip degli isospin"<<std::endl;
      //   std::cout<<"calcolando accettanza..."<<std::endl;
      acceptance = std::norm(wf_.PoP(state_, move));
      // std::cout<<"accettanza calcolata..."<<std::endl;
    }

    else if (RandspinH2(flips_, mirrorflips_, nflips)) {
      move = flips_;
      move.insert(move.end(), mirrorflips_.begin(), mirrorflips_.end());
      // count proposal type
      // std::cout << "Mossa da eseguire: ";
      // for (int i = 0; i < move.size(); i++) {
      // std::cout << move[i] << std::endl;
      // }
      // std::cout<<"calcolando accettanza..."<<std::endl;
      // MetroRatio_ stores q(new->old)/q(old->new), so the Metropolis-Hastings
      // acceptance multiplies |psi'|^2/|psi|^2 by this factor. Using the
      // inverse would bias the walk and violate detailed balance.
      auto pop = wf_.PoP(state_, move);
      acceptance = std::norm(pop);
      // std::cout<<"acc="<<acceptance<<"\t
      // metroratio="<<MetroRatio_<<std::endl;
      if (acceptance > 1)
        acceptance = 1;
      // std::cout<<"accettanza calcolata..."<<std::endl;
    } else {
      acceptance = 0;
      // std::cout<<"randspin falso, nessuna mossa eseguita"<<std::endl;
    }
    //   std::cout << "Mossa da eseguire: ";
    // for (int i = 0; i < move.size(); i++) {
    //   std::cout << move[i] << std::endl;
    // }
    // valuto l'accettanza sul movimento di entrambi
    for (auto f : move) {
      if (f < 0 || f >= state_.size()) {
        std::cerr << "❌ invalid flip index " << f
                  << " (state size = " << state_.size() << ")\n";
        return;
      }
    }

    if (acceptance > Uniform()) {
      SpinIndex.All.clear();

      // aggiorno le Lt se accettato
      wf_.UpdateLt(state_, move);

      // aggiorno lo stato
      for (int i = 0; i < move.size(); i += 2) {
        int from = move[i];
        int to = move[i + 1];
        if (from < 0 || to < 0 || from >= state_.size() ||
            to >= state_.size()) {
          std::cerr << "Indice fuori range in aggiornamento stato\n";
          exit(1);
        }
        if (from < 0 || to < 0 || from >= state_.size() ||
            to >= state_.size()) {
          std::cerr << "Indice fuori range in aggiornamento stato\n";
          exit(1);
        }
        if (state_[to] == +1) {
          std::cerr
              << "move ha generato un indice su di un sito gia occupato \n";
          exit(1);
        }

        state_[from] = -1; // disattiva / spin down / vecchio sito
        state_[to] = +1;   // attiva / spin up / nuovo sito
        // std::cout << "sito disattivato:" << from << " -> attivato:" << to <<
        // std::endl;
      }
      int count{};
      for (int i = 0; i < state_.size(); i++) {
        if (state_[i] == 1) {
          SpinIndex.All.push_back(i);
          count++;
        }
      }
      assert(count == 2);

      // aggiorno gli spinindex, la cosa più semplice è resettare direttamente
      // tutto...

      // SpinIndex.All.clear();
      // SpinIndex.Up.clear();
      // SpinIndex.Down.clear();
      /*  for (int i = 0; i < state_.size(); i++) {

          if (state_[i] == 1) { // sito occupato
            SpinIndex.All.push_back(i);
           // if (CheckSpinUp(i))
            //  SpinIndex.Up.push_back(i);
           // else
          //    SpinIndex.Down.push_back(i);
          }
        }*/
      // riordino gli indici

      // SpinIndex.AllUnsorted = SpinIndex.All;
      std::sort(SpinIndex.All.begin(), SpinIndex.All.end());
      // std::cout<<"rab="<<hamiltonian_.Distance(SpinIndex.All[0]%nv_,
      // SpinIndex.All[1]%nv_)<<std::endl;
      if (SpinIndex.All.size() != 2) {
        std::cerr << "⚠️ Incoerenza nel rebuild: trovati "
                  << SpinIndex.All.size() << " nucleoni attivi!" << std::endl;

        // scan di debug
        for (int i = 0; i < state_.size(); i++) {
          if (state_[i] == 1)
            std::cerr << "occupato: " << i << std::endl;
        }

        exit(1);
      }

      // aggiorno accettanza
      accept_++;
    }
    // if proposal reduced r and was accepted, increment accepted_reduce counter
    // stampe di debug
    //     std::cout << state_.size() << std::endl;
    //   std::cout << "spin up in " << SpinIndex.Up[0] << std::endl;
    //    std::cout << "spin down in " << SpinIndex.Down[0] << std::endl;
    nmoves_++;
  }

  // ritorna true se il sito passato contiene uno spin up, false altrimenti (!
  // presuppone che il sito sia occupato)
  inline bool CheckSpinUp(int site) {
    if ((site >= 0 && site < nv_) || (site >= 2 * nv_ && site < 3 * nv_))
      return true;
    else
      return false;
  }
  std::vector<int> FlipSpin(int isospin) {
    std::vector<int> move;
    move.clear();
    std::sort(SpinIndex.All.begin(), SpinIndex.All.end());
    int site = SpinIndex.All[isospin];
    move.push_back(site);
    if (CheckSpinUp(site)) // up-->down
      move.push_back(site + nv_);
    else // down->up
      move.push_back(site - nv_);
    return move;
  }

  /*
  std::sort(SpinIndex.All.begin(), SpinIndex.All.end());
int site=SpinIndex.All[isospin];
int backup= SpinIndex.All[1-isospin];
SpinIndex.All.clear();
SpinIndex.Up.clear();
SpinIndex.Down.clear();
if(CheckSpinUp(site)) //il nucleone selezionato è spin su--> spin giu (traslo di
nv) SpinIndex.Up.push_back(site+nv_); else //down--> up (-nv)
SpinIndex.Down.push_back(site-nv_);

if(CheckSpinUp(backup))
SpinIndex.Up.push_back(backup);
else
SpinIndex.Down.push_back(backup);

SpinIndex.All=SpinIndex.Up;
SpinIndex.All.insert(SpinIndex.All.end(), SpinIndex.Down.begin(),
SpinIndex.Down.end()); SpinIndex.AllUnsorted=SpinIndex.All;
std::sort(SpinIndex.All.begin(), SpinIndex.All.end());


          for(int i=0; i<2; i++){
    if(SpinIndex.All[i]<0 ||SpinIndex.All[i]>state_.size()){
      std::cout<<"fuori range dopo spin flip"<<std::endl;
      exit(1);
    }
}*/

  std::vector<int> FlipRandomSpin() {
    // prende un protone/neutrone e gli cambia spin
    // coin flip per scegliere se flippare lo spin del protone o del neutrone
    std::vector<int> move;
    double r = Uniform();
    if (r > 0.5) // flippo protone
      move = FlipSpin(
          0); // in 0 ci sono i protoni per l ordinamento di spinindexall
    else
      move = FlipSpin(1); // flip neutrone
    return move;
  }

  // scambia protone e neutrone (flippa gli isospin)
  std::vector<int> FlipPNH2() {
    // fondamentalmente devo traslare di 2*nv entrambi
    std::vector<int> move;
    move.clear();

    // ancora spinindexall è ordinato, quindi il primo indice è per forza un
    // protone e il secondo un nutrone
    std::sort(SpinIndex.All.begin(), SpinIndex.All.end());
    // std::cout<<"---------------PRIMA DEL FLIP
    // PN--------------spinindexall=("<<SpinIndex.All[0]<<" ,
    // "<<SpinIndex.All[1]<<")"<<std::endl;

    int p = SpinIndex.All[0];
    int n = SpinIndex.All[1];
    if ((p < 2 * nv_ && p + 2 * nv_ < state_.size()) &&
        (n >= 2 * nv_ && n - 2 * nv_ >= 0) && state_[p + 2 * nv_] != 1 &&
        state_[n - 2 * nv_] != 1) {
      move.push_back(p);
      move.push_back(p + 2 * nv_);
      move.push_back(n);
      move.push_back(n - 2 * nv_);
    } else if (state_[p + 2 * nv_] == 1 &&
               state_[n - 2 * nv_] ==
                   1) { // in questo caso il flip è inutile, ritorno mossa vuota
      move.clear();
    } else {
      std::cerr << "FlipPNH2 incoerente: p=" << p << " n=" << n << std::endl;
      exit(1);
    }

    return move;
  }
  /*
      SpinIndex.All.clear();
      //sposto il protone
      state_[p]=-1;
      state_[p+2*nv_]=+1;
      SpinIndex.All.push_back((p+2*nv_));

      //sposto il neutrone
      state_[n]=-1;
      state_[n-(2 * nv_)]=+1;
      SpinIndex.All.push_back(n-(2*nv_));
      //rifaccio gli indici
   SpinIndex.Up.clear();
   SpinIndex.Down.clear();
        for(int i=0; i<2 ; i++){
          if(CheckSpinUp(SpinIndex.All[i]))
          SpinIndex.Up.push_back(SpinIndex.All[i]);
          else
          SpinIndex.Down.push_back(SpinIndex.All[i]);
        }

    SpinIndex.AllUnsorted=SpinIndex.All;
    std::sort(SpinIndex.All.begin(), SpinIndex.All.end());
    for(int i=0; i<2; i++){

      if(SpinIndex.All[i]<0 ||SpinIndex.All[i]>state_.size()){
        std::cout<<"fuori range dopo isospin flip"<<std::endl;
        std::cout<<"spinindexall=("<<SpinIndex.All[0]<<" ,
   "<<SpinIndex.All[1]<<")"<<std::endl; exit(1);
      }
    }
    std::cout<<"swap! ("<<p<<","<<n<<")---->("<<SpinIndex.All[0]<<" ,
   "<<SpinIndex.All[1]<<")"<<std::endl;
      }


  */
  void Plot(double nsweeps, double thermfactor = 0.1, int sweepfactor = 1) {
    std::ofstream en("Energy.dat");
    std::ofstream out("positionsH2.dat");

    int nflips = hamiltonian_.MinFlips();
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Svuoto memorie precedenti
    stateHistory_.clear();
    stateIndexHistory_.clear();
    energy_.clear();
    derLogs_.clear();

    InitRandomState();
    wf_.InitLt(state_);
    ResetAv();

    int movesPerMCStep = nspins_ * sweepfactor;
    // thermalization
    for (double n = 0; n < nsweeps * thermfactor; n++) {
      for (int k = 0; k < movesPerMCStep; k++) {
        MoveH2(nflips);
      }
    }
    ResetAv();

    // sampling
    for (double n = 0; n < nsweeps; n++) {

      for (int k = 0; k < movesPerMCStep; k++) {
        MoveH2(nflips);
      }

      RecordPosition();
      MeasureEnergy();
      en << energy_.back() << std::endl;
      if (CheckSpinUp(SpinIndex.All[0]))
        out << "protonUp \t" << SpinIndex.All[0] << std::endl;
      else
        out << "protonDown \t" << SpinIndex.All[0] << std::endl;
      if (CheckSpinUp(SpinIndex.All[1]))
        out << "NeutronUp \t" << SpinIndex.All[1] << std::endl;
      else
        out << "NeutronDown \t" << SpinIndex.All[1] << std::endl;

      std::cout << "sweep " << n << " completed" << std::endl;
    }
    OutputEnergy();
    en.close();
    out.close();
  }

  void realPart(const std::vector<std::complex<double>> &x,
                std::vector<double> &y) {
    for (int i = 0; i < x.size(); i++) {
      y[i] = x[i].real();
    }
  }

  void SetFileStates(
      std::string filename) { // predispone file per salvare la configurazione
                              // attuale su file esterno
    writestates_ = true;
    filestates_.open(filename.c_str());
    if (!filestates_.is_open()) {
      std::cerr << "# Error : Cannot open file " << filename << " for writing"
                << std::endl;
      std::abort();
    } else {
      std::cout << "# Saving sampled configuration to file " << filename
                << std::endl;
    }
  }

  void WriteState() { // stampa su file esterno valori e posizioni correnti
                      // degli spin, magnetizzazione, accettanza
    int Mag = 0;
    for (const auto &spin_value : state_) {
      filestates_ << std::setw(2) << spin_value << " ";
      Mag += spin_value;
    }
    filestates_ << std::setw(2) << "Pos= ";
    for (const auto &spin_pos : SpinIndex.All) {
      filestates_ << std::setw(2) << spin_pos << ",";
    }
    Mag = (nspins_ + Mag) / 2;
    filestates_ << std::setw(2) << "Mag= " << Mag << ", ";
    filestates_ << std::setw(2) << "Acceptance= " << Acceptance() << " ";

    filestates_ << std::endl;
  }

  // Measuring the value of the local energy
  // on the current state
  void MeasureEnergy() {
    std::complex<double> en = 0.;

    // Finds the non-zero matrix elements of the hamiltonian
    // on the given state
    // i.e. all the state' such that <state'|H|state> = mel(state') \neq 0
    // state' is encoded as the sequence of spin flips to be performed on state

    // SpinIndex.AllUnsorted = SpinIndex.Up;
    // SpinIndex.AllUnsorted.insert(SpinIndex.AllUnsorted.end(),
    //                            SpinIndex.Down.begin(), SpinIndex.Down.end());
    std::sort(SpinIndex.All.begin(), SpinIndex.All.end());
    hamiltonian_.FindConn(state_, SpinIndex.All, flipsh_, mel_, Nup_, nv_);
    /*std::cout << "\n--- DEBUG MeasureEnergy ---\n";
     std::cout << "flipsh_.size() = " << flipsh_.size()
               << ", mel_.size() = " << mel_.size()
               << ", state_.size() = " << state_.size() << std::endl;

     for (int i = 0; i < flipsh_.size(); i++) {
       std::cout << "flip[" << i << "] = ";
       for (auto f : flipsh_[i]) {
         std::cout << f << " ";
         if (f < 0 || f >= state_.size()) {
           std::cout << "❌(out of range)";
         }
       }
       std::cout << " | mel = " << mel_[i] << std::endl;
     }
     std::cout << "--- END DEBUG ---\n";
     std::cout << "\n--- DEBUG MeasureEnergy ---\n";
     std::cout << "flipsh_.size() = " << flipsh_.size()
               << ", mel_.size() = " << mel_.size()
               << ", state_.size() = " << state_.size() << std::endl;

     for (int i = 0; i < flipsh_.size(); i++) {
       std::cout << "flip[" << i << "] = ";
       for (auto f : flipsh_[i]) {
         std::cout << f << " ";
         if (f < 0 || f >= state_.size()) {
           std::cout << "❌(out of range)";
         }
       }
       std::cout << " | mel = " << mel_[i] << std::endl;
     }
     std::cout << "--- END DEBUG ---\n";


 std::cout << "flipsh_.size() = " << flipsh_.size() << std::endl;
 */
    // std::cout << "state_.size() = " << state_.size() << std::endl;
    // std::cout << "state hash = " << std::accumulate(state_.begin(),
    // state_.end(), 0) << std::endl;

    for (int i = 0; i < flipsh_.size(); ++i) {

      auto flips_copy = flipsh_[i];
      for (int f : flips_copy) {
        if (f < 0 || f >= state_.size()) {
          std::cerr << "❌ Flip index out of range: " << f
                    << ", state size = " << state_.size() << std::endl;
          std::exit(1);
        }
      }
      auto pop = wf_.PoP(state_, flips_copy);
      if (std::isnan(pop.real()) || std::isnan(pop.imag())) {
        std::cerr << "❌ PoP returned NaN at i = " << i << std::endl;
        std::cerr << "flips_copy = ";
        for (int f : flips_copy)
          std::cerr << f << " ";
        std::cerr << std::endl;
        std::exit(1);
      }
      en += pop * mel_[i];
    }

    if (std::isnan(en.real()) || std::isnan(en.imag())) {
      std::cerr << "❌ eloc è NaN al sample " << std::endl;

      std::exit(1);
    }
    energy_.push_back(en);
    mel_.clear();
    flipsh_.clear();

    // std::cout<<"vivo E="<<en<<std::endl;
  }
  void RecordPosition() { // salva lo stat//o corrende del reticolo nei vettori
                          // di history
    stateIndexHistory_.push_back(SpinIndex.All);
    stateHistory_.push_back(state_);
  }

  void CalcDerlogs() {
    auto O = wf_.DerLog(state_);
    const int npar = O.size();
    // energia locale dell'ultimo campione (parte reale, come in Variational)
    // E_loc COMPLESSA: per parametri olomorfi il gradiente e' <O* (E_loc-<E>)>
    // e scartare Im(E_loc) rimuove una componente reale del gradiente.
    const std::complex<double> e = energy_.back();

    if (sumDer_.empty()) {
      sumDer_.assign(npar, {0., 0.});
      sumEDer_.assign(npar, {0., 0.});
    }
    for (int j = 0; j < npar; ++j) {
      sumDer_[j] += O[j];
      sumEDer_[j] += e * std::conj(O[j]);
    }
    ++nAccum_;

    // per il pretraining serve la matrice completa per-campione
    if (storeFull_)
      derLogs_.push_back(std::move(O));
  }

  // Run the Monte Carlo sampling
  // nsweeps is the total number of sweeps to be done
  // thermfactor is the fraction of nsweeps to be discarded during the initial
  // equilibration sweepfactor set the number of single spin flips per sweep to
  // nspins*sweepfactor nflipss is the number of random spin flips to be done,
  // it is automatically set to 1 or 2 depending on the hamiltonian
  void Run(double nsweeps, double thermfactor = 0.1, int sweepfactor = 1,
           int nflipss = -1) {

    // nsweeps da options

    int nflips = nflipss;

    if (nflips == -1) {
      nflips = hamiltonian_.MinFlips(); // fissato a 2 da hamiltonian
    }

    // checking input consistency
    if (nflips > 2 || nflips < 1) {
      std::cerr
          << "# Error : The number of spin flips should be equal to 1 or 2.";
      std::cerr << std::endl;
      std::abort();
    }
    if (thermfactor > 1 || thermfactor < 0) {
      std::cerr << "# Error : The thermalization factor should be a real "
                   "number between 0 and 1";
      std::cerr << std::endl;
      std::abort();
    }
    if (nsweeps < 50) {
      std::cerr << "# Error : Please enter a number of sweeps sufficiently "
                   "large (>50)";
      std::cerr << std::endl;
      std::abort();
    }

    std::cout << "# Starting Monte Carlo sampling" << std::endl;
    std::cout << "# Number of sweeps to be performed is " << nsweeps
              << std::endl;

    InitRandomState();
    std::cout << "check" << std::endl;

    if (writestates_) {
      WriteState();
    }

    flips_.resize(nflips);
    std::cout << "check1" << std::endl;
    // initializing look-up tables in the wave-function
    wf_.InitLt(state_);
    std::cout << "check2" << std::endl;
    ResetAv();

    std::cout << "# Thermalization... ";
    std::flush(std::cout);

    // int movesPerMCStep = 320; //
    int movesPerMCStep = nspins_ * sweepfactor;

    // thermalization
    for (double n = 0; n < nsweeps * thermfactor; n += 1) {

      for (int i = 0; i < movesPerMCStep; i++) {
        MoveH2(nflips);
        if (writestates_) {
          WriteState();
        }
      }
    }
    std::cout << " DONE " << std::endl;
    std::flush(std::cout);

    ResetAv();

    std::cout << "# Sweeping... ";
    std::flush(std::cout);

    // sequence of sweeps
    for (double n = 0; n < nsweeps; n += 1) {
      for (int i = 0; i < movesPerMCStep; i++) { // nspins_*sweepfactor
        MoveH2(nflips);
      }
      if (writestates_) {
        WriteState();
      }
      MeasureEnergy();
      // RecordPosition();
      // record radius sample for diagnostics
    }

    std::cout << " DONE " << std::endl;
    std::flush(std::cout);

    OutputEnergy();
    OutputEnergyValues();
    OutputPositions();
  }

  void Sample(double nsweeps, double thermfactor = 0.6, int sweepfactor = 1) {

    int nflips =
        hamiltonian_.MinFlips(); // minflips è il numero minimo di spin da
                                 // flippare per lasciare la magnetizzazione
                                 // invariata (per coulomb3d è 2)
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // svuoto tutte le memorie di campionamenti precedenti
    stateHistory_.clear();
    stateIndexHistory_.clear();
    energy_.clear();
    derLogs_.clear();
    sumDer_.clear();
    sumEDer_.clear();
    nAccum_ = 0;
    // SpinIndex.All.clear();
    // SpinIndex.Up.clear();
    // SpinIndex.Down.clear();
    InitRandomState(); // inizializza il reticolo con Nup spin up ed Ndown
                       // spin down, tutti gli altri posti non occupati hanno
                       // assegnato -1

    wf_.InitLt(state_); // inizializza lookup tables(somme sui nodi nascosti)

    ResetAv(); // azzera accumulatori di accettanza metropolis
    // Default: l^2*sweepfactor. Per un deutone (2 corpi) la decorrelazione
    // scala come la diffusione nella box (~l^2), non col volume (l^3): validato
    // a pesi fissi, stessa <E> e stesso tau (~0.5 sweep) di l^3, con l x meno
    // lavoro. movesOverride_>0 forza un valore (es. l^3 per confronto).
    int movesPerMCStep =
        (movesOverride_ > 0) ? movesOverride_ : l_ * l_ * sweepfactor;

    // thermalization

    for (double n = 0; n < nsweeps * thermfactor; n += 1) {
      for (int i = 0; i < movesPerMCStep; i++) {
        MoveH2(nflips);
      }
    }
    // std::cout<<"fine termalizzazione-------------------"<<std::endl;
    ResetAv();
    // sequence of sweeps
    for (double n = 0; n < nsweeps; n += 1) {
      for (int i = 0; i < movesPerMCStep; i++) { // nspins_*sweepfactor
        // std::cout<<"------------- Sample sta per chiamare Move: giro numero
        // "<<i<<std::endl;
        MoveH2(nflips); // metropolis, flippa un numero casuale di spin
      }

      if (rank == 0) {
        if (writestates_) {
          WriteState(); // stampa su file esterno valori e posizioni correnti
                        // degli spin, magnetizzazione, accettanza
        }
      }
      if (storeFull_)
        RecordPosition(); // history dei campioni: serve solo al pretraining,
                          // in training e' peso morto (nsweeps x 4nv int)
      MeasureEnergy(); // calcola Eloc (da aggiornare per i nuclei!!)
      // std::cout << "DEBUG: indirizzo flipsh_[0] = " << &flipsh_[0] <<
      // std::endl;

      // std::cout<<"dopo di measure energy so vivo"<<std::endl;
      CalcDerlogs(); // calcola le derivate logaritmiche
                     //  std::cout<<"dopo calcderlogs so vivo"<<std::endl;
      //  std::cout << "DEBUG: derLogs_.size() = " << derLogs_.size() <<
      //  std::endl; if (!derLogs_.empty()) {
      //   std::cout << "DEBUG: derLogs_[0].size() = " << derLogs_[0].size()
      //           << std::endl;
      //  }
    }
  }

  void ResetAv() {
    accept_ = 0;
    nmoves_ = 0;
  }

  std::vector<int> GetState() const { return state_; }

  std::vector<std::vector<int>> GetSamplingConfig() { return stateHistory_; }

  std::vector<std::vector<int>> GetSamplingIndex() {
    return stateIndexHistory_;
  }

  std::vector<std::complex<double>> GetSamplingEnergies() { return energy_; }

  // accumulatori del gradiente (ritorno per riferimento: nessuna copia)
  const std::vector<std::complex<double>> &GetSumDer() const { return sumDer_; }
  const std::vector<std::complex<double>> &GetSumEDer() const {
    return sumEDer_;
  }
  long GetNAccum() const { return nAccum_; }
  void SetStoreFull(bool b) { storeFull_ = b; } // attivato dal pretraining
  void SetMovesPerSweep(int m) { movesOverride_ = m; } // <=0 => default l^3*sf
  int GetL() const { return l_; }

  std::vector<std::vector<std::complex<double>>> GetDerLogs() const {
    return derLogs_;
  }

  Wf &GetNqs() { return wf_; }

  Hamiltonian &GetHamiltonian() { return hamiltonian_; }

  inline double Acceptance() const { // variabile di controllo a posteriori: se
                                     // è molto bassa qualcosa non va
    return accept_ / nmoves_;
  }

  void OutputPositions() {
    std::ofstream out;
    out.open("SamplingTest/positions.dat", std::ios_base::out);
    if (!out.is_open()) {
      std::cerr << "# Error : Cannot open file for writing position values"
                << std::endl;
      std::abort();
    }
    out << "position" << std::endl;
    for (const auto &config : stateIndexHistory_) {
      for (const auto &pos : config) {
        out << pos << " ";
      }
      out << std::endl;
    }

    out.close();
    std::cout << "done writing positions" << std::endl;
  }

  void OutputEnergyValues() {
    std::ofstream out;
    out.open("SamplingTest/EnergyValues.dat", std::ios_base::out);
    if (!out.is_open()) {
      std::cerr << "# Error : Cannot open file for writing energy values"
                << std::endl;
      std::abort();
    }
    out << "energy" << std::endl;
    for (const auto &energy : energy_) {
      out << energy.real() << std::endl;
    }

    out.close();
    std::cout << "done writing energy values" << std::endl;
  }

  void OutputEnergy() {
    int nblocks = 50;

    int blocksize = std::floor(double(energy_.size()) / double(nblocks));

    double enmean = 0;
    double enmeansq = 0;

    double enmean_unblocked = 0;
    double enmeansq_unblocked = 0;

    for (int i = 0; i < nblocks; i++) {
      double eblock = 0;
      for (int j = i * blocksize; j < (i + 1) * blocksize; j++) {
        eblock += energy_[j].real();
        assert(j < energy_.size());

        double delta = energy_[j].real() - enmean_unblocked;
        enmean_unblocked += delta / double(j + 1);
        double delta2 = energy_[j].real() - enmean_unblocked;
        enmeansq_unblocked += delta * delta2;
      }
      eblock /= double(blocksize);
      double delta = eblock - enmean;
      enmean += delta / double(i + 1);
      double delta2 = eblock - enmean;
      enmeansq += delta * delta2;
    }

    enmeansq /= (double(nblocks - 1)); // diviso n-1 ?
    enmeansq_unblocked /= (double((nblocks * blocksize - 1)));

    double estav = enmean; // estimated average
    double esterror = std::sqrt(enmeansq / double(nblocks)); // estimated error

    int ndigits = std::log10(esterror);
    if (ndigits < 0) {
      ndigits = -ndigits + 2;
    } else {
      ndigits = 0;
    }

    std::cout << "# Estimated average energy : " << std::endl;
    std::cout << "# " << std::scientific << std::setprecision(ndigits) << estav;
    std::cout << " +/-  " << std::setprecision(0) << esterror << std::endl;
    std::cout << "# Error estimated with binning analysis consisting of ";
    std::cout << nblocks << " bins " << std::endl;
    std::cout << "# Block size is " << blocksize << std::endl;
    std::cout << "# Estimated autocorrelation time is ";
    std::cout << std::setprecision(0);
    std::cout << 0.5 * double(blocksize) * enmeansq / enmeansq_unblocked
              << std::endl;
  }
};
