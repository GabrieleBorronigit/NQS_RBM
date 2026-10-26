/*
############################ COPYRIGHT NOTICE ##################################

Code provided by G. Carleo and M. Troyer, written by G. Carleo, December 2016.

Permission is granted for anyone to copy, use, modify, or distribute the
accompanying programs and documents for any purpose, provided this copyright
notice is retained and prominently displayed, along with a complete citation of
the published version of the paper:
 ______________________________________________________________________________
| G. Carleo, and M. Troyer                                                     |
| Solving the quantum many-body problem with artificial neural-networks        |
|______________________________________________________________________________|

The programs and documents are distributed without any warranty, express or
implied.

These programs were written for research purposes only, and are meant to
demonstrate and reproduce the main results obtained in the paper.

All use of these programs is entirely at the user's own risk.

################################################################################
*/


#include <vector>
#include <random>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ctime>
#include <algorithm>

//Simple Monte Carlo sampling of a spin
//Wave-Function
template<class Wf,class Hamiltonian> class Sampler{

  //wave-function
  Wf & wf_;

  //Hamiltonian
  Hamiltonian & hamiltonian_;
  
  //dimension of lattice
  //const int l_;

  //number of spins
  const int nspins_;

  //current state in the sampling
  std::vector<int> state_;
  std::vector<int> stateUp_;
  std::vector<int> stateDown_;
  std::vector<int> stateIndex_;
  std::vector<int> stateIndexUp_;
  std::vector<int> stateIndexDown_;
  std::vector<int> stateIndexUnsorted_;

  int Nup_;
  int Ndown_;
  int Mag_;

  //random number generators and distributions
  std::mt19937 gen_;
  std::uniform_real_distribution<> distu_;
  std::uniform_int_distribution<> distn_;
  std::uniform_int_distribution<> distm_;
  std::uniform_int_distribution<> distdir_;

  //sampling statistics
  double accept_;
  double nmoves_;

  //container for indices of randomly chosen spins to be flipped
  std::vector<int> flips_;
  std::vector<int> flipsRight_;
  bool spinUp_=true;

  //option to write the sampled configuration on a file
  bool writestates_;
  std::ofstream filestates_;

  //quantities needed by the hamiltonian
  //non-zero matrix elements
  std::vector<std::complex<double> > mel_;

  //flip connectors for the hamiltonian(see below for details)
  std::vector<std::vector<int> > flipsh_;

  //storage for measured values of the energy
  std::vector<std::complex<double> > energy_;

  //storage for sampled positions in MC
  std::vector<std::vector<int> > stateIndexHistory_;
  std::vector<std::vector<int> > stateHistory_;

  //storage for sampled logarithmic derivatives
  std::vector<std::vector<std::complex<double>>> derLogs_;

public:

  Sampler(Wf & wf, Hamiltonian & hamiltonian, int seed, int Nup, int Ndown):
          wf_(wf),hamiltonian_(hamiltonian),distu_(0,1),nspins_(wf.Nspins()),Mag_(hamiltonian.getMagnetizazion()),
	  Nup_(Nup),Ndown_(Ndown),distn_(0,nspins_-1), distm_(0,hamiltonian.getMagnetizazion()-1), 
	  distdir_(0,hamiltonian.getDimensionSpace()*2 - 1){
    writestates_=true;
    Seed(seed);
    ResetAv();
  }

  ~Sampler(){
    if(writestates_){
      filestates_.close();
    }
  }

  //Uniform random number in [0,1)
  inline double Uniform(){
    return distu_(gen_);
  }

  inline void Seed(int seed){
    if(seed<0){
      gen_.seed(std::time(nullptr));
    }
    else{
      gen_.seed(seed);
    }
  }

  //Random spin flips (max 2 spin flips in this implementation)
  // si basa su STATEINDEX
  inline bool RandSpin(std::vector<int> & flips,int nflips){
    flips.resize(nflips);
        
    std::vector<int> nn;
    nn.resize(hamiltonian_.getDimensionSpace()*2);
    int part = distm_(gen_);				//random number in [0, Mag_-1]
    //std::cout<<part<<std::endl;
    int direz = distdir_(gen_); //sceglie direzione random
    flips[0]= stateIndex_[part]; //mette il primo flip proposto (su un sito random) in posizione 0 di flips
    
    nn= hamiltonian_.FindNeighbours(flips[0]); //controlla se i vicini del sito estratto random sono accessibili (non occupati o non al bordo della box)
    if(flips[0]<wf_.Nspins()){				//se true è up
    //se il vicino up è occupato ->-1
    //std::cout<<"UP"<<std::endl;
      spinUp_=true;
      for(int k=0; k<stateIndexUp_.size(); k++){
        for(int j=0; j<nn.size(); j++){
          if(nn[j]==stateIndexUp_[k]){
            nn[j]=-1;
          }
        }	 
      }
    }
    else { 						//else è down
    //se il vicino down è occupato ->-1
    //std::cout<<"DOWN"<<std::endl;
      spinUp_=false;
      for(int k=0; k<stateIndexDown_.size(); k++){
        for(int j=0; j<nn.size(); j++){
          if(nn[j]==stateIndexDown_[k]){
            nn[j]=-1;
          }
        }
      }  
    }   
    
    flips[1]=nn[direz];

    //boundary? si vanno fatte perchè poi flips va in wf.PoP. -> in hamiltonian non bastano

    if(flips[1]<0){
      //std::cout<<"check"<<std::endl;
      return false;
    }

    return true;
  }

  //Initializes a random spin state
  void InitRandomState(){

    state_.resize(nspins_*2);
    stateUp_.resize(nspins_);
    stateDown_.resize(nspins_);
    stateIndex_.resize(Mag_);
    stateIndexUp_.resize(Nup_);
    stateIndexDown_.resize(Ndown_);

      for(int i=0;i<nspins_;i++){	//inizializzo il (doppio)lattice ponendo tutti gli slot a -1
	      stateUp_[i]=-1;
        stateDown_[i]=-1;
      }

      for(int i=0;i<Nup_;i++){	//riempio il lattice up estraendo random Nup_ particelle. Pongo = 1 gli slot occupati
	  bool nuovaPos = false;
	  while(!nuovaPos){
	    stateIndexUp_[i] = distn_(gen_);	//random number in [0,nspins_-1]
	    nuovaPos = true;
	    for(int j=0;j<i;j++){
	      if(stateIndexUp_[i]==stateIndexUp_[j]){
	      nuovaPos = false;
	      }
	    }
	  }
         /*std::cout<<" spin up "<<i<<std::endl;
         std::cout<<stateIndex_[i]<<std::endl;*/
         stateUp_[stateIndexUp_[i]]=1;
      }

      for(int i=0;i<Ndown_;i++){//riempio il lattice down estraendo random Ndown_ particelle. Pongo = 1 gli slot occupati
          bool nuovaPos = false;
	  while(!nuovaPos){
	    stateIndexDown_[i] = distn_(gen_);	//random number in [0,nspins_-1]
	    nuovaPos = true;
	    for(int j=0;j<i;j++){
	      if(stateIndexDown_[i]==stateIndexDown_[j]){
	      nuovaPos = false;
	      }
	    }
	  }
          /*std::cout<<" spin down "<<i<<std::endl;
          std::cout<<stateIndex_[i]<<std::endl;*/
	  stateDown_[stateIndexDown_[i]]=1;
        }

      state_=stateUp_;
      state_.insert(state_.end(),stateDown_.begin(),stateDown_.end());
      stateIndex_=stateIndexUp_;
      stateIndex_.insert(stateIndex_.end(),stateIndexDown_.begin(),stateIndexDown_.end());

      std::sort(stateIndex_.begin(),stateIndex_.end());
	/*int j=0;
	for(int i=0;i<nspins_;i++){ 	//riordina stateIndex_ sort(stateIndex)???
	  if(state_[i]==1){
	    stateIndex_[j]=i;
	    j+=1;
	  }
	}*/
  }

  void ResetAv(){
    accept_=0;
    nmoves_=0;
  }

  std::vector<int> GetState(){
    return state_;
  }

  std::vector<std::vector<int> > GetSamplingConfig(){
    return stateHistory_;
  }

  std::vector<std::vector<int> > GetSamplingIndex(){
    return stateIndexHistory_;
  }

  std::vector<std::complex<double> > GetSamplingEnergies(){
    return energy_;
  }

  std::vector<std::vector<std::complex<double>>> GetDerLogs(){
    return derLogs_;
  }

  Wf & GetNqs(){
      return wf_;
  }

  Hamiltonian & GetHamiltonian(){
      return hamiltonian_;
  }

  inline double Acceptance()const{ //credo sia una variabile di controllo a posteriori: se è molto bassa qualcosa non va
    return accept_/nmoves_;
  }

  void Move(int nflips){


    flipsRight_.resize(nflips);
    //Picking "nflips" random spins to be flipped
    if(RandSpin(flips_,nflips)){
      if(spinUp_) {
        flipsRight_=flips_;
        //std::cout<<"flipsRight_ Vector:"<<std::endl;
        //StampaVector(flipsRight_);
      }
      else {
        flipsRight_={flips_[0]+nspins_,flips_[1]+nspins_};
        //std::cout<<"flipsRight_ Vector:"<<std::endl;
        //StampaVector(flipsRight_);
      }
      //Computing acceptance probability
      double acceptance=std::norm(wf_.PoP(state_,flipsRight_)); //USA IL MODULO QUADRO PER IL SAMPLING! :)
      //double acceptance = 1.;

      //Metropolis-Hastings test
      if(acceptance>Uniform()){

        //Updating look-up tables in the wave-function
        wf_.UpdateLt(state_,flipsRight_);

        //Moving to the new configuration
        for(const auto& flip : flipsRight_){
          state_[flip]*=-1;
        }
		
        //update di stateIndex_
        if(spinUp_) {
          for(int i=0;i<Nup_;i++){
            if(stateIndexUp_[i]==flips_[0]){
              stateIndexUp_[i]=flips_[1];
            }
          }
        }
        else {
          for(int i=0;i<Ndown_;i++){
            if(stateIndexDown_[i]==flips_[0]){
              stateIndexDown_[i]=flips_[1];
            }
          }
        }

        stateIndex_=stateIndexUp_;
        stateIndex_.insert(stateIndex_.end(),stateIndexDown_.begin(),stateIndexDown_.end());
        std::sort(stateIndex_.begin(),stateIndex_.end());

        /*std::cout<<"stateIndex_ : "<<std::endl;
        StampaVector(stateIndex_);
        std::cout<<"state_ : "<<std::endl;
        StampaVector(state_);*/
        accept_+=1;
      } 
    }

    nmoves_+=1;
  }

  void SetFileStates(std::string filename){ //predispone file per salvare la configurazione attuale su file esterno
    writestates_=true;
    filestates_.open(filename.c_str());
    if(!filestates_.is_open()){
      std::cerr<<"# Error : Cannot open file "<<filename<<" for writing"<<std::endl;
      std::abort();
    }
    else{
      std::cout<<"# Saving sampled configuration to file "<<filename<<std::endl;
    }
  }

  void WriteState(){ //stampa su file esterno valori e posizioni correnti degli spin, magnetizzazione, accettanza
    int Mag = 0;
    for(const auto & spin_value : state_){
      filestates_<<std::setw(2)<<spin_value<<" ";
      Mag += spin_value;
    }
    filestates_<<std::setw(2)<<"Pos= ";
	  for(const auto & spin_pos : stateIndex_){
      filestates_<<std::setw(2)<<spin_pos<<",";
    }
    Mag= (nspins_+Mag) /2;
    filestates_<<std::setw(2)<<"Mag= "<<Mag<<", ";
    filestates_<<std::setw(2)<<"Acceptance= "<<Acceptance()<<" ";

    filestates_<<std::endl;
  }

  //Measuring the value of the local energy
  //on the current state
  void MeasureEnergy(){
    std::complex<double> en=0.;

    //Finds the non-zero matrix elements of the hamiltonian
    //on the given state
    //i.e. all the state' such that <state'|H|state> = mel(state') \neq 0
    //state' is encoded as the sequence of spin flips to be performed on state

    stateIndexUnsorted_=stateIndexUp_;
    stateIndexUnsorted_.insert(stateIndexUnsorted_.end(),stateIndexDown_.begin(),stateIndexDown_.end());

    /*for(int i=0; i<stateIndexUnsorted_.size(); i++){
      std::cout<<stateIndexUnsorted_[i]<<std::endl;
    }*/

    hamiltonian_.FindConn(stateIndexUnsorted_,flipsh_,mel_,Nup_);
 
    for(int i=0;i<flipsh_.size();i++){

      //std::cout<<mel_[i]<<std::endl;
      en+=wf_.PoP(state_,flipsh_[i])*mel_[i];
      //std::cout<<wf_.PoP(state_,flipsh_[i])<<std::endl;
    }
    std::ofstream out;
    out.open("energyimag.dat",std::ios::app);
    out<<en.imag()<<std::endl;
    energy_.push_back(en);  //perchè è complessa e poi usa solo la parte reale??
  }

  void RecordPosition(){ //salva lo stato corrende del reticolo nei vettori di history
    stateIndexHistory_.push_back(stateIndex_);
    stateHistory_.push_back(state_);
  }

  void CalcDerlogs(){
    derLogs_.push_back(wf_.DerLog(state_)); //matrice che contiene tutte i vettori derlogs per ogni stato
  }

  //Run the Monte Carlo sampling
  //nsweeps is the total number of sweeps to be done
  //thermfactor is the fraction of nsweeps to be discarded during the initial equilibration
  //sweepfactor set the number of single spin flips per sweep to nspins*sweepfactor
  //nflipss is the number of random spin flips to be done, it is automatically set to 1 or 2 depending on the hamiltonian
  void Run(double nsweeps,double thermfactor=0.1,int sweepfactor=1,int nflipss=-1){

	//nsweeps da options

    int nflips=nflipss;

    if(nflips==-1){
      nflips=hamiltonian_.MinFlips(); //fissato a 2 da hamiltonian
    }

    //checking input consistency
    if(nflips>2 || nflips<1){
      std::cerr<<"# Error : The number of spin flips should be equal to 1 or 2.";
      std::cerr<<std::endl;
      std::abort();
    }
    if(thermfactor>1 || thermfactor<0){
      std::cerr<<"# Error : The thermalization factor should be a real number between 0 and 1";
      std::cerr<<std::endl;
      std::abort();
    }
    if(nsweeps<50){
      std::cerr<<"# Error : Please enter a number of sweeps sufficiently large (>50)";
      std::cerr<<std::endl;
      std::abort();
    }

    std::cout<<"# Starting Monte Carlo sampling"<<std::endl;
    std::cout<<"# Number of sweeps to be performed is "<<nsweeps<<std::endl;

    InitRandomState();
    std::cout<<"check"<<std::endl;

	if(writestates_){
        WriteState();
    }

    flips_.resize(nflips);
    std::cout<<"check1"<<std::endl;
    //initializing look-up tables in the wave-function
    wf_.InitLt(state_);
    std::cout<<"check2"<<std::endl;
    ResetAv();

    std::cout<<"# Thermalization... ";
    std::flush(std::cout);

    //int movesPerMCStep = 320; //
    int movesPerMCStep = nspins_*sweepfactor;

    //thermalization
    for(double n=0;n<nsweeps*thermfactor;n+=1){
      //std::cout<<"check2"<<std::endl;
      for(int i=0;i<movesPerMCStep;i++){
        Move(nflips);
	  if(writestates_){		//??????????????????
            WriteState();
          }
      }
    }
    std::cout<<" DONE "<<std::endl;
    std::flush(std::cout);

    ResetAv();

    std::cout<<"# Sweeping... ";
    std::flush(std::cout);

    //sequence of sweeps
    for(double n=0;n<nsweeps;n+=1){
      for(int i=0;i<movesPerMCStep;i++){ //nspins_*sweepfactor
        Move(nflips);
      }
      if(writestates_){
        WriteState();
      }
      MeasureEnergy();
      RecordPosition();

    }
    std::cout<<" DONE "<<std::endl;
    std::flush(std::cout);

    OutputEnergy();
    OutputEnergyValues();
    OutputPositions();
  }

  void Sample(double nsweeps,double thermfactor=0.1,int sweepfactor=1){

    int nflips=hamiltonian_.MinFlips(); //minflips è il numero minimo di spin da flippare per lasciare la magnetizzazione invariata (per coulomb3d è 2)
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    //svuoto tutte le memorie di campionamenti precedenti
    stateHistory_.clear();
    stateIndexHistory_.clear();
    energy_.clear();
    derLogs_.clear();
    InitRandomState(); //inizializza il reticolo con Nup spin up ed Ndown spin down, tutti gli altri posti non occupati hanno assegnato -1
    wf_.InitLt(state_); //inizializza lookup tables(somme sui nodi nascosti)
    ResetAv(); //azzera accumulatori di accettanza metropolis
    //int movesPerMCStep = 320; //
    int movesPerMCStep = nspins_*sweepfactor;

    //thermalization
    for(double n=0;n<nsweeps*thermfactor;n+=1){
      for(int i=0;i<movesPerMCStep;i++){
        Move(nflips);
      }
    }
    ResetAv();
    //sequence of sweeps
    for(double n=0;n<nsweeps;n+=1){
      for(int i=0;i<movesPerMCStep;i++){ //nspins_*sweepfactor
      
	  Move(nflips); //metropolis, flippa un numero casuale di spin generando nuove config campionate come psi^2
    }
         if(rank==0){
    if(writestates_){
    WriteState(); //stampa su file esterno valori e posizioni correnti degli spin, magnetizzazione, accettanza
    }

      }
      RecordPosition(); //salva lo stato corrende del reticolo nei vettori di history
      MeasureEnergy(); //calcola Eloc
      CalcDerlogs(); //calcola le derivate logaritmiche
    }
     
  }

  void OutputPositions(){
    std::ofstream out;
  	out.open("SamplingTest/positions.dat", std::ios_base::out);
  	if(!out.is_open()){
        std::cerr<<"# Error : Cannot open file for writing position values"<<std::endl;
        std::abort();
      }
    out<<"position"<<std::endl;
    for(const auto& config : stateIndexHistory_){
      for(const auto& pos : config){
        out<<pos<<" ";
      }
      out<<std::endl;
    }

  	out.close();
  	std::cout<<"done writing positions"<<std::endl;
  }

  void OutputEnergyValues(){
    std::ofstream out;
  	out.open("SamplingTest/EnergyValues.dat", std::ios_base::out);
  	if(!out.is_open()){
        std::cerr<<"# Error : Cannot open file for writing energy values"<<std::endl;
        std::abort();
      }
    out<<"energy"<<std::endl;
    for(const auto& energy : energy_){
      out<<energy.real()<<std::endl;
    }

  	out.close();
  	std::cout<<"done writing energy values"<<std::endl;
  }

  void OutputEnergy(){
    int nblocks=50;

    int blocksize=std::floor(double(energy_.size())/double(nblocks));

    double enmean=0;
    double enmeansq=0;

    double enmean_unblocked=0;
    double enmeansq_unblocked=0;

    for(int i=0;i<nblocks;i++){
      double eblock=0;
      for(int j=i*blocksize;j<(i+1)*blocksize;j++){
        eblock+=energy_[j].real();
        assert(j<energy_.size());

        double delta=energy_[j].real()-enmean_unblocked;
        enmean_unblocked+=delta/double(j+1);
        double delta2=energy_[j].real()-enmean_unblocked;
        enmeansq_unblocked+=delta*delta2;
      }
      eblock/=double(blocksize);
      double delta=eblock-enmean;
      enmean+=delta/double(i+1);
      double delta2=eblock-enmean;
      enmeansq+=delta*delta2;
    }

    enmeansq/=(double(nblocks-1)); //diviso n-1 ?
    enmeansq_unblocked/=(double((nblocks*blocksize-1)));

    double estav=enmean; //estimated average
    double esterror=std::sqrt(enmeansq/double(nblocks)); //estimated error

    int ndigits=std::log10(esterror);
    if(ndigits<0){
      ndigits=-ndigits+2;
    }
    else{
      ndigits=0;
    }

    std::cout<<"# Estimated average energy : "<<std::endl;
    std::cout<<"# "<<std::scientific<<std::setprecision(ndigits)<<estav;
    std::cout<<" +/-  "<<std::setprecision(0)<<esterror<<std::endl;
    std::cout<<"# Error estimated with binning analysis consisting of ";
    std::cout<<nblocks<<" bins "<<std::endl;
    std::cout<<"# Block size is "<<blocksize<<std::endl;
    std::cout<<"# Estimated autocorrelation time is ";
    std::cout<<std::setprecision(0);
    std::cout<<0.5*double(blocksize)*enmeansq/enmeansq_unblocked<<std::endl;
  }
 /*//Small functions to set up the lattice
 //Horizontal Pbc
 int PbcH(int nn,int s){
    if(s%l_==0 && nn==(s-1))
      return (s+l_-1);
    else if((s+1)%l_==0 && nn==(s+1))
      return (s-l_+1);
    else
      return nn;
  }

  //Depht (towards and back) Pbc
  int PbcI(int nn){
    if(nn>=nspins_)
      return (nn-nspins_);
    else if(nn<0)
      return (nspins_+nn);
    else
      return nn;
  }

  //Vertical Pbc
  int PbcV(int nn, int s){
    if(nn>=nspins_)	
      return (nn-l_*l_);
    else if(nn<0)
      return (l_*l_+nn);
    else if( int(nn/(l_*l_))==int(s/(l_*l_)) )
      return nn;
    else if(nn>s)
      return nn-l_*l_;
    else 
      return nn+l_*l_;
  }

   //Horizontal without pbc
   int H(int nn,int s){
    if(s%l_==0 && nn==(s-1))
      return (-1);
    else if((s+1)%l_==0 && nn==(s+1))
      return (-1);
    else
      return nn;
  }


  //Depht (towards and back) without pbc
  int I(int nn){
    if(nn>=nspins_)
      return -1;
    else if(nn<0)
      return -1;
    else
      return nn;
  }

  //Vertical without pbc
  int V(int nn, int s){
    if(nn>=nspins_)
      return -1;
    else if(nn<0)
      return -1;
    else if( int(nn/(l_*l_))==int(s/(l_*l_)) )
      return nn;
    else 
      return (-1);
  }*/
 void Plot(double nsweeps, double thermfactor = 0.1, int sweepfactor = 1) {
    std::ofstream en("Energy.dat");
    std::ofstream out("positions.dat");

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
        Move(nflips);
      }
    }
    ResetAv();

    // sampling
    for (double n = 0; n < nsweeps; n++) {

      for (int k = 0; k < movesPerMCStep; k++) {
        Move(nflips);
      }

      RecordPosition();
      MeasureEnergy();
      en << energy_.back() << std::endl;

      for (int pos : stateIndexUp_)
        out << "electronSpinUp " << pos << std::endl;

      for (int pos : stateIndexDown_)
        out << "electronSpinDown " << pos << std::endl;

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
  void OutputLattice(int i) { // per visualizzare su ovito, attenzione:con tanti
                              // sweep riempie la memoria
    std::ofstream out("Config/Config" + std::to_string(i) + ".xyz");
    int l = 11;
    int nv = 1331;
    out << "4" << std::endl; // particles
    out << "frame " << i << std::endl;
    out << "Nucleus" << " " << ((nv - 1) / 2) % l << " "
        << int(((nv - 1) / 2) / l) % l << " " << int(((nv - 1) / 2) / (l * l))
        << std::endl;
    for (const auto &pos : stateIndexUp_) {
      out << "electronSpinUp" << " " << pos % l << " " << int(pos / l) % l
          << " " << int(pos / (l * l)) << std::endl;
    }
    for (const auto &pos : stateIndexDown_) {
      out << "electronSpinDown" << " " << pos % l << " " << int(pos / l) % l
          << " " << int(pos / (l * l)) << std::endl;
    }
  }

  void outputXYZ(std::ofstream &out) {
    int l = 11;
    int nv = 1331;
    for (const auto &pos : stateIndexUp_) {
      out << "electronSpinUp" << " " << pos % l << " " << int(pos / l) % l
          << " " << int(pos / (l * l)) << std::endl;
    }
    for (const auto &pos : stateIndexDown_) {
      out << "electronSpinDown" << " " << pos % l << " " << int(pos / l) % l
          << " " << int(pos / (l * l)) << std::endl;
    }
  }



};
