/*
############################ COPYRIGHT NOTICE ##################################

Code provided by L.Lazzarino, written by L.Lazzarino , October 2021.

Permission is granted for anyone to copy, use, modify, or distribute the
accompanying programs and documents for any purpose, provided this copyright
notice is retained and prominently displayed, along with a complete citation of
the published version of the paper:
 ______________________________________________________________________________
| L.Lazzarino                                                                  |
| Solving the quantum many-body problem with artificial neural-networks        |
|______________________________________________________________________________|

The programs and documents are distributed without any warranty, express or
implied.

These programs were written for research purposes only, and are meant to
demonstrate and reproduce the main results obtained in the paper.

All use of these programs is entirely at the user's own risk.

################################################################################
*/

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

//Transverse-field Ising model in 1d
class Coulomb1d{
  
  //dimension of space
  int dim_=1;
  //number of spins
  const int nspins_;		//es. 10

  //number of particles
  const double Mag_;		//es. 2

  int connectedcomponents_;

  //width of the trap
  const double traplength_;

  //constants
  const double hbar_;
  const double mass_;
  double KEconst_;

  //pre-computed quantities
  std::vector<std::complex<double> > mel_;
  std::vector<std::vector<int> > flipsh_;
  //std::vector<std::complex<double> > boundCondMel_;
  std::vector<std::vector<int>> nn_;//near neighbours

  double alpha_;	//coupling constant Ve-e interaction


public:

  Coulomb1d(int nspins,int Mag, double traplength=1.,double hbar=1., double mass=1.):nspins_(nspins), Mag_(Mag),
	traplength_(traplength), hbar_(hbar), mass_(mass){
    connectedcomponents_ = 1+2*Mag_;		//es. 5
    KEconst_= - std::pow(hbar_,2) / 2 / mass_ / std::pow(traplength_/nspins_,2);
    Init();
  }

  int getMagnetizazion(){
	return Mag_;
  }

  void Init(){ //parte indep dalla configurazione  *pow(1.602176634e-19,2)

    alpha_=10.;

    mel_.resize(connectedcomponents_);		//es {0,0,0,0,0}
    flipsh_.resize(connectedcomponents_);	//es {0,0,0,0,0}

    //no spins flipped = no particle moves:
    mel_[0]= KEconst_ * (-2) * Mag_;
    flipsh_[0]= std::vector<int>();

    for(int i=1;i<connectedcomponents_;i++){
      mel_[i]= KEconst_ * (1);
    }										////es C{-4,1,1,1,1}

    /*boundCondMel_.resize(connectedcomponents_-1); //considera boundary diversi il muro e un'altra particella	//es {0,0}
    boundCondMel_[0]= KEconst_ * (-1);			//es C{-1,0}
    boundCondMel_[connectedcomponents_-2]= KEconst_ * (-1);	//es C{-1,-1}
    for(int n=1;n<connectedcomponents_-2;n++){
      boundCondMel_[n] = 0.;
    }*/	

    std::cout<<"# Coulomb potential in a trap for "<<Mag_<<" particles in one dimension with alpha = "<<alpha_<<std::endl;
  }//es. mel_= C{-4,1,1,1,1}	flipsh={vector,0,0,0,0}	boundCondMel_=C{-2,1/3,1/3,-2}

  //Finds the non-zero matrix elements of the hamiltonian
  //on the given state
  //i.e. all the state' such that <state'|H|state> = mel(state') \neq 0
  //state' is encoded as the sequence of spin flips to be performed on state
  void FindConn(const std::vector<int> & stateIndex,std::vector<std::vector<int> > & flipsh,std::vector<std::complex<double> > & mel, 
		int Nup){

    mel.resize(connectedcomponents_);
    flipsh.resize(connectedcomponents_);

    //assigning pre-computed matrix elements
    mel=mel_;
    flipsh=flipsh_;
    nn_.resize(Mag_);

    //Coulomb potential dependent on particle stateIndex
    for(int i=0;i<Mag_;i++){
      for(int j=i+1;j<Mag_;j++){
        double rab = abs(stateIndex[j]-stateIndex[i])/double(nspins_);	//traplenght=1. -> dx=1/nspins -> rab= distance*dx
        mel[0] += alpha_/rab;
      }
    }

    for(int i=0; i<Mag_; i++){
      nn_[i]= FindNeighbours(stateIndex[i]); //find Neighbours of the particle
      if(i<Nup){
        for(int j=0; j<nn_[i].size(); j++){
          for(int k=0; k<Nup; k++){
            if(nn_[i][j]==stateIndex[k]){
              nn_[i][j]=-1;
            }
          }	 
        }
      }
      else {
        for(int j=0; j<nn_[i].size(); j++){
          for(int k=Nup; k<Mag_; k++){
            if(nn_[i][j]==stateIndex[k]){
              nn_[i][j]=-1;
            }
          }	 
        }
      }
      if(stateIndex[i]==0 || stateIndex[i]==nspins_-1){
        mel[0]+= KEconst_ * (-1);
      }
    }

    int index=1;
    for(int i=0; i<Mag_; i++){	  
      for(int k=0; k<2; k++){     // 2= nn_[i].size()
        if(i<Nup){
          if(nn_[i][k]>=0){
	    flipsh[index]=std::vector<int>{stateIndex[i], nn_[i][k]};
          /*std::cout<<stateIndex[i]<<"  "<<vicino<<std::endl;
          std::cout<<nn_[i][k]<<std::endl;*/
            index++;
          }
          else {
            mel[index]=0.;
            flipsh[index]=std::vector<int>();
            index++;
          }
        }
        else {
          if(nn_[i][k]>=0){
	    flipsh[index]=std::vector<int>{stateIndex[i]+nspins_, nn_[i][k]+nspins_};
          /*std::cout<<stateIndex[i]<<"  "<<vicino<<std::endl;
          std::cout<<nn_[i][k]<<std::endl;*/
            index++;
          }
          else {
            mel[index]=0.;
            flipsh[index]=std::vector<int>();
            index++;
          }
        } 
      }
    }
    //implementing boundary conditions by turning mel to 0 and emptying flipsh
    /*std::vector<int> partPosD;
    partPosD.resize(connectedcomponents_-1); //andranno da 0 a 2*Mag_  partPosD={0,0,0,0}
    std::vector<int> boundCond;
    boundCond.resize(connectedcomponents_-1);// boundCond={0,0,0,0}

    boundCond[0]=-1;
    boundCond[connectedcomponents_-2]=nspins_;
    for(int n=1;n<connectedcomponents_-2;n++){
      int v = n/2 + (n%2) * 2 - 1; //particella vicina
      boundCond[n]= stateIndex[v];
    }	//es partPosD={0,0,0,0}	es boundCond={-1,stateIndex[1],stateIndex[1],10}

    for(int n=0;n<connectedcomponents_-1;n++){
      int d = n/2; //raddoppio posizioni particelle	//n=0 n=1 -> d=0  //n=2 n=3 -> d=1
      partPosD[n]=stateIndex[d];

      if(std::abs(partPosD[n]-boundCond[n]) == 1){
        mel[0] += boundCondMel_[n];
        mel[n+1] = 0;
        flipsh[n+1]=std::vector<int>();
      }
      else{
        flipsh[n+1]=std::vector<int>({stateIndex[n/2], stateIndex[n/2] + (n%2) * 2 - 1});
      }
    }*/
  }

  int MinFlips()const{ //mantains magnetization
    return 2;
  }
  
  int getDimensionSpace(){
	return dim_;
  }

  std::vector<int> FindNeighbours(int s){	//PBC if pbc_=true; else spins in a box. If there's not a near slot -1.
    
    int dim=dim_;
    bool pbc_=false;

    std::vector<int> n_;
    n_.resize(dim*2);	//1d -> 2, 2d -> 4, 3d -> 6
  			
    n_[0]=(pbc_)?PbcH(s-1,s):H(s-1,s);			//sx
    n_[1]=(pbc_)?PbcH(s+1,s):H(s+1,s);			//dx
    
    return n_;
  }
  
  //Horizontal without pbc
  int H(int nn,int s){
    if(s==0 && nn==(s-1))
      return (-1);
    else if(s==(nspins_-1) && nn==(s+1))
      return (-1);
    else
      return nn;
  }

  //Small functions to find neighbours
  //Horizontal Pbc
  int PbcH(int nn,int s){
    if (s==0 && nn==(s-1))
      return (nspins_-1);
    else if (s==(nspins_-1) && nn==(s+1))
      return 0;
    else
      return nn;
  }

};


//////////////////////cestino
/*    flipsh.resize(connectedcomponents_);

    //assigning pre-computed matrix elements
    mel=mel_;
  	flipsh=flipsh_;

  	for(int i=0;i<Mag_;i++){ //particella
  		for(int j=0;j<2;j++){ //direzione
  			flipsh[2*i+j+1]=std::vector<int>({stateIndex[i],stateIndex[i]+2*j-1});
  		}
  	}

  	//implementing boundary conditions by turning mel to 0 and emptying flipsh
  	std::vector<int> partPosD;
  	partPosD.resize(connectedcomponents_-1);
  	std::vector<int> boundCond;
  	boundCond.resize(connectedcomponents_-1);

  	for(int i=0;i<Mag_;i++){ //particella
  		for(int j=0;j<2;j++){ //direzione
  			partPosD[2*i+j+1]= stateIndex[i];

  			int posPartVicina = i+2*j-1;

  			if(posPartVicina==-1){
  				boundCond[2*i+j+1]=-1;
  			}
  			else if(posPartVicina==Mag_){
  				boundCond[2*i+j+1]=nspins_;
  			}
  			else{
  				boundCond[2*i+j+1]= stateIndex[posPartVicina];
  			}

  			if(std::abs( partPosD[2*i+j+1]-boundCond[2*i+j+1] ) == 1 ){
  				mel[2*i+j+1] = 0;
          flipsh[2*i+j+1].resize(0);
  			}
  		}
    }*/


    ///////////////////////////////////////
