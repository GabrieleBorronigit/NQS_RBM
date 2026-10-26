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
#include <fstream>
#include <getopt.h> //permette l'uso della funzione getopt_long, che analizza gli argomenti della riga di comando
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <variant>

#include "mpi.h"

// Various utilities to read the command line options

// cerca il tipo di modello nel nome del file dalla riga di comando
std::string FindModel(std::string strarg) {
  std::size_t found = strarg.find(
      "Ising"); //.find() restituisce la posizione (quanti caratteri dopo) in
                //cui compare la stringa contenunta in strarg
  if (found != std::string::npos) { // npos viene restituito quando non trova
                                    // quello che cerchi
    return "Ising1d";
  }
  found = strarg.find("Heisenberg1d");
  if (found != std::string::npos) {
    return "Heisenberg1d";
  }
  found = strarg.find("Heisenberg2d");
  if (found != std::string::npos) {
    return "Heisenberg2d";
  }
  found = strarg.find("Trap3d");
  if (found != std::string::npos) {
    return "Trap3d";
  }
  found = strarg.find("Trap1d");
  if (found != std::string::npos) {
    return "Trap1d";
  }
  found = strarg.find("Coulomb1d");
  if (found != std::string::npos) {
    return "Coulomb1d";
  }
  found = strarg.find("Coulomb3d");
  if (found != std::string::npos) {
    return "Coulomb3d";
  }

  return "None";
}

std::string
FindCoupling(std::string strarg) { // salva il terzo argomento nel nome file
                                   // (dopo 2 underscore_) -> coupling constant
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  size_t found = strarg.find("_"); // cerca il primo underscore
  size_t found1 = std::string::npos;
  size_t found2 = std::string::npos;
  if (found != std::string::npos) {
    found1 = strarg.find("_", found + 1); // cerca il secondo underscore
    if (found1 != std::string::npos) {
      found2 = strarg.find("_", found1 + 1); // cerca il terzo underscore
    }
  }
  if (found1 != std::string::npos &&
      found2 != std::string::npos) { // se trova sia il secondo che il terzo
                                     // underscore, salva la stringa tra i due e
                                     // la restituisce
    return (strarg.substr(found1 + 1, found2 - found1 - 1));
  } else {
    if (rank == 0) {
      std::cerr << "# Error : the filename is not in the format specified for "
                   "the Ising/Heisenberg model"
                << std::endl;
    }
    std::abort();
  }
  std::exit(0);
  return "error";
}

// stampa intenstazione informativa sul programma
void PrintHeader() {

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::cout << std::endl;
    std::cout << "\t|   Neural-network quantum states sampler   |" << std::endl;
    std::cout << "\t| written by Giuseppe Carleo, December 2016 |" << std::endl
              << std::endl;
  }
}

// stampa informazioni su come usare il programma
void PrintInfoMessage() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rank == 0) {
    std::cout << "Usage : ./nqs_run OPTIONS" << std::endl << std::endl;

    std::cout << "Allowed OPTIONS are : " << std::endl << std::endl;

    std::cout << "--filename=...  " << std::endl;
    std::cout << "\tname of the file containing neural-network weights"
              << std::endl;
    std::cout << "\t(chose one in directories Ground/ or Unitary/)" << std::endl
              << std::endl;

    std::cout << "--nsweeps=... " << std::endl;
    std::cout << "\tnumber of Monte Carlo sweeps" << std::endl;
    std::cout << "\t(default value is 1.0e4)" << std::endl << std::endl;

    std::cout << "--seed=... " << std::endl;
    std::cout << "\tinteger seed for pseudo-random numbers" << std::endl;
    std::cout << "\tseed<0 sets it to the internal clock value" << std::endl;
    std::cout << "\t(default value is -1)" << std::endl << std::endl;

    std::cout << "--filestates=... " << std::endl;
    std::cout << "\tname of the file to print sampled configurations"
              << std::endl;
    std::cout << "\t(by default it is not set)" << std::endl << std::endl;

    std::cout << "--spin_up=... " << std::endl;
    std::cout << "\tnumber of spin-up states" << std::endl;
    std::cout << "\t(required option)" << std::endl << std::endl;

    std::cout << "--spin_down=... " << std::endl;
    std::cout << "\tnumber of spin-down states" << std::endl;
    std::cout << "\t(required option)" << std::endl << std::endl;
  }
}

// stampa l'intestazione e definice una mappa per salvare le opzioni
std::map<std::string, std::string> ReadOptions(int argc, char *argv[]) {

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  PrintHeader();

  std::map<std::string, std::string> options;

  // se c'è solo il nome del programma, termina dando info
  if (argc == 1) {
    PrintInfoMessage();
    std::exit(0);
  }

  while (1) {
    static struct option long_options[] = {
        /* These options don’t set a flag.
           We distinguish them by their indices. */
        {"filename", required_argument, 0,
         'a'}, // required argument sarebbe che nella linea di codice troverà
               // filename=pippo
        {"nsweeps", required_argument, 0, 'b'},
        {"seed", required_argument, 0, 'c'},
        {"filestates", required_argument, 0, 'd'},
        {"spin_up", required_argument, 0, 'e'},
        {"spin_down", required_argument, 0, 'f'},
        {0, 0, 0, 0} // segnala la fine delle opzioni
    };

    /* getopt_long stores the option index here. */
    int option_index = 0;

    int c = getopt_long(
        argc, argv,
        "a:b:c:d:e:f:", // se getopt_long trova fra le argv uno della struct
                        // option restituisce il carattere a,b,c o d, altrimenti
                        // restituisce -1
        long_options, &option_index);

    /* Detect the end of the options. */
    if (c == -1)
      break;

    switch (c) {
    case 'a':
      options["filename"] = optarg; // si salva l'argomento dell'opzione (il
                                    // nome del file o il numero di sweeps)
      break;

    case 'b':
      options["nsweeps"] = optarg;
      break;

    case 'c':
      options["seed"] = optarg;
      break;

    case 'd':
      options["filestates"] = optarg;
      break;

    case 'e':
      options["spin_up"] = optarg;
      break;
    case 'f':
      options["spin_down"] = optarg;
      break;

    case '?':
      PrintInfoMessage();
      break;

    default:
      std::abort();
    }
  }

  if (options.count("filename") ==
      0) { // la stringa col filename è obbligatoria, se non c'è scrivo
           // messaggio di errore
    if (rank == 0) {
      std::cerr << "# Error: Option filename must be specified with the option "
                   "--filename=FILENAME"
                << std::endl;
    }
    std::abort();
  }

  if (options.count("nsweeps") == 0) {
    options["nsweeps"] = "10000";
  }

  if (options.count("seed") == 0) {
    options["seed"] = "-1";
  }

  if (options.count("spin_up") == 0 || options.count("spin_down") == 0) {
    if (rank == 0) {
      std::cerr << "# Error: Options spin_up and spin_down must be specified!"
                << std::endl;
    }
    std::abort();
  }

  int magnetization = std::stoi(
      FindCoupling(options["filename"])); // ma allora il secondo numero è la
                                          // magnetizzazione o no???
  int spin_up = std::stoi(options["spin_up"]);
  int spin_down = std::stoi(options["spin_down"]);

  if (magnetization != (spin_up + spin_down)) {
    if (rank == 0) {
      std::cerr << "# Error: Magnetization does not match spin up and spin "
                   "down values!"
                << std::endl;
    }
    std::abort();
  }
  if (rank == 0)
    std::cout << "# Magnetization verified successfully!" << std::endl;

  options["model"] =
      FindModel(options["filename"]); // determina il modello fisico in base al
                                      // nome del file

  if (options["model"] == "Ising1d") {
    options["hfield"] = FindCoupling(
        options["filename"]); // estraggo un parametro dal nome del file
  }

  if (options["model"] == "Heisenberg1d") {
    options["jz"] = FindCoupling(options["filename"]);
  }

  if (options["model"] == "Heisenberg2d") {
    options["jz"] = FindCoupling(options["filename"]);
  }

  if (options["model"] == "Trap3d") {
    options["M"] = FindCoupling(options["filename"]);
  }

  if (options["model"] == "Trap1d") {
    options["M"] = FindCoupling(options["filename"]);
  }

  if (options["model"] == "Coulomb1d") {
    options["M"] = FindCoupling(options["filename"]);
  }

  if (options["model"] == "Coulomb3d") {
    options["M"] = FindCoupling(options["filename"]);
  }

  return options;
}

// metodo più semplice che legge tutti gli input da un file preconfezionato
// (G.Borroni)

std::map<std::string, std::variant<int, double, std::string, bool>>
ReadOptionsFromFile(std::string filename) {

  std::ifstream in;
  in.open(filename);

  if (!in) {
    std::cerr << "unable to open file " << filename << std::endl;
    exit(1);
  }
  std::map<std::string, std::variant<int, double, std::string, bool>> options;
  std::string property;

  while (in >> property) {
    if (property == "nv") {
      int nv;
      in >> nv;
      options["nv"] = nv;
    } else if (property == "nh") {
      int nh;
      in >> nh;
      options["nh"] = nh;
    } else if (property == "sigma") {
      double sigma;
      in >> sigma;
      options["sigma"] = sigma;
    } else if (property == "loadtype") {
      int loadtype;
      in >> loadtype;
      options["loadtype"] = loadtype;
    } else if (property == "model") {
      std::string model;
      in >> model;
      options["model"] = model;
    } else if (property == "M") {
      int M;
      in >> M;
      options["M"] = M;
    } else if (property == "spin_up") {
      int up;
      in >> up;
      options["spin_up"] = up;
    } else if (property == "spin_down") {
      int down;
      in >> down;
      options["spin_down"] = down;
    } else if (property == "seed") {
      int seed;
      in >> seed;
      options["seed"] = seed;
    } else if (property == "nsweeps") {
      int nsweeps;
      in >> nsweeps;
      options["nsweeps"] = nsweeps;
    } else if (property == "filestates") {
      std::string filestates;
      in >> filestates;
      options["filestates"] = filestates;
    } else if (property == "printstates") {
      int print;
      bool b = false;
      in >> print;
      if (print == 1)
        b = true;
      options["printstates"] = b;
    } else if (property == "restartfn") {
      std::string fn;
      in >> fn;
      options["restart"] = fn;
    } else if (property == "get_pars") {
      int n;
      in >> n;
      options["get_pars"] = n;
    } else if (property == "niter") {
      int niter;
      in >> niter;
      options["niter"] = niter;
    }

    else if (property == "Nprotons") {
      double n = 0.0;
      in >> n;
      options["Nprotons"] = n;
    }

    else if (property == "traplength") {
      double l = 0;
      in >> l;
      options["traplength"] = l;
    } else if (property == "isospin") {
      int iso;
      in >> iso;
      bool use = false;
      if (iso == 1)
        use = true;
      options["isospin"] = use;
    } else if (property == "iso_up") {
      int u = 0;
      in >> u;
      options["iso_up"] = u;
    } else if (property == "iso_down") {
      int d = 0;
      in >> d;
      options["iso_down"] = d;
    } else if (property == "eta") {
      double eta = 0;
      in >> eta;
      options["eta"] = eta;
    } else if (property == "eta1") {
      double eta1 = 0;
      in >> eta1;
      options["eta1"] = eta1;
    } else if (property == "label") {
      std::string label;
      in >> label;
      options["label"] = label;
    } else if (property == "optimizer") { // "sgd" (default) oppure "cg"
      std::string o;
      in >> o;
      options["optimizer"] = o;
    } else if (property == "cg_lambda") { // damping RELATIVO alla scala di S
      double x = 0;
      in >> x;
      options["cg_lambda"] = x;
    } else if (property == "cg_rtol") { // tolleranza relativa del CG
      double x = 0;
      in >> x;
      options["cg_rtol"] = x;
    } else if (property == "cg_maxiter") { // iterazioni massime del CG
      int n = 0;
      in >> n;
      options["cg_maxiter"] = n;
    } else if (property == "cg_maxstep") { // trust region su |delta|
      double x = 0;
      in >> x;
      options["cg_maxstep"] = x;
    } else if (property == "trainingfn") {
      std::string fn;
      in >> fn;
      options["trainingfn"] = fn;
    } else if (property == "smear_b_factor") { // larghezza regolatore AV6Lattice, in unita' di dx
      double x = 0;
      in >> x;
      options["smear_b_factor"] = x;
    }

    else {
      std::cerr << "Unknown option: " << property << std::endl;
      exit(1);
    }
  }

  in.close();
  return options;
}

// mpic++ -std=c++17 opt.cc -o nqs_run
// mpirun -np 8 ./nqs_run--filename=WfTest/Coulomb3d_2_2_test.wf
// --filestates=test.dat --spin_up=2 --spin_down=0 --nsweeps=160
