// Diagonalizzazione esatta dell'hamiltoniana effettiva del deuterio a COM
// fisso, costruita chiamando ArgonneV6::FindConn su TUTTI gli stati della
// varieta' campionata: (posizione protone xp, spin protone, spin neutrone),
// con il neutrone alla posizione speculare xn = nv-1-xp.
// Verifica l'hermiticita' e stampa lo spettro basso + proprieta' del GS.
#include "ArgonneV6.cc"
#include <Eigen/Eigenvalues>
#include <cassert>
#include <iostream>
#include <vector>

using Eigen::MatrixXcd;

void runCase(int l, double tl) {
  //def box
  const int nv = l * l * l;
  //hamiltoniana
  ArgonneV6 H(nv, 2, tl);
  //numero di siti
  const int nst = 4 * nv; // xp * (sp,sn)
  //passo reticolare
  const double dx = tl / l;
//indicizzazione 0,1,2,3 a seconda di spin/isospin di particella
  auto idxOf = [&](int xp, int sp, int sn) { return (xp * 2 + sp) * 2 + sn; };
//matrice totale
  MatrixXcd M = MatrixXcd::Zero(nst, nst);
  //flip proposti dalle mosse non locali av6
  std::vector<std::vector<int>> flipsh;
  //elementi di matrice del potenziale
  std::vector<std::complex<double>> mel;

  //loop sulle posizioni delle particelle nel reticolo (protone)
  for (int xp = 0; xp < nv; ++xp) {
    const int xn = nv - 1 - xp; // sito speculare rispetto al centro (neutrone)
    //loop sugli spin delle particelle
    for (int sp = 0; sp < 2; ++sp) {
      for (int sn = 0; sn < 2; ++sn) {
        //definisco le posizioni delle particelle nei rispettivi canali
        const int p = xp + sp * nv;
        const int n = xn + (2 + sn) * nv;
        //stato del reticolo
        std::vector<int> state(4 * nv, -1);
        //creo particelle nei siti di prima
        state[p] = 1;
        state[n] = 1;
        //salvo i siti occupati
        std::vector<int> occ = {p, n}; // gia' ordinato: p < 2nv <= n
        //pulisco e calcolo potenziale
        flipsh.clear();
        mel.clear();
        H.FindConn(state, occ, flipsh, mel, 1, nv);
//indice riga
        const int row = idxOf(xp, sp, sn);
        //loop sugli stati flippati dall av6
        for (size_t k = 0; k < flipsh.size(); ++k) {
          if (k > 0 && flipsh[k].empty())
            continue; // hop rigettato alla parete
          // applica i flip per trovare lo stato d'arrivo
          std::vector<int> occ2 = {p, n};
          for (int s : flipsh[k]) {
            auto it = std::find(occ2.begin(), occ2.end(), s);
            if (it != occ2.end())
              occ2.erase(it);
            else
              occ2.push_back(s);
          }
          if (occ2.size() != 2) {
            std::cerr << "termine con numero di particelle sbagliato\n";
            std::exit(1);
          }
          int ps = -1, ns = -1;
          for (int s : occ2) {
            if (s < 2 * nv)
              ps = s;
            else
              ns = s;
          }
          assert(ps >= 0 && ns >= 0);
          const int xp2 = ps % nv, sp2 = ps / nv;
          const int nloc = ns - 2 * nv;
          const int xn2 = nloc % nv, sn2 = nloc / nv;
          if (xn2 != nv - 1 - xp2) {
            std::cerr << "stato d'arrivo fuori dalla varieta' COM!\n";
            std::exit(1);
          }
          M(row, idxOf(xp2, sp2, sn2)) += mel[k];
        }
      }
    }
  }

  const double herm = (M - M.adjoint()).cwiseAbs().maxCoeff();
  std::cout << "== l=" << l << " tl=" << tl << " fm  (dx=" << dx
            << ", passo relativo 2dx=" << 2 * dx << " fm) ==\n";
  std::cout << "   errore hermiticita' max = " << herm << std::endl;

  const bool wantVectors = (nst <= 6000);
  Eigen::SelfAdjointEigenSolver<MatrixXcd> es(
      M, wantVectors ? Eigen::ComputeEigenvectors : Eigen::EigenvaluesOnly);
  const auto &ev = es.eigenvalues();
  std::cout << "   E0..E4 [MeV]: ";
  for (int i = 0; i < 5 && i < nst; ++i)
    std::cout << ev(i) << "  ";
  std::cout << std::endl;

  if (wantVectors) {
    // proprieta' del ground state: <rab> e peso alle separazioni piccole
    const auto v0 = es.eigenvectors().col(0);
    double rmean = 0, w_small = 0;
    for (int xp = 0; xp < nv; ++xp) {
      const int xn = nv - 1 - xp;
      const double r = H.Distance(xp, xn) * dx;
      double w = 0;
      for (int sp = 0; sp < 2; ++sp)
        for (int sn = 0; sn < 2; ++sn)
          w += std::norm(v0(idxOf(xp, sp, sn)));
      rmean += r * w;
      if (r < 1.9 * dx + 1e-9) // r=0 o prima shell (2dx)
        w_small += w;
    }
    std::cout << "   GS: <r_pn> = " << rmean
              << " fm,  peso su r<=2dx: " << w_small;
  }
  std::cout << "\n" << std::endl;
}

int main(int argc, char **argv) {
  // scan di convergenza: E0 in funzione del passo relativo 2dx
  //runCase(7, 3.6);
  //runCase(9, 3.2);
  //runCase(9, 4.0);
  //runCase(11, 3.2);
  //runCase(11, 4.0);
  //runCase(13, 4.0);
  runCase(9,4.0);
  return 0;
}
