// Diagonalizzazione esatta del deuterio (AV6') su reticolo, stessa struttura
// della diagonalizzazione dell'idrogeno: cinetica con prodotti di Kronecker,
// potenziale in posizione, autovalori con Eigen.
//
// Base: |x> (x) |sp sn>, protone nel sito x, neutrone nel sito speculare
// xbar = Ns-1-x (COM fisso al centro). Dimensione D = 4*n^3.
//
//   H = T (x) I4 + A + B*(-1 - 2S)
//   A(x) = v0 + v2 s.s + v4 S12     (termini senza isospin)
//   B(x) = v1 + v3 s.s + v5 S12     (termini che moltiplicano tau.tau)
//   S    = scambio p<->n: x -> xbar e sp <-> sn (tau.tau = -1 - 2S, segno
//          di scambio fermionico incluso)
//
// Build: mpic++ -std=c++17 -O3 -I. diagH2kron.cc -o diagH2kron
// Run:   ./diagH2kron [n] [trapleght]          (default n=9, trapleght=4.0)
#include "src/ArgonneV6.cc"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <vector>

using namespace Eigen;
using namespace std;

// Funzione per costruire la matrice m di dimensione n (derivata seconda 1D)
MatrixXd buildMatrix(int n) {
    // Creiamo una matrice n x n di zeri
    MatrixXd m = MatrixXd::Zero(n, n);

    // Diagonale principale -2 ovunque, anche agli angoli: parete a un passo
    // intero fuori dal reticolo, come nel NQS (ArgonneV6) e in diagH2.cc
    for (int i = 0; i < n; ++i) {
        m(i, i) = -2;
    }

    // Riempiamo la diagonale sopra e sotto con 1
    for (int i = 0; i < n - 1; ++i) {
        m(i, i + 1) = 1;  // diagonale sopra
        m(i + 1, i) = 1;  // diagonale sotto
    }

    return m;
}

// Funzione per calcolare il prodotto di Kronecker di due matrici
MatrixXcd kroneckerProduct(const MatrixXcd &A, const MatrixXcd &B) {
    int rowsA = A.rows();
    int colsA = A.cols();
    int rowsB = B.rows();
    int colsB = B.cols();

    // La matrice risultante avrà dimensione (rowsA * rowsB) x (colsA * colsB)
    MatrixXcd result(rowsA * rowsB, colsA * colsB);

    // Calcoliamo il prodotto di Kronecker
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsA; ++j) {
            result.block(i * rowsB, j * colsB, rowsB, colsB) = A(i, j) * B;
        }
    }

    return result;
}

//ottiene le coordinate del punto sul reticolo dato
std::vector<int> CalculateCoo(int s, int n) {
    return std::vector<int>{int(int(s / n) / n), int(s / n) % n, s % n};
}

int main(int argc, char **argv) {

    int n = 9;                 // punti per lato
    double trapleght = 4.0;    // lato della scatola [fm]
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) trapleght = atof(argv[2]);

    const int Ns = n * n * n;  // siti del reticolo
    const int D = 4 * Ns;      // siti x (spin protone, spin neutrone)
    double spacing = trapleght / n;
    double center = (n - 1) / 2.;

    const double hc = 197.327053;                    // MeV fm (come ArgonneV6)
    const double mass = (938.27231 + 939.56563) / 2; // MeV    (mp+mn)/2
    // cinetica relativa: massa ridotta m/2 e passo relativo 2*spacing
    double factor = -hc * hc / (4 * mass * pow(spacing, 2));

    // ---------------- energia cinetica ----------------
    MatrixXcd m = buildMatrix(n).cast<complex<double>>();
    MatrixXcd Id = MatrixXcd::Identity(n, n);

    // Prodotto di Kronecker m ⊗ I ⊗ I
    MatrixXcd m_kron_Id_kron_Id = kroneckerProduct(m, kroneckerProduct(Id, Id));

    // Prodotto di Kronecker I ⊗ m ⊗ I
    MatrixXcd Id_kron_m_kron_Id = kroneckerProduct(kroneckerProduct(Id, m), Id);

    // Prodotto di Kronecker I ⊗ I ⊗ m
    MatrixXcd Id_kron_Id_kron_m = kroneckerProduct(kroneckerProduct(Id, Id), m);

    // Sommiamo i tre prodotti di Kronecker
    MatrixXcd M = m_kron_Id_kron_Id + Id_kron_m_kron_Id + Id_kron_Id_kron_m;
    M = M * factor;

    // ---------------- matrici di spin 4x4 ----------------
    // base |sp sn>, indice 2*sp+sn con 0 = up (convenzione di Pauli)
    MatrixXcd sx(2, 2), sy(2, 2), sz(2, 2);
    sx << 0, 1, 1, 0;
    sy << 0, complex<double>(0, -1), complex<double>(0, 1), 0;
    sz << 1, 0, 0, -1;
    MatrixXcd I4 = MatrixXcd::Identity(4, 4);

    // sigma1 . sigma2
    MatrixXcd SS = kroneckerProduct(sx, sx) + kroneckerProduct(sy, sy) +
                   kroneckerProduct(sz, sz);
    // scambio degli spin sp <-> sn
    MatrixXcd Pspin = 0.5 * (I4 + SS);

    // ---------------- hamiltoniana ----------------
    // parte cinetica: T ⊗ I4 (la cinetica non tocca gli spin)
    MatrixXcd H = kroneckerProduct(M, I4);

    // potenziale: blocchi 4x4 in x (diagonale) e tra x e xbar (scambio)
    ArgonneV6 av(Ns, 2, trapleght); // serve solo per v_i(r) e S12
    for (int x = 0; x < Ns; ++x) {
        std::vector<int> coo = CalculateCoo(x, n);
        double d[3] = {coo[0] - center, coo[1] - center, coo[2] - center};
        double dn = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);

        double r = 2 * spacing * dn;       // distanza protone-neutrone
        std::vector<double> rhat(3, 0.0);  // al centro r=0: S12 = 0
        if (dn > 0)
            for (int q = 0; q < 3; ++q) rhat[q] = d[q] / dn;

        // ordine Argonne: v[0]=1 v[1]=tt v[2]=ss v[3]=ss tt v[4]=S12 v[5]=S12 tt
        std::vector<double> v = av.AV6operators(r);
        MatrixXcd S12 = av.Compute_S12(rhat);

        MatrixXcd A = v[0] * I4 + v[2] * SS + v[4] * S12;
        MatrixXcd B = v[1] * I4 + v[3] * SS + v[5] * S12;

        int xbar = Ns - 1 - x;             // sito speculare
        H.block(4 * x, 4 * x, 4, 4) += A - B;
        H.block(4 * x, 4 * xbar, 4, 4) += -2.0 * B * Pspin;
    }

    // controllo di hermiticità
    cout << "# Deuterio AV6'  n=" << n << "  trapleght=" << trapleght
         << " fm  (spacing=" << spacing << " fm)  D=" << D << endl;
    cout << "errore hermiticita' max = " << (H - H.adjoint()).cwiseAbs().maxCoeff()
         << endl;

    // Calcolare gli autovalori di H (hermitiana: autovalori reali, ordinati)
    SelfAdjointEigenSolver<MatrixXcd> solver(H, EigenvaluesOnly);
    VectorXd eigenvalues = solver.eigenvalues();

    cout << "E0..E4 [MeV]: ";
    for (int i = 0; i < 5 && i < D; ++i) cout << eigenvalues(i) << "  ";
    cout << endl;

    return 0;
}
