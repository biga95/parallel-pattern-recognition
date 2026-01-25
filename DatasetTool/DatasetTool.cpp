#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>    // round

int main() {

    std::string in_file = "../dataset/ECG5000_TEST.txt";
    std::string out_S = "../dataset/S.txt";
    std::string out_Q = "../dataset/Q.txt";

    int LEN = 140;          // ECG5000: 140 valori per riga
    int SCALE = 1000;       // float -> int (x * SCALE)
    int rows_for_S = 5000;  // quante righe concateno per fare S
    int q_row = 10;          // quale riga usare come pattern

    std::ifstream in(in_file);
    if (!in) {
        std::cout << "Errore apertura file: " << in_file << "\n";
        return 1;
    }

    std::vector<int> S;
    std::vector<int> Q;
    S.reserve(rows_for_S * LEN);

    for (int r = 0; r < rows_for_S; r++) {

        double label;
        if (!(in >> label)) {
            std::cout << "Fine file o errore lettura alla riga " << r << "\n";
            break;
        }

        std::vector<int> row(LEN);

        for (int i = 0; i < LEN; i++) {
            double x;
            in >> x;
            row[i] = (int)std::round(x * SCALE);
        }

        if (r == q_row) {
            Q = row;
            Q[0] += 1;   // piccolo disturbo, così non è match perfetto
        }


        S.insert(S.end(), row.begin(), row.end());
    }

    if (S.empty() || Q.empty()) {
        std::cout << "Errore: S o Q vuota\n";
        return 1;
    }

    std::ofstream out1(out_S);
    if (!out1) {
        std::cout << "Errore scrittura: " << out_S << "\n";
        return 1;
    }
    for (int v : S) out1 << v << "\n";

    std::ofstream out2(out_Q);
    if (!out2) {
        std::cout << "Errore scrittura: " << out_Q << "\n";
        return 1;
    }
    for (int v : Q) out2 << v << "\n";

    std::cout << "Creato " << out_S << " (len=" << (int)S.size() << ")\n";
    std::cout << "Creato " << out_Q << " (len=" << (int)Q.size() << ")\n";

    return 0;
}
