#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <string>
#include <limits>


// SAD in una posizione start: confronto Q con la finestra di S
long long sad_at(const std::vector<int>& S, const std::vector<int>& Q, int start) {
    long long sum = 0;
    int M = (int)Q.size();

    for (int j = 0; j < M; j++) {
        sum += std::abs(S[start + j] - Q[j]);
    }

    return sum;
}

std::string results_path(const std::string& filename) {
    return "../results/" + filename;
}

std::vector<int> read_vector_from_file(const std::string& filename) {
    std::ifstream in(filename);
    std::vector<int> v;

    if (!in) {
        std::cout << "Errore apertura file: " << filename << "\n";
        return v;
    }

    int x;
    while (in >> x) {
        v.push_back(x);
    }

    return v;
}


int main() {


    std::vector<int> S = read_vector_from_file("../dataset/S.txt");
    std::vector<int> Q = read_vector_from_file("../dataset/Q.txt");
    int N = (int)S.size();
    int M = (int)Q.size();
    std::cout << "Letti N=" << N << " valori per S, M=" << M << " valori per Q\n";

    /*
    // genero dati ripetibili
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 9);

    for (int i = 0; i < N; i++)
        S[i] = dist(rng);

    for (int j = 0; j < M; j++)
        Q[j] = dist(rng);
    */

    auto t_start = std::chrono::high_resolution_clock::now();

    long long bestVal = std::numeric_limits<long long>::max();
    int bestIdx = 0;

    // scansione sequenziale di tutte le posizioni
    for (int start = 0; start <= N - M; start++) {
        long long v = sad_at(S, Q, start);
        if (v < bestVal) {
            bestVal = v;
            bestIdx = start;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double time_seq = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\nRISULTATO:\n";
    std::cout << "bestIdx = " << bestIdx << "\n";
    std::cout << "bestSAD = " << bestVal << "\n";
    std::cout << "tempo (seq) = " << time_seq << " s\n";

    std::string out_file = results_path("seq_times.txt");
    std::ofstream out(out_file);

    if (!out) {
        std::cout << "Errore apertura file " << out_file << "\n";
        return 1;
    }

    // salvo tempo e parametri usati
    out << "pattern_sad " << time_seq << "\n";
    out << "N " << N << "\n";
    out << "M " << M << "\n";
    out << "bestIdx " << bestIdx << "\n";
    out << "bestSAD " << bestVal << "\n";

    std::cout << "Salvato su " << out_file << "\n";
    return 0;
}
