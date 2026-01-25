#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <omp.h>
#include <limits>

// SAD in una posizione start: confronto Q con la finestra di S che parte da start
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

// leggo il tempo sequenziale dal file (riga: pattern_sad <tempo>)
double read_seq_time(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) return -1.0;  // errore apertura file

    std::string key;
    double val;

    while (in >> key >> val) {
        if (key == "pattern_sad")
            return val;
    }

    return -1.0;  // valore non trovato
}

// leggo i tempi paralleli dal file (righe: T <threads> <time>)
std::vector<std::pair<int, double>> read_par_times(const std::string& filename) {
    std::vector<std::pair<int, double>> data;

    std::ifstream in(filename);
    if (!in) return data;

    std::string key;
    while (in >> key) {
        if (key == "T") {
            int t;
            double time;
            int idx;
            long long sad;
            in >> t >> time >> idx >> sad;
            data.push_back({ t, time });
        }
        else {
            std::string rest;
            std::getline(in, rest); // salto il resto della riga (tipo N..., M..., cols...)
        }

    }

    return data;
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

    if (S.empty() || Q.empty() || N < M) {
        std::cout << "Errore: file vuoti o dimensioni sbagliate\n";
        return 1;
    }


    // provo diversi numeri di thread e salvo i tempi
    int thread_list[] = { 1, 2, 4, 8, 16 };
    int nt = sizeof(thread_list) / sizeof(thread_list[0]);
    int num_tests = 5;

    std::string out_file = results_path("par_times.txt");
    std::ofstream out(out_file);

    if (!out) {
        std::cout << "Errore apertura file " << out_file << "\n";
        return 1;
    }

    out << "N " << N << "\n";
    out << "M " << M << "\n";
    out << "cols T time bestIdx bestSAD\n";

    for (int ti = 0; ti < nt; ti++) {
        int T = thread_list[ti];
        omp_set_num_threads(T);

        double best_time = 1e100;
        long long bestVal = 0;
        int bestIdx = 0;

        // ripeto più run e tengo il migliore (per stabilità)
        for (int r = 0; r < num_tests; r++) {

            auto t_start = std::chrono::high_resolution_clock::now();

            long long globalBestVal = std::numeric_limits<long long>::max();
            int globalBestIdx = 0;

#pragma omp parallel
            {
                long long localBestVal = std::numeric_limits<long long>::max();
                int localBestIdx = 0;

#pragma omp for
                for (int start = 0; start <= N - M; start++) {
                    long long v = sad_at(S, Q, start);
                    if (v < localBestVal) {
                        localBestVal = v;
                        localBestIdx = start;
                    }
                }

                // merge dei best locali
#pragma omp critical
                {
                    if (localBestVal < globalBestVal) {
                        globalBestVal = localBestVal;
                        globalBestIdx = localBestIdx;
                    }
                }
            }

            auto t_end = std::chrono::high_resolution_clock::now();
            double time_par = std::chrono::duration<double>(t_end - t_start).count();

            if (time_par < best_time) {
                best_time = time_par;
                bestVal = globalBestVal;
                bestIdx = globalBestIdx;
            }
        }

        std::cout << "T=" << T
            << " bestIdx=" << bestIdx
            << " bestSAD=" << bestVal
            << " time=" << best_time << " s\n";

        out << "T " << T << " " << best_time << " " << bestIdx << " " << bestVal << "\n";

    }

    std::cout << "Salvato su " << out_file << "\n";
    out.close();

    // speedup = Tseq / Tpar
    std::string seq_file = results_path("seq_times.txt");
    double t_seq = read_seq_time(seq_file);

    if (t_seq < 0) {
        std::cout << "Errore lettura " << seq_file << "\n";
        return 1;
    }

    auto par_data = read_par_times(out_file);
    if (par_data.empty()) {
        std::cout << "Errore lettura " << out_file << "\n";
        return 1;
    }

    std::string sp_file = results_path("speedup.txt");
    std::ofstream sp(sp_file);

    if (!sp) {
        std::cout << "Errore apertura file " << sp_file << "\n";
        return 1;
    }

    sp << "T time speedup\n";
    for (auto& p : par_data) {
        int T = p.first;
        double t = p.second;
        sp << T << " " << t << " " << (t_seq / t) << "\n";
    }

    std::cout << "Speedup salvato su " << sp_file << "\n";

    return 0;
}
