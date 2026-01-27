
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>


std::vector<int> read_vector_from_file(const std::string& filename) {
    std::ifstream in(filename);
    std::vector<int> v;

    if (!in) {
        std::cout << "Errore apertura file: " << filename << "\n";
        return v;
    }

    int x;
    while (in >> x) v.push_back(x);
    return v;
}

__constant__ int cQ[140];

__global__ void sad_kernel_global(const int* S, const int* Q, long long* out, int N, int M)
{
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int maxStart = N - M;
    if (start > maxStart) return;

    long long sad = 0;
    for (int j = 0; j < M; j++) {
        sad += llabs((long long)S[start + j] - (long long)Q[j]);
    }
    out[start] = sad;
}


__global__ void sad_kernel_const(const int* S, long long* out, int N, int M) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int maxStart = N - M;

    if (start > maxStart) return;

    long long sum = 0;
    for (int j = 0; j < M; j++) {
        int diff = S[start + j] - cQ[j];
        if (diff < 0) diff = -diff;
        sum += diff;
    }

    out[start] = sum;
}

__global__ void sad_kernel_shared(const int* S, const int* Q, long long* out, int N, int M)
{
    extern __shared__ int sQ[];   // dimensione = M * sizeof(int)

    // carico Q in shared (tutti i thread del blocco collaborano)
    for (int j = threadIdx.x; j < M; j += blockDim.x) {
        sQ[j] = Q[j];
    }
    __syncthreads();

    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int maxStart = N - M;
    if (start > maxStart) return;

    long long sad = 0;
    for (int j = 0; j < M; j++) {
        sad += llabs((long long)S[start + j] - (long long)sQ[j]);
    }

    out[start] = sad;
}

double read_seq_time(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) return -1.0;

    std::string key;
    double val;

    while (in >> key >> val) {
        if (key == "pattern_sad")
            return val;   // in secondi
    }
    return -1.0;
}


int main() {
    auto S = read_vector_from_file("../dataset/S.txt");
    auto Q = read_vector_from_file("../dataset/Q.txt");

    std::cout << "Letti N=" << (int)S.size() << " valori per S, M=" << (int)Q.size() << " valori per Q\n";

    if (S.empty() || Q.empty()) {
        std::cout << "File vuoti o path sbagliato\n";
        return 1;
    }

    int N = (int)S.size();
    int M = (int)Q.size();
    int maxStart = N - M;

    int* dS = nullptr;
    int* dQ = nullptr;
    long long* dOut = nullptr;

    cudaMalloc(&dS, N * sizeof(int));
    cudaMalloc(&dQ, M * sizeof(int));
    cudaMalloc(&dOut, (maxStart + 1) * sizeof(long long));

    auto t_all_start = std::chrono::high_resolution_clock::now();

    cudaMemcpy(dS, S.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dQ, Q.data(), M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(cQ, Q.data(), M * sizeof(int));

    int threads = 256;
    int blocks = (maxStart + 1 + threads - 1) / threads;

    // ---------------- CUDA GLOBAL ----------------
    auto t_global_start = std::chrono::high_resolution_clock::now();

    cudaEvent_t gStart, gStop;
    cudaEventCreate(&gStart);
    cudaEventCreate(&gStop);

    cudaEventRecord(gStart);
    sad_kernel_global << <blocks, threads >> > (dS, dQ, dOut, N, M);
    cudaEventRecord(gStop);
    cudaEventSynchronize(gStop);

    float global_ms = 0.0f;
    cudaEventElapsedTime(&global_ms, gStart, gStop);

    cudaEventDestroy(gStart);
    cudaEventDestroy(gStop);

    std::vector<long long> out_global(maxStart + 1);
    cudaMemcpy(out_global.data(), dOut, (maxStart + 1) * sizeof(long long), cudaMemcpyDeviceToHost);

    // cerco il minimo su CPU
    long long global_bestVal = out_global[0];
    int global_bestIdx = 0;
    for (int i = 1; i <= maxStart; i++) {
        if (out_global[i] < global_bestVal) {
            global_bestVal = out_global[i];
            global_bestIdx = i;
        }
    }

    auto t_global_end = std::chrono::high_resolution_clock::now();
    double global_total_ms = std::chrono::duration<double, std::milli>(t_global_end - t_global_start).count();

    std::cout << "CUDA(global) kernel_ms=" << global_ms << "\n";
    std::cout << "CUDA(global) total_ms=" << global_total_ms << "\n";
    std::cout << "CUDA(global) bestIdx=" << global_bestIdx << " bestSAD=" << global_bestVal << "\n";

    // ---------------- CUDA CONSTANT ----------------
    auto t_const_start = std::chrono::high_resolution_clock::now();

    cudaEvent_t cStart, cStop;
    cudaEventCreate(&cStart);
    cudaEventCreate(&cStop);

    cudaEventRecord(cStart);
    sad_kernel_const << <blocks, threads >> > (dS, dOut, N, M);
    cudaEventRecord(cStop);
    cudaEventSynchronize(cStop);

    float const_ms = 0.0f;
    cudaEventElapsedTime(&const_ms, cStart, cStop);

    cudaEventDestroy(cStart);
    cudaEventDestroy(cStop);

    std::vector<long long> out_const(maxStart + 1);
    cudaMemcpy(out_const.data(), dOut, (maxStart + 1) * sizeof(long long), cudaMemcpyDeviceToHost);

    // cerco il minimo su CPU
    long long const_bestVal = out_const[0];
    int const_bestIdx = 0;
    for (int i = 1; i <= maxStart; i++) {
        if (out_const[i] < const_bestVal) {
            const_bestVal = out_const[i];
            const_bestIdx = i;
        }
    }

    auto t_const_end = std::chrono::high_resolution_clock::now();
    double const_total_ms = std::chrono::duration<double, std::milli>(t_const_end - t_const_start).count();

    std::cout << "CUDA(const) kernel_ms=" << const_ms << "\n";
    std::cout << "CUDA(const) total_ms=" << const_total_ms << "\n";
    std::cout << "CUDA(const) bestIdx=" << const_bestIdx << " bestSAD=" << const_bestVal << "\n";


    std::ofstream outFile("../results/cuda_times.txt");
    if (outFile) {
        outFile << "N " << N << "\n";
        outFile << "M " << M << "\n";

        outFile << "cuda_global_kernel_ms " << global_ms << "\n";
        outFile << "cuda_global_total_ms " << global_total_ms << "\n";
        outFile << "cuda_global_bestIdx " << global_bestIdx << "\n";
        outFile << "cuda_global_bestSAD " << global_bestVal << "\n";

        outFile << "cuda_const_kernel_ms " << const_ms << "\n";
        outFile << "cuda_const_total_ms " << const_total_ms << "\n";
        outFile << "cuda_const_bestIdx " << const_bestIdx << "\n";
        outFile << "cuda_const_bestSAD " << const_bestVal << "\n";
    }


    // ---------------- CUDA SHARED ----------------
    auto t_shared_start = std::chrono::high_resolution_clock::now();

    cudaEvent_t sStart, sStop;
    cudaEventCreate(&sStart);
    cudaEventCreate(&sStop);

    size_t shmemBytes = M * sizeof(int);

    cudaEventRecord(sStart);
    sad_kernel_shared << <blocks, threads, shmemBytes >> > (dS, dQ, dOut, N, M);
    cudaEventRecord(sStop);
    cudaEventSynchronize(sStop);

    float shared_ms = 0.0f;
    cudaEventElapsedTime(&shared_ms, sStart, sStop);

    cudaEventDestroy(sStart);
    cudaEventDestroy(sStop);

    std::vector<long long> out_shared(maxStart + 1);
    cudaMemcpy(out_shared.data(), dOut, (maxStart + 1) * sizeof(long long), cudaMemcpyDeviceToHost);

    // cerco il minimo su CPU
    long long shared_bestVal = out_shared[0];
    int shared_bestIdx = 0;
    for (int i = 1; i <= maxStart; i++) {
        if (out_shared[i] < shared_bestVal) {
            shared_bestVal = out_shared[i];
            shared_bestIdx = i;
        }
    }

    auto t_shared_end = std::chrono::high_resolution_clock::now();
    double shared_total_ms = std::chrono::duration<double, std::milli>(t_shared_end - t_shared_start).count();

    std::cout << "CUDA(shared) kernel_ms=" << shared_ms << "\n";
    std::cout << "CUDA(shared) total_ms=" << shared_total_ms << "\n";
    std::cout << "CUDA(shared) bestIdx=" << shared_bestIdx << " bestSAD=" << shared_bestVal << "\n";

    outFile << "cuda_shared_kernel_ms " << shared_ms << "\n";
    outFile << "cuda_shared_total_ms " << shared_total_ms << "\n";
    outFile << "cuda_shared_bestIdx " << shared_bestIdx << "\n";
    outFile << "cuda_shared_bestSAD " << shared_bestVal << "\n";

    double t_seq = read_seq_time("../results/seq_times.txt");
if (t_seq < 0) {
    std::cout << "Errore lettura ../results/seq_times.txt\n";
}
else {
    std::ofstream sp("../results/cuda_speedup.txt");
    if (!sp) {
        std::cout << "Errore apertura ../results/cuda_speedup.txt\n";
    }
    else {
        sp << "version time_s speedup\n";

        // GLOBAL
        sp << "cuda_global_kernel " << (global_ms / 1000.0) << " " << (t_seq / (global_ms / 1000.0)) << "\n";
        sp << "cuda_global_total "  << (global_total_ms / 1000.0) << " " << (t_seq / (global_total_ms / 1000.0)) << "\n";

        // CONST
        sp << "cuda_const_kernel " << (const_ms / 1000.0) << " " << (t_seq / (const_ms / 1000.0)) << "\n";
        sp << "cuda_const_total "  << (const_total_ms / 1000.0) << " " << (t_seq / (const_total_ms / 1000.0)) << "\n";

        // SHARED
        sp << "cuda_shared_kernel " << (shared_ms / 1000.0) << " " << (t_seq / (shared_ms / 1000.0)) << "\n";
        sp << "cuda_shared_total "  << (shared_total_ms / 1000.0) << " " << (t_seq / (shared_total_ms / 1000.0)) << "\n";
    }

    std::cout << "Speedup CUDA salvato su ../results/cuda_speedup.txt\n";
}

    std::cout << "\n=== CUDA SPEEDUP (vs CPU sequenziale) ===\n";

    std::cout << "CUDA global  (kernel)  speedup = "
    << (t_seq / (global_ms / 1000.0)) << "\n";

    std::cout << "CUDA global  (total)   speedup = "
    << (t_seq / (global_total_ms / 1000.0)) << "\n";

    std::cout << "CUDA constant(kernel)  speedup = "
    << (t_seq / (const_ms / 1000.0)) << "\n";

    std::cout << "CUDA constant(total)   speedup = "
    << (t_seq / (const_total_ms / 1000.0)) << "\n";

    std::cout << "CUDA shared  (kernel)  speedup = "
    << (t_seq / (shared_ms / 1000.0)) << "\n";

    std::cout << "CUDA shared  (total)   speedup = "
    << (t_seq / (shared_total_ms / 1000.0)) << "\n";


    return 0;
}


