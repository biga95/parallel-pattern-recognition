import os
import time
import multiprocessing as mp
from utils import read_series_txt


S_GLOBAL = None
Q_GLOBAL = None


def init_worker(S, Q): 
    global S_GLOBAL, Q_GLOBAL
    S_GLOBAL = S
    Q_GLOBAL = Q


def worker_range(args):
    """Ogni worker riceve un intervallo di start: [start_begin, start_end)
    e trova il minimo SAD dentro quell'intervallo.
    Ritorna (bestIdxLocale, bestSADLocale).    """
    start_begin, start_end = args
    S = S_GLOBAL
    Q = Q_GLOBAL

    m = len(Q)

    best_idx = start_begin
    best_sad = None

    for start in range(start_begin, start_end):
        acc = 0.0
        for j in range(m):
            acc += abs(S[start + j] - Q[j])

        if best_sad is None or acc < best_sad:
            best_sad = acc
            best_idx = start

    return best_idx, best_sad


def make_chunks(first_start, last_start_inclusive, chunk_size):
    chunks = []
    s = first_start
    last_exclusive = last_start_inclusive + 1

    while s < last_exclusive:
        e = min(s + chunk_size, last_exclusive)
        chunks.append((s, e))
        s = e

    return chunks


def main():
    base_dir = os.path.dirname(__file__)
    project_dir = os.path.abspath(os.path.join(base_dir, ".."))
    dataset_dir = os.path.join(project_dir, "dataset")
    results_dir = os.path.join(project_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    s_path = os.path.join(dataset_dir, "S.txt")
    q_path = os.path.join(dataset_dir, "Q.txt")

    if not os.path.exists(s_path):
        print("ERRORE: non trovo", s_path)
        return
    if not os.path.exists(q_path):
        print("ERRORE: non trovo", q_path)
        return

    S = read_series_txt(s_path)
    Q = read_series_txt(q_path)

    n = len(S)
    m = len(Q)
    if m == 0 or n < m:
        print("ERRORE: dimensioni non valide")
        return

    first_start = 0
    last_start = n - m  # incluso

    print("len(S) =", n)
    print("len(Q) =", m)

    "num_proc = mp.cpu_count()  # usa tutti i core logici"
    num_proc = 4             # 4 processi valori migliori
    chunk_size = 5000         

    chunks = make_chunks(first_start, last_start, chunk_size)

    print("\nMultiprocessing:")
    print("processi =", num_proc)
    print("chunk_size =", chunk_size)
    print("num_chunks =", len(chunks))

 
    t0 = time.perf_counter()


    with mp.Pool(processes=num_proc, initializer=init_worker, initargs=(S, Q)) as pool:
        local_results = pool.map(worker_range, chunks)


    best_idx = None
    best_sad = None
    for idx, sad in local_results:
        if best_sad is None or sad < best_sad:
            best_sad = sad
            best_idx = idx

    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0
    # Leggo il tempo sequenziale salvato prima (se esiste) per calcolare speedup
    seq_time_path = os.path.join(results_dir, "python_times.txt")
    seq_ms = None

    if os.path.exists(seq_time_path):
        with open(seq_time_path, "r", encoding="utf-8") as f:
            lines = [x.strip() for x in f.readlines() if x.strip()]
            
            if len(lines) >= 2:
                seq_ms = float(lines[1])

    if seq_ms is not None and elapsed_ms > 0:
        speedup = seq_ms / elapsed_ms
        print("speedup_vs_seq =", round(speedup, 3))

        out_speed = os.path.join(results_dir, "python_speedup.txt")
        with open(out_speed, "w", encoding="utf-8") as f:
            f.write("python_seq_ms python_mp_ms speedup\n")
            f.write(f"{seq_ms:.3f} {elapsed_ms:.3f} {speedup:.6f}\n")

        print("Salvato:", "python_speedup.txt")
    else:
        print("Nota: non trovo python_times.txt, quindi non calcolo speedup.")

    print("\nRISULTATO (Python multiprocessing)")
    print("bestIdx =", best_idx)
    print("bestSAD =", best_sad)
    print("time_ms =", round(elapsed_ms, 3))

    # Salvo tempo
    out_time = os.path.join(results_dir, "python_times_mp.txt")
    with open(out_time, "w", encoding="utf-8") as f:
        f.write("python_mp_ms\n")
        f.write(f"{elapsed_ms:.3f}\n")

    # Salvo risultato
    out_res = os.path.join(results_dir, "python_mp_result.txt")
    with open(out_res, "w", encoding="utf-8") as f:
        f.write(f"bestIdx {best_idx}\n")
        f.write(f"bestSAD {best_sad}\n")

    print("\nFile salvati in results/:")
    print("-", "python_times_mp.txt")
    print("-", "python_mp_result.txt")


if __name__ == "__main__":
    main()
