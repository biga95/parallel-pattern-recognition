import os
import time
from utils import read_series_txt, sad_search


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

    # leggo i dati
    S = read_series_txt(s_path)
    Q = read_series_txt(q_path)

    print("len(S) =", len(S))
    print("len(Q) =", len(Q))

    # misuro il tempo del calcolo
    t0 = time.perf_counter()
    best_idx, best_sad = sad_search(S, Q)
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000.0

    print("\nRISULTATO (Python sequenziale)")
    print("bestIdx =", best_idx)
    print("bestSAD =", best_sad)
    print("time_ms =", round(elapsed_ms, 3))

    # salvo tempo
    out_time = os.path.join(results_dir, "python_times.txt")
    with open(out_time, "w", encoding="utf-8") as f:
        f.write("python_seq_ms\n")
        f.write(f"{elapsed_ms:.3f}\n")

    # salvo risultato
    out_res = os.path.join(results_dir, "python_seq_result.txt")
    with open(out_res, "w", encoding="utf-8") as f:
        f.write(f"bestIdx {best_idx}\n")
        f.write(f"bestSAD {best_sad}\n")

    print("\nFile salvati in results/:")
    print("-", "python_times.txt")
    print("-", "python_seq_result.txt")


if __name__ == "__main__":
    main()
