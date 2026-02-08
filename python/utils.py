
def read_series_txt(path: str):
    """
    Legge numeri da un file di testo.
    - numeri separati da spazi OK
    - numeri uno per riga OK
    - se ci sono virgole le tratta come spazi
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            line = line.replace(",", " ")
            parts = line.split()

            for p in parts:
                data.append(float(p))

    return data


def sad_search(S, Q):
    """
    Cerca la posizione start che minimizza:

      SAD(start) = somma_j |S[start+j] - Q[j]|

    Ritorna:
      bestIdx, bestSAD
    """
    n = len(S)
    m = len(Q)

    if m == 0 or n < m:
        raise ValueError("Errore: serve len(S) >= len(Q) e len(Q) > 0")

    best_idx = 0
    best_sad = None

    # start va da 0 a (n - m) incluso
    for start in range(0, n - m + 1):
        acc = 0.0

        for j in range(m):
            acc += abs(S[start + j] - Q[j])

        if best_sad is None or acc < best_sad:
            best_sad = acc
            best_idx = start

    return best_idx, best_sad
