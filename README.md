# Pattern Recognition (SAD) – Programmazione Parallela

Progetto finale per il corso di Programmazione Parallela.

L’obiettivo del progetto è cercare un pattern (una sequenza di numeri) all’interno di una serie più lunga,
calcolando per ogni posizione la differenza totale (SAD – Sum of Absolute Differences) e individuando
la posizione in cui il pattern combacia meglio.

Il progetto è diviso in due parti:
- versione sequenziale
- versione parallela con OpenMP

Strumenti utilizzati:
- C++
- Visual Studio 2022
- OpenMP




Dato un segnale lungo `S` e un pattern più corto `Q`, l’algoritmo cerca la posizione in cui `Q` è più simile a `S` usando come metrica la SAD (Sum of Absolute Differences).

I dati provengono dal dataset ECG5000.  
Un tool (`DatasetTool`) viene usato per generare i file `S.txt` e `Q.txt`.

Sono presenti due versioni:
- una sequenziale
- una parallela usando OpenMP

I tempi di esecuzione e lo speedup sono salvati nella cartella `results/`.
