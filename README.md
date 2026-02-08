# Parallel Computing – FinalTerm Pattern Recognition

Progetto per l’esame di Parallel Computing (9 CFU).

Il progetto risolve un problema di pattern matching su serie temporali usando la metrica SAD (Sum of Absolute Differences).

Sono presenti due implementazioni parallele:
- CUDA (C++) su GPU  
- Python (multiprocessing) su CPU  

---
Struttura

- `CudaProject/` → implementazioni CUDA  
- `SequentialProject/`, `ParallelProject/` → versioni CPU in C++  
- `python/` → versione Python (sequenziale e multiprocessing)  
- `dataset/` → file di input  
- `results/` → tempi e risultati degli esperimenti  

---

 Esecuzione (Python)

```bash
cd python
python sad_seq.py
python sad_mp.py


I risultati vengono salvati in results/.