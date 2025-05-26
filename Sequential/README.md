# Gaussian Blur Image Processor

Implementarea unui filtru Gaussian pentru procesarea imaginilor, cu versiuni secvențiale și paralele.

## Ce este inclus

- Implementare secvențială în C# folosind OpenCvSharp
- Implementare paralelă în Python folosind MPI4Py
- Suport pentru procesarea imaginilor de dimensiuni între 128×128 și 32768×32768 pixeli
- Aplicarea unui filtru Gaussian cu rază de 30 pixeli și σ=10
- Măsurarea și afișarea timpilor exacți de procesare
- Salvarea rezultatelor ca `filtered_[nume_original].jpg`

## Versiunea secvențială (C#)

### Cerințe
- .NET 9.0
- Visual Studio 2022
- OpenCvSharp

### Cum să rulați
1. **Deschideți soluția**:
   - Lansați Visual Studio
   - Deschideți `FilterImages.sln` din folderul rădăcină
2. **Rulați proiectul**:
   - Apăsați F5 pentru a depana, sau
   - Folosiți meniul: `Debug > Start Without Debugging`

## Versiunea paralelă (Python + MPI) 

### -> gaussian_blur_mpi

### Cerințe
- Python 3.9 sau mai recent
- MPI4Py
- OpenCV pentru Python
- NumPy

### Cum să rulați
1. **Instalați dependențele**:
   ```bash
   pip install mpi4py opencv-python numpy
   ```

2. **Rulați cu MPI**:
   ```bash
   mpiexec -n <număr_procese> python gaussian_filter_mpi.py
   ```
   Unde `<număr_procese>` este numărul de procese dorit (de ex. 4 sau 8)

## Performanță comparativă

| Dimensiune imagine | Secvențial (ms) | Paralel 4 procese (ms) | Paralel 8 procese (ms) |
|-------------------|-----------------|------------------------|------------------------|
| 128x128 px        | 4,71            | 1,45                   | 1,47                   |
| 256x256 px        | 6,45            | 2,81                   | 2,32                   |
| 512x512 px        | 15,79           | 6,37                   | 5,65                   |
| 1024x1024 px      | 30,18           | 17,63                  | 11,95                  |
| 2048x2048 px      | 97,91           | 43,74                  | 50,61                  |
| 4096x4096 px      | 406,92          | 146,05                 | 111,65                 |
| 8192x8192 px      | 1285,24         | 515,54                 | 349,52                 |
| 16384x16384 px    | 4531,74         | 2013,94                | 1524,81                |
| 32768x32768 px    | 19887,24        | 10622,58               | 8176,18                |

## Documentație detaliată

Pentru o analiză completă a algoritmilor și rezultatelor experimentale, consultați [documentația](https://docs.google.com/document/d/1Lvr94UeL-PmCsR4VaBGzNiWaocePwUWuNJOjV7AGTTY/edit?usp=sharing).
