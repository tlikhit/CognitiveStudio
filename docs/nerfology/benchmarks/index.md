# Benchmarks

Here, we compare the performance of official results (from the papers) and our
implementations.

Reported values are PSNR.

The suffix ``(P)`` denotes official results from the paper, while ``(C)`` means
CognitiveStudio (this repository) implementation.

## Sparse

Sparse training sets (few GT training images) are difficult to fit well.

The DTU dataset, consisting of objects on a white table, is a common benchmark.
3, 6, or 9 images are selected from the full set as the training views.

| Implementation  | 3 view | 6 view | 9 view |
| --------------- | ------ | ------ | ------ |
| RegNerf (P)     | 18.89  | 22.20  | 24.93  |
| RegNerf (C)     | 11.27  | 17.93  | 24.19  |
| FreeNerf (P)    | 19.63  | 23.73  | 25.13  |
| FreeNerf (C)    | 17.50  | 20.74  | 24.37  |
