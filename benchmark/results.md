# NumCompute benchmark results

## Environment

- python: 3.15.0
- numpy: 2.5.4
- platform: Windows-11-10.0.26200-SP0
- machine: AMD64
- processor: AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD
- cpu_count: 16
- date: 2026-11-18 01:58 UTC
- seed: 0
- repeats: 3 (best time reported)
- mode: quick

## Standardise columns (10 features)

| Size | Loop | Vectorised | Speed-up | Match |
|---:|---:|---:|---:|:---:|
| 200 | 2.21 ms | 127.1 us | 17.4x | yes |
| 1,000 | 7.56 ms | 154.0 us | 49.1x | yes |
| 5,000 | 39.38 ms | 1.22 ms | 32.2x | yes |

## Row softmax (20 classes)

| Size | Loop | Vectorised | Speed-up | Match |
|---:|---:|---:|---:|:---:|
| 100 | 377.7 us | 46.1 us | 8.2x | yes |
| 500 | 1.91 ms | 148.7 us | 12.8x | yes |
| 2,000 | 8.80 ms | 513.2 us | 17.2x | yes |

## Pairwise Euclidean distances (8 features)

| Size | Loop | Vectorised | Speed-up | Match |
|---:|---:|---:|---:|:---:|
| 20 | 646.4 us | 16.9 us | 38.2x | yes |
| 50 | 4.10 ms | 26.4 us | 155.2x | yes |
| 100 | 16.20 ms | 48.5 us | 334.0x | yes |

## Average ranks with ties

| Size | Loop | Vectorised | Speed-up | Match |
|---:|---:|---:|---:|:---:|
| 100 | 445.1 us | 60.4 us | 7.4x | yes |
| 300 | 3.65 ms | 78.9 us | 46.2x | yes |
| 600 | 14.18 ms | 83.2 us | 170.4x | yes |

## Top-10 of n values

| Size | Loop | Vectorised | Speed-up | Match |
|---:|---:|---:|---:|:---:|
| 1,000 | 233.2 us | 80.8 us | 2.9x | yes |
| 10,000 | 3.58 ms | 176.8 us | 20.3x | yes |
| 50,000 | 72.21 ms | 1.24 ms | 58.3x | yes |

## Histogram (20 bins)

| Size | Loop | Vectorised | Speed-up | Match |
|---:|---:|---:|---:|:---:|
| 1,000 | 175.0 us | 43.7 us | 4.0x | yes |
| 10,000 | 1.77 ms | 301.5 us | 5.9x | yes |
| 50,000 | 13.73 ms | 2.18 ms | 6.3x | yes |

## Streaming mean/variance (Welford loop vs batch)

| Size | Loop | Vectorised | Speed-up | Match |
|---:|---:|---:|---:|:---:|
| 1,000 | 178.9 us | 43.8 us | 4.1x | yes |
| 10,000 | 882.1 us | 71.6 us | 12.3x | yes |
| 50,000 | 5.04 ms | 885.4 us | 5.7x | yes |

## Finite-difference gradient (one call per point vs one batched call)

| Size | Loop | Vectorised | Speed-up | Match |
|---:|---:|---:|---:|:---:|
| 20 | 180.1 us | 38.1 us | 4.7x | yes |
| 100 | 793.6 us | 58.0 us | 13.7x | yes |
| 300 | 3.19 ms | 1.62 ms | 2.0x | yes |
