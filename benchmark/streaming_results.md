# Streaming Decision Tree Benchmark

## Experimental setup

- Training samples: 2,400
- Test samples: 600
- Fixed incoming chunk size: 16 rows
- Buffer sizes: 16, 32, 64, 128, and 256
- Repeated runs per buffer size: 3
- Training time: median across the three runs
- Dataset: reproducibly generated with random seed 42

## Measured results

| Buffer Size | Chunk Size | Accuracy | Median Training Time (seconds) | Retrain Count | Runs |
|---:|---:|---:|---:|---:|---:|
| 16 | 16 | 0.7067 | 0.121408 | 150 | 3 |
| 32 | 16 | 0.7183 | 0.102590 | 75 | 3 |
| 64 | 16 | 0.7150 | 0.086192 | 37 | 3 |
| 128 | 16 | 0.7450 | 0.067467 | 18 | 3 |
| 256 | 16 | 0.7617 | 0.055604 | 9 | 3 |

## Interpretation

In this experiment, the 256-row buffer achieved the highest observed accuracy
(0.7617), the shortest median training time (0.055604 seconds), and the fewest
retraining events (9). Increasing the buffer size reduced the frequency of
retraining and the measured training time in this run. These results describe
this generated dataset and execution environment; they are not universal
performance guarantees.

Regenerate the experiment from the repository root with:

```powershell
python benchmark/streaming_benchmark.py
```
