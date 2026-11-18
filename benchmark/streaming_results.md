# Streaming Decision Tree Benchmark

The previous results were generated with a fixed 64-row incoming chunk. That
made the 16-, 32-, and 64-sample buffer comparisons difficult to interpret,
because an incoming chunk could equal or exceed the buffer threshold.

The benchmark methodology has been corrected to use the same 16-row incoming
chunks for every buffer size and the median training time across three runs.
All tested buffer sizes are multiples of 16.

Regenerate the results from the repository root:

```powershell
python benchmark/streaming_benchmark.py
```

The script writes the new measured results to this file. Do not quote the
previous accuracy or timing figures; they belong to the superseded experiment.
