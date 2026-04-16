# Project Instructions

1. For every function, add a comment immediately after its definition describing its inputs and outputs.
2. For every implementation, create a testing script, run it, and iterate until the implementation works correctly.
3. When constructing pipeline configs, use a root config with `data.csv_path` and contiguous numbered stages `stage1`, `stage2`, ... only. Each stage config must declare an instantiable `_target_`, and its declared input and output artifacts must be compatible with the previous and next pipeline stages.
