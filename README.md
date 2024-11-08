# Table for benchmark result

By torch

TFLOPS

| machine | fp32 | fp64 |
|---------|------|------|
| surface pro 6 (cpu) | 0.28 | 0.12 |
| R7 5800H            | 0.66 | 0.28 |
| Gold 6252 (4cores)  | 0.68 | 0.34 |
| W-2145              | 1.53 | 0.75 |

| machine   | fp32 | fp64 | fp16 | tf32 |
|-----------|------|------|------|------|
| 2080Ti    | 11.4 | 0.52 | 44.9 | 11.2 |
| 3070Laptop|  9.1 | 0.27 | 32.0 | 15.6 |
| A40       | 18.9 | 0.43 | 40   | 40   |
| 3090Ti    | 21   | 0.50 | 66   | 33   |
| A100(40G) | 14.1 | 13.7 | 153  | 108  |

```bash
# test command
python busy_pytorch_blas.py 8000 10 -dcuda -pfp32
```
