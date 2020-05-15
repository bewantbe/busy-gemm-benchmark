#!/usr/bin/env python3
# Run
# ./busy_pytorch_blas.py 8000 3
# srun -J test_blas -p szsc --nodes=1 --time=2:00 bash -c "module load apps/PyTorch;  python3 $HOME/test_usage/busy_pytorch_blas.py 8000 3"
# Ex2:
# ./busy_pytorch_blas.py 8000 3 -dcuda:0 -pFP32

# Reference speed
# GFLOPS     FP32    FP64
# 2080Ti    12372     503
# V100      13601    6860
# W-2145     1465     723

import sys
from time import time, localtime, strftime

import torch
from torch import randn, norm, eye

print("cuda state: ", torch.cuda.is_available());

device_name = 'cuda'    # can be cpu, cuda, cuda:0
str_precision = 'fp32'  # default precision

#simple parse
argv_nonoption = []
for j in range(len(sys.argv)):
    if j==0: continue
    if sys.argv[j].startswith('-d'):
        device_name = sys.argv[j][2:]
    elif sys.argv[j].startswith('-p'):
        str_precision = sys.argv[j][2:].lower()
    else:
        argv_nonoption.append(sys.argv[j])

if str_precision in ['fp32', 'float32', 'float', 'binary32']:
    dtype = torch.float
elif str_precision in ['fp64', 'float64', 'double', 'binary64']:
    dtype = torch.double
elif str_precision in ['fp16', 'float16', 'half', 'binary16']:
    dtype = torch.float16
else:
    dtype = str_precision

#device = torch.device("cpu")
device = torch.device(device_name)

print('Using device:', device_name)
print('Precision:', str_precision)

# Keep CPU or GPU busy, that's it.
# n     : size of matrix
# k_max : number of loops
def busy_gemm(n = 4000, k_max = 2**31-2):
    is_gpu = device.type != 'cpu'
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    gflo = n ** 3 * 2 / 1e9
    v = randn(n,1, device=device, dtype=dtype);
    v = v / norm(v)
    u = randn(n,1, device=device, dtype=dtype);
    u = u / norm(u)
    # a (not very) random (but fast generated) orthogonal matrix
    a = eye(n, device=device, dtype=dtype) \
         - 2 * u.mm(u.t()) - 2 * v.mm(v.t()) \
         + (4 * u * (u.t().mm(v))).mm(v.t())
    c = a;
    for k in range(1, k_max+1):
        if is_gpu:
            start_event.record()
        else:
            t0 = time()
        c = c.mm(a)      # the payload
        if is_gpu:
            end_event.record()
            torch.cuda.synchronize(device)  # Wait for the events to be recorded!
            t = start_event.elapsed_time(end_event) / 1000.0
        else:
            t = time() - t0
        s = strftime("%Y-%m-%d %H:%M:%S %Z", localtime())
        print('%s, t=%.3f, #%d, GFLOPS=%5.1f.' % (s, t, k, gflo/t))

if __name__ == '__main__':
    param = [int(i) for i in argv_nonoption]
    busy_gemm(*param)

