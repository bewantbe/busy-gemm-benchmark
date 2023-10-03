#!/usr/bin/env python3
# Keep CPU or GPU busy by loops of matrix multiply, that's it.
# Specify the matrix dimension and number of loops in command line arguments.
# Example 1
# ./busy_pytorch_blas.py 8000 3
# Example 2:
# ./busy_pytorch_blas.py 8000 3 -dcuda:0 -pFP32
# Example 3 (for SLURM):
# srun -J test_blas -p szsc --nodes=1 --time=2:00 bash -c "module load apps/PyTorch;  python3 $HOME/test_usage/busy_pytorch_blas.py 8000 3"

# change of pytorch v1.12
# Disable TF32 for matmul by default and add high-level control of fp32 matmul precision
# https://dev-discuss.pytorch.org/t/pytorch-and-tensorfloat32/504

import sys
from time import time, localtime, strftime

import torch
from torch import randn, norm, eye

print('Torch version : ', torch.__version__)
print("CUDA available: ", torch.cuda.is_available());

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

# https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype
# https://en.wikipedia.org/wiki/Bfloat16_floating-point_format

# TF32 was default for pytorch v1.7~v1.11 we don't like that
# https://dev-discuss.pytorch.org/t/pytorch-and-tensorfloat32/504
torch.backends.cuda.matmul.allow_tf32 = False

if str_precision in ['fp32', 'float32', 'float', 'binary32']:
    # FP32: 1 + 8 + 23 (32bits)
    dtype = torch.float
elif str_precision in ['fp64', 'float64', 'double', 'binary64']:
    # FP64: 1 + 11 + 52 (64bits)
    dtype = torch.double
elif str_precision in ['fp16', 'float16', 'half', 'binary16']:
    # FP16: 1 + 5 + 10 (16bits)
    dtype = torch.float16
elif str_precision in ['tf32', 'tensorfloat32']:
    # TensorFloat32 (TF32)  operation
    # TF32: 1 + 8 + 10 (19bits)
    dtype = torch.float
    torch.backends.cuda.matmul.allow_tf32 = True
    #torch.backends.cudnn.allow_tf32 = True
elif str_precision in ['bf16', 'bfloat16']:
    # Bfloat16: 1 + 8 + 7 (16bits)
    dtype = torch.bfloat16
else:
    dtype = str_precision

#device = torch.device("cpu")
device = torch.device(device_name)

info_cuda = '  ver=' + torch.version.cuda if device_name=='cuda' else ''
print('Using device  :', device_name, info_cuda)
print('Data type     :', str_precision)

if device.type != 'cpu' and not torch.cuda.is_available():
    print('Selected device not available. (Check torch GPU support or CUDA hardware availability)')
    exit()

# Keep CPU or GPU busy, that's it.
# n     : size of matrix
# k_max : number of loops
def busy_gemm(n = 8000, k_max = 2**31-2):
    is_gpu = device.type != 'cpu'

    # contruct the timer
    if is_gpu:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event   = torch.cuda.Event(enable_timing=True)
        cuda_stream = torch.cuda.Stream(device)
        torch.cuda.set_stream(cuda_stream)  # although discouraged
        #print('Current cuda_stream: ', torch.cuda.current_stream())
        def start_timer():
            start_event.record()
            return None

        def stop_timer(st):
            end_event.record()
            # Wait for the events to be recorded!
            torch.cuda.synchronize(device)
            t = start_event.elapsed_time(end_event) / 1000.0
            return t
    else:
        def start_timer():
            t0 = time()
            return t0

        def stop_timer(t0):
            t = time() - t0
            return t

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
        timer = start_timer()
        c = c.mm(a)                 # the core payload
        t = stop_timer(timer)
        s = strftime("%Y-%m-%d %H:%M:%S %Z", localtime())
        print('%s, t=%.3f, #%d, GFLOPS=%5.1f.' % (s, t, k, gflo/t))

if __name__ == '__main__':
    param = [int(i) for i in argv_nonoption]
    busy_gemm(*param)

#              fp32      fp64
#cpu W-2145:  1554.5     750.5
# 2080Ti   : 13332.0     516.1
#  AMD ?   : 10499.4    5414.9
# V100     : 13601.2    6860.6
