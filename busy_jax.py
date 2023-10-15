#!/usr/bin/env python3
# Run
# source ~/pyenv/jax/bin/activate
# ./busy_jax.py 8000 3

import sys
from time import time, localtime, strftime
from jax import config
import jax
from jax import random
from jax import device_put
import jax.numpy as jnp
from jax.numpy.linalg import norm

# Keep machine busy by using gemm in JAX
def busy_jax_gemm(n = 8000, k_max = 2**31-2,
                  dtype = jnp.float64,
                  device = 'cpu',
                  rnd_key = jnp.array([0, 0], dtype=jnp.uint32)):
    if dtype == 'tensorfloat32':
        dtype_mul = dtype
        dtype = jnp.float32
    else:
        dtype_mul = dtype.dtype.name
    gflo = n ** 3 * 2 / 1e9
    # Random
    # Ref. https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#%F0%9F%94%AA-Random-Numbers
    rnd_key, sub_key = random.split(rnd_key)
    v = random.normal(sub_key, (n,1), dtype=dtype);  v = v/norm(v)
    rnd_key, sub_key = random.split(rnd_key)
    u = random.normal(sub_key, (n,1), dtype=dtype);  u = u/norm(u)
    # a (not very) random (but fast generated) orthogonal matrix
    a = jnp.eye(n, dtype=dtype) - 2 * u @ u.T - 2 * v @ v.T \
                                + 4 * u * (u.T @ v) @ v.T
    c = a;
    print('Data on device:', c.device_buffer.device())
    print('Data type:', dtype.dtype)
    if dtype.dtype.itemsize == 4:
        print('  - Matmul precision:', dtype_mul)
        # https://jax.readthedocs.io/en/latest/_autosummary/jax.default_matmul_precision.html
        with jax.default_matmul_precision(dtype_mul):
            for k in range(1, k_max+1):
                t0 = time()
                c = (c @ a).block_until_ready()
                #c = jnp.dot(c, a).block_until_ready()
                t = time() - t0
                s = strftime("%Y-%m-%d %H:%M:%S %Z", localtime())
                print('%s, t=%.3f, #%d, GFLOPS=%5.1f.' % (s, t, k, gflo/t))
    else:
        for k in range(1, k_max+1):
            t0 = time()
            c = (c @ a).block_until_ready()
            #c = jnp.dot(c, a).block_until_ready()
            t = time() - t0
            s = strftime("%Y-%m-%d %H:%M:%S %Z", localtime())
            print('%s, t=%.3f, #%d, GFLOPS=%5.1f.' % (s, t, k, gflo/t))

if __name__ == '__main__':
    config.update("jax_enable_x64", True)  # this only works on startup!
    rnd_key = random.PRNGKey(0)

    print('Available devices: ', jax.devices())
    #config.parse_flags_with_absl()
    param = [int(i) for i in sys.argv[1:]]
    busy_jax_gemm(*param, dtype=jnp.float32, rnd_key=rnd_key)
    # float64, float32, 'tensorfloat32', bfloat16, float16

