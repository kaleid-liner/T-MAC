import numpy as np
import os
import tvm
from tvm.autotvm.measure.measure_methods import request_remote
from t_mac.ops import QGeMMLUTBitsCodegen, QGeMMLUTBitsPreprocessorCodegen
from t_mac.weights import preprocess_weights
from t_mac.utils import get_default_device_kwargs, nmse
import logging
from typing import Optional, Tuple


def hvx_preprocess_weights(
    w: np.ndarray,
    scales: np.ndarray,
    zeros: Optional[np.ndarray] = None,
    bits: int = 4,
    g: int = 4,
    tile_p: int = 512,
    tile_q: int = 64,
    vec_p: int = 128,
    vec_q: int = 4,
    vec_c: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:

    assert(w.dtype == "uint8")

    M, K = w.shape

    P = M * bits
    Q = K // g

    # (M, K, bits)
    w = np.stack([(w >> ib) & 1 for ib in range(bits)], axis=-1)
    # (M, K, bits) -> (M, bits, K) -> (M, bits, K) -> (M, bits, K // g, g)
    w = w.transpose(0, 2, 1).reshape(M, bits, Q, g)
    w = sum([(w[:, :, :, ig] << ig) for ig in range(g)])
    # (P // vec_c, vec_c, Q)
    w = w.reshape(M // vec_c, vec_c, bits, Q).transpose(0, 2, 1, 3)
    assert (M // vec_c) % 2 == 0, "M must be divisible by vec_c * 2"
    # Place bits=0 in even bytes, bits=1 in odd bytes, bits=2 in even bytes, etc.
    w = w.reshape(P // vec_c // 2, 2, vec_c, Q).transpose(0, 2, 1, 3)
    w = w.reshape(P // tile_p, tile_p, Q // tile_q, tile_q).transpose(0, 2, 1, 3)
    #             0            1            2                3      4                5
    w = w.reshape(P // tile_p, Q // tile_q, tile_p // vec_p, vec_p, tile_q // vec_q, vec_q).transpose(0, 1, 2, 4, 5, 3)
    # Pack and interleave: q = 0 -> lo_bo, q = 1 -> lo_to, q = 2 -> hi_bo, q = 3 -> hi_to
    # lo -> low 128 bytes, hi -> high 128 bytes, bo -> bot 4 bit in a byte, to -> top 4 bit in a byte
    w = w.reshape(-1, vec_q, vec_p).reshape(-1, vec_q // 2, 2, vec_p).transpose(0, 1, 3, 2)
    w = sum([(w[:, :, :, n] << (n * g)) for n in range(2)])
    w = w.reshape(P // tile_p, Q // tile_q, tile_p // vec_p, tile_q // vec_q, vec_q // 2, vec_p)

    if scales.size >= M:
        group_size = K // scales.shape[1]
        scales = scales.reshape(P // tile_p, tile_p // bits, K // group_size).transpose(0, 2, 1)
        scales = scales.reshape(P // tile_p, K // group_size, tile_p // bits // vec_c, vec_c)
        if zeros is not None:
            zeros = zeros.reshape(P // tile_p, tile_p // bits, K // group_size).transpose(0, 2, 1)
            zeros = zeros.reshape(P // tile_p, K // group_size, tile_p // bits // vec_c, vec_c)
            scales = np.stack([scales, zeros], axis=-2)
        # input size of current TVM API
        scales = scales.reshape(P // tile_p, K // group_size, -1)
    else:
        if zeros is not None:
            scales = np.concatenate([scales, zeros])
    return w, scales


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

bits = 2
M = 4096 * bits
N = 1
K = 4096
zero_point = False
dtype = "int8"
g = 4
group_size = 128
act_group_size = -1
m_groups = 1  # should be -1 or 1 in test_e2e.py

if act_group_size == -1:
    act_group_size = K

device_kwargs = get_default_device_kwargs()

out_dtype = device_kwargs["out_dtype"]

remote_kwargs = None
codegen_kwargs = {
    "save_dir": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out"),
    "dtype": dtype,
    "target": device_kwargs["target"],
    "verify": True,
    "tune": False,
    "remote_kwargs": device_kwargs["remote_kwargs"],
    "bits": bits,
    "out_dtype": out_dtype,
    "act_group_size": act_group_size,
    "cc_opts": device_kwargs["cc_opts"],
}

preprocessor = QGeMMLUTBitsPreprocessorCodegen(name="preprocessor", fast_aggregation_k=0, **codegen_kwargs)
qgemm = QGeMMLUTBitsCodegen(name="qgemm_lut", group_size=group_size, m_groups=m_groups, aggregation_dtype=device_kwargs["aggregation_dtype"], zero_point=zero_point, **codegen_kwargs)

pf, _ = preprocessor.compile(N, K)
qf, _ = qgemm.compile(M, N, K)

bm = qgemm.bm
kfactor = qgemm.kfactor
weight_dtype = qgemm.weight_dtype

# Inputs
Aref = np.random.randint(0, 2 ** bits, size=(M // bits, K)).astype(weight_dtype)
# Aref = np.full_like(Aref, 2)
Zref = None
if m_groups == -1:
    Sref = np.abs(np.random.randn(M // bits, K // group_size).astype(out_dtype))
    if zero_point:
        Zref = np.random.randn(M // bits, K // group_size).astype(out_dtype)
else:
    Sref = np.abs(np.random.randn(m_groups,).astype(out_dtype))
Bref = np.random.randn(N, K).astype(out_dtype)

Bref = np.fromfile("test_data/x.bin", dtype=Bref.dtype).reshape(Bref.shape)
import pdb; pdb.set_trace()
Sref = np.fromfile("test_data/s.bin", dtype=Sref.dtype).reshape(Sref.shape)

# Outputs
if m_groups == -1:
    Adq = Aref.T.reshape(K // group_size, group_size, M // bits).astype(out_dtype) - (2 ** (bits - 1))
    Adq = Adq.transpose(1, 0, 2) * Sref.T
    if zero_point:
        Adq = Adq - Zref.T
    Adq = Adq.transpose(1, 0, 2).reshape(K, M // bits)
else:
    Adq = (Aref.T.astype(out_dtype) - (2 ** (bits - 1))) * Sref[0]

Cref = Bref.dot(Adq)
print(Cref)

dev = tvm.device("llvm")
# TVM Inputs
A_t, Scales_t = preprocess_weights(Aref, Sref, Zref, bits=bits, g=g, bm=bm, kfactor=kfactor)
A_t = tvm.nd.array(A_t, dev)
B_t = tvm.nd.array(Bref, dev)
Scales_t = tvm.nd.array(Scales_t, dev)

# TVM Outputs
C_t = tvm.nd.array(Cref, dev)

# TVM Intermediates
LUT_Scales = tvm.nd.array(np.zeros((N, K // act_group_size), dtype=out_dtype), dev)
LUT_Biases = tvm.nd.array(np.zeros((N, K // act_group_size), dtype=out_dtype), dev)
QLUT = tvm.nd.array(np.zeros((N, K // g, 1 << g), dtype=dtype), dev)

# B_t.numpy().tofile("test_data/x.bin")
HVX_A, HVX_S = hvx_preprocess_weights(Aref, Sref, Zref, bits=bits, g=g, tile_p=M)
# HVX_A.tofile("test_data/w.bin")
# HVX_S.tofile("test_data/s.bin")

pf(B_t, LUT_Scales, LUT_Biases, QLUT)
print(LUT_Scales.numpy())
print(LUT_Biases.numpy())
print(QLUT.numpy())
qf(A_t, QLUT, Scales_t, LUT_Scales, LUT_Biases, C_t)
# C_t.numpy().tofile("test_data0/y.bin")

print(C_t.numpy())

print(nmse(Cref, C_t.numpy()))
