import pytest
import numpy as np
from scipy.linalg.blas import dgemm


def run_gemm(a, b, c):
    alpha = 1.0
    res = dgemm(alpha, a, b, c=c, overwrite_c=True)
    return res


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_gemm(benchmark, n):
    a = np.eye(n, dtype=float, order='F')
    b = np.eye(n, dtype=float, order='F')
    c = np.empty((n, n), dtype=float, order='F')
    result = benchmark(run_gemm, a, b, c)
    assert result is c

