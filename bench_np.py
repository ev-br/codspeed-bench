import pytest
import numpy as np
from numpy.testing import assert_allclose

def run_matmul(a, b, c):
    res = np.matmul(a, b, out=c)
    return res


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_gemm(benchmark, n):
    rndm = np.random.RandomState(1234)
    a = np.asarray(rndm.uniform(size=(n, n)), order='F')
    b = np.asarray(rndm.uniform(size=(n, n)), order='F')
    c = np.empty((n, n), dtype=float, order='F')
    result = benchmark(run_matmul, a, b, c)

    # check no copies
    assert result is c

def run_solve(a, b):
    res = np.linalg.solve(a, b)
    return res


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_solve(benchmark, n):
    rndm = np.random.RandomState(1234)
    a = np.eye(n, dtype=float) + rndm.uniform(size=(n,n))
    b = rndm.uniform(size=n)
    result = benchmark(run_solve, a, b)

    assert_allclose(a @ result, b, atol=1e-12)


def run_svd(a):
    res = np.linalg.svd(a)
    return res


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_svd(benchmark, n):
    rndm = np.random.RandomState(1234)
    a = np.random.uniform(size=(n, n))
    u, s, vt = benchmark(run_svd, a)

    assert_allclose(u @ np.diag(s) @ vt, a, atol=1e-12) 
