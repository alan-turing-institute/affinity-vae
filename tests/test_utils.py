import pytest

from avae.utils import dims_after_pooling


@pytest.mark.parametrize(
    "n_pools, expected", [(0, 64), (1, 32), (2, 16), (3, 8)]
)
def test_dims_after_pooling_ndim(n_pools, expected):
    """Test dimension calculation after 2x2 pooling op."""
    start = 64
    after_pool = dims_after_pooling(start, n_pools)
    assert after_pool == expected
