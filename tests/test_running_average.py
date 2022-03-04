from ..GAAF.utils import RunningAverage

import numpy as np ## For generating test data
import math

def test_mean_one_by_one():
    """
    Generate 1 million normally distributes random samples

    The mean should be close to 0.0

    Feed them one by one
    """
    rng = np.random.default_rng()
    samples = rng.normal(size=1000000)
    
    r_av = RunningAverage()
    for s in samples:
        r_av.update(s)

    ## Three things tested:
    ## - The count is right
    ## - The mean is close to zero (weak test)
    ## - The mean is close to that produced by numpy (stronger)
    assert(r_av.count == 1000000)
    assert(math.isclose(r_av.avg, 0.0, rel_tol=1e-2, abs_tol=1e-2))
    assert(math.isclose(r_av.avg, np.mean(samples)))


def test_mean_five_by_five():
    """
    Generate 1 million normally distributed random samples

    The mean should be close to 0.0

    Feed five at a time
    """
    rng = np.random.default_rng()
    samples = rng.normal(size=1000000)
    
    r_av = RunningAverage()
    for s in samples:
        r_av.update(s, n=5)

    ## Three things tested:
    ## - The count is right
    ## - The mean is close to zero (weak test)
    ## - The mean is close to that produced by numpy (stronger)
    assert(r_av.count == 5000000)
    assert(math.isclose(r_av.avg, 0.0, rel_tol=1e-2, abs_tol=1e-2))
    assert(math.isclose(r_av.avg, np.mean(samples)))
