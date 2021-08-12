import numpy as np
from gbm import *

def test_gbm():
    num_samples = 10000
    spot = 2
    time = 0.5
    vol = 0.8
    drift = 0.1
    samples = get_samples(num_samples,spot,time,vol,drift)

    avg = np.mean(samples)
    var = np.var(samples)

    expected_avg = spot*np.exp(drift*time)
    expected_var = (spot**2)*np.exp(2*drift*time)*(np.exp(vol**2*time) - 1)

    assert np.abs(avg - expected_avg)/expected_avg < 0.05, (avg, expected_avg)
    assert np.abs(var - expected_var)/expected_var < 0.05, (var, expected_var)


if __name__ == "__main__":
    test_gbm()