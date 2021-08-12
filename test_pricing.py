import numpy as np
from gbm import *
from black_scholes import black_scholes

# Demonstrate sampling gives the same result as black scholes
def test_black_scholes():
    num_samples = 10000
    spot = 2
    time = 0.5
    vol = 0.8
    drift = 0.2
    samples = get_samples(num_samples,spot,time,vol,drift)

    strike = spot * 1.1

    discounted_payoff = np.mean(np.maximum(samples-strike,0) * np.exp(time*drift*-1))

    calced = black_scholes(spot, strike, time, drift, vol)

    assert np.abs(discounted_payoff - calced)/calced < 0.05, (discounted_payoff, calced)



if __name__ == "__main__":
    test_black_scholes()