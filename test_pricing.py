import numpy as np
from gbm import *
from black_scholes import black_scholes
from power_perp import power_perp_price, everlasting_power_perp_price

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

# Demonstrate our power perp pricing is the same as sampling
def test_power_perp():
    num_samples = 100000
    spot = 2
    time = 0.5
    vol = 1.2
    drift = 0.2
    power = 3
    samples = get_samples(num_samples,spot,time,vol,drift)

    discounted_payoff = np.mean(samples**power * np.exp(time*drift*-1))

    calced = power_perp_price(spot, time, vol, drift, power)

    assert np.abs(discounted_payoff - calced)/calced < 0.05, (discounted_payoff, calced)

def test_everlasting_power_perp():
    spot = 2
    vol = 1.2
    drift = 0.2
    power = 3
    funding_period = 1/7

    num_iters = 1000
    est = 0
    for i in range(1,num_iters,1):
        est += power_perp_price(spot, i * funding_period, vol, drift, power) / 2**i

    calced = everlasting_power_perp_price(spot, funding_period, vol, drift, power)

    assert np.abs(est - calced)/est < 0.05, (est, calced)



if __name__ == "__main__":
    test_everlasting_power_perp()

    for period in range(1,100):
        print((period,everlasting_power_perp_price(2,1/period,1.2,0,3)))