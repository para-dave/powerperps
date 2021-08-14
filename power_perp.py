import numpy as np


def power_perp_price(spot, time, vol, drift, power):
    return spot ** power * np.exp(
        (power - 1) * (drift + power / 2 * vol ** 2) * time
    )

def everlasting_power_perp_price(spot, funding_period, vol, drift, power):
    return spot ** power * (
            1 / (2* np.exp(-funding_period/2 * (power - 1) * (2 * drift + power * vol **2)) - 1)
    )
