import numpy as np


def get_sample(spot, time, vol, drift):
    """Computes new price randomly based on price and volatility and updates"""
    rand = np.random.normal(loc=0.0, scale=1.0)

    return spot * np.exp(
        ((drift - (0.5 * (vol ** 2))) * time) + (vol * np.sqrt(time) * rand)
    )


def get_samples(num_samples, spot, time, vol, drift):
    return np.array([get_sample(spot, time, vol, drift) for _ in range(num_samples)])