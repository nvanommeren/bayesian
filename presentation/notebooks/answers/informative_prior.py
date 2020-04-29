import numpy as np
from matplotlib import pyplot as plt
from scipy.special import binom

# The data
n_heads = 4
n_flips = 10

theta = np.linspace(0, 1, 1001)
prior_distribution = np.where(

    # where theta is between 0.3 and 0.7
    np.logical_and(theta > 0.3, theta < 0.7),

    # set the probability to 0.9 (probability) / 0.4 (interval length)
    0.9 / 0.4,

    # else: set the probability to 0.1 (probability) / 0.3 (interval length)
    0.1 / 0.3
)

# hacky solution to make the plot look nice
prior_distribution[0], prior_distribution[-1] = 0, 0

# plot the prior
plt.plot(theta, prior_distribution)
plt.ylim(0, 3)
plt.axhline(0, color='black')
plt.show()

# Compute the likelihood
likelihood = binom(10, 4) * theta ** n_heads * (1 - theta) ** (n_flips - n_heads)

# Posterior distribution
posterior = likelihood * prior_distribution

# Normalize the posterior distribution (for comparability)
posterior /= np.max(posterior)

_, ax = plt.subplots(figsize=(6,3))
ax.plot(theta, posterior)
ax.set_xlabel("Theta", size=15)
ax.set_ylabel("Posterior value", size=15)
plt.show()
